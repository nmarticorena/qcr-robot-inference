"""Controller interface for the Gear Sonic ZMQ locomotion planner.

Typical use::

    with GearSonicInterface() as controller:
        controller.set_velocity(PlanarVelocity(vx=0.3, omega=0.2))

The public velocity command is a body-frame planar twist. Gear Sonic accepts an
absolute facing direction rather than angular velocity, so the publishing loop
integrates ``omega`` to produce that facing direction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
import json
import math
import struct
import threading
import time
from typing import Any

import zmq

from rs_imle_policy.configs.g1.locomotion import GearSonicConfig
from rs_imle_policy.controllers.g1.locomotion.commands import PlanarVelocity

HEADER_SIZE = 1280
# Left wrist, right wrist, and neck targets for Gear Sonic's default G1 pose.
DEFAULT_VR_POSITIONS = (
    0.379777,
    0.162449,
    0.095733,
    0.379777,
    -0.162439,
    0.095733,
    0.0,
    0.0,
    0.4,
)
DEFAULT_VR_ORIENTATIONS = (
    0.995001,
    0.099863,
    0.000018,
    -0.000098,
    0.995001,
    -0.099863,
    0.000018,
    0.000098,
    1.0,
    0.0,
    0.0,
    0.0,
)
_ZERO_TOLERANCE = 1e-6
PLANNER_WAYPOINT_COUNT = 4


class LocomotionMode(IntEnum):
    """Locomotion modes understood by the Gear Sonic planner."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19
    LEDGE_WALKING = 20
    OBJECT_CARRYING = 21
    STEALTH_WALK_2 = 22
    HAPPY_DANCE_WALK = 23
    ZOMBIE_WALK = 24
    GUN_WALK = 25
    SCARE_WALK = 26


@dataclass(frozen=True)
class PlannerWaypoints:
    """One SONIC planner token of world-frame root targets.

    The planner ONNX model requires exactly four consecutive 30 Hz target
    positions and headings. Positions use MuJoCo's Z-up world frame and are
    flattened in frame-major order: ``(x0, y0, z0, ..., x3, y3, z3)``.
    """

    positions: tuple[float, ...]
    headings: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.positions) != PLANNER_WAYPOINT_COUNT * 3:
            raise ValueError("positions must contain four xyz targets (12 values)")
        if len(self.headings) != PLANNER_WAYPOINT_COUNT:
            raise ValueError("headings must contain four yaw targets")
        if not all(math.isfinite(value) for value in (*self.positions, *self.headings)):
            raise ValueError("waypoint values must be finite")

    @classmethod
    def from_frames(
        cls,
        positions: Sequence[Sequence[float]],
        headings: Sequence[float],
    ) -> PlannerWaypoints:
        """Build targets from four ``(x, y, z)`` frames and four yaws."""

        if len(positions) != PLANNER_WAYPOINT_COUNT:
            raise ValueError("positions must contain exactly four frames")
        flattened: list[float] = []
        for position in positions:
            if len(position) != 3:
                raise ValueError("each waypoint position must contain x, y, z")
            flattened.extend(float(value) for value in position)
        return cls(
            positions=tuple(flattened),
            headings=tuple(float(value) for value in headings),
        )


@dataclass(frozen=True)
class PlannerCommand:
    """High-level world-frame command consumed by SONIC's planner."""

    mode: LocomotionMode
    movement: tuple[float, float, float]
    facing: tuple[float, float, float]
    speed: float = -1.0
    height: float = -1.0
    waypoints: PlannerWaypoints | None = None

    def __post_init__(self) -> None:
        if len(self.movement) != 3 or len(self.facing) != 3:
            raise ValueError("movement and facing must contain three values")
        values = (*self.movement, *self.facing, self.speed, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("planner command values must be finite")
        if math.hypot(self.facing[0], self.facing[1]) <= _ZERO_TOLERANCE:
            raise ValueError("facing must have a non-zero horizontal component")


class PackedPublisher:
    """Encode and publish Gear Sonic's packed ZMQ messages.

    A ZeroMQ socket must only be used by the thread that created it. The
    high-level :class:`GearSonicInterface` therefore owns this publisher inside
    its publishing thread.
    """

    def __init__(self, endpoint: str) -> None:
        self.context = zmq.Context()
        # XPUB is wire-compatible with ordinary SUB clients, but also exposes
        # their subscriptions.  Waiting for the command and planner
        # subscriptions avoids losing the pulse-like start command to ZeroMQ's
        # PUB/SUB slow-joiner behaviour.
        self.socket = self.context.socket(zmq.XPUB)
        self.socket.setsockopt(zmq.SNDHWM, 3)
        self.socket.setsockopt(zmq.LINGER, 0)
        try:
            self.socket.bind(endpoint)
        except BaseException:
            self.socket.close()
            self.context.term()
            raise
        self._closed = False

    def wait_for_subscribers(
        self,
        topics: Sequence[str],
        timeout: float,
        stop_event: threading.Event | None = None,
    ) -> bool:
        """Wait until subscribers for all requested topic prefixes connect."""

        if self._closed:
            raise RuntimeError("publisher is closed")

        required = {topic.encode("ascii") for topic in topics}
        if not required:
            return True

        subscribed: set[bytes] = set()
        deadline = time.monotonic() + timeout
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while not required.issubset(subscribed):
            if stop_event is not None and stop_event.is_set():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False

            events = dict(poller.poll(max(1, min(100, math.ceil(remaining * 1000.0)))))
            if not events.get(self.socket, 0) & zmq.POLLIN:
                continue

            notification = self.socket.recv()
            if not notification or notification[0] == 0:
                continue
            prefix = notification[1:]
            # An empty subscription accepts every topic.  Otherwise a ZMQ
            # subscription is a prefix filter, not necessarily an exact name.
            subscribed.update(topic for topic in required if topic.startswith(prefix))
        return True

    def send(self, topic: str, fields: list[dict[str, Any]], data: bytes) -> None:
        if self._closed:
            raise RuntimeError("publisher is closed")

        header = json.dumps(
            {"v": 1, "endian": "le", "count": 1, "fields": fields},
            separators=(",", ":"),
        ).encode("utf-8")
        if len(header) > HEADER_SIZE:
            raise ValueError(f"ZMQ header is {len(header)} bytes; limit is {HEADER_SIZE}")
        self.socket.send(topic.encode("ascii") + header.ljust(HEADER_SIZE, b"\0") + data)

    def command(self, *, start: bool = False, stop: bool = False) -> None:
        """Pulse start/stop and select planner mode."""

        fields = [
            {"name": "start", "dtype": "u8", "shape": [1]},
            {"name": "stop", "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ]
        self.send("command", fields, struct.pack("<BBB", start, stop, True))

    def planner(
        self,
        *,
        mode: int | LocomotionMode,
        movement: tuple[float, float, float],
        facing: tuple[float, float, float],
        speed: float,
        height: float = -1.0,
        waypoints: PlannerWaypoints | None = None,
        vr_positions: tuple[float, ...] | None = None,
        vr_orientations: tuple[float, ...] | None = None,
    ) -> None:
        """Publish one planner command in the documented packed format."""

        if (vr_positions is None) != (vr_orientations is None):
            raise ValueError("vr_positions and vr_orientations must either both be set or both be None")
        if vr_positions is not None and len(vr_positions) != 9:
            raise ValueError("vr_positions must contain 9 values")
        if vr_orientations is not None and len(vr_orientations) != 12:
            raise ValueError("vr_orientations must contain 12 values")

        fields = [
            {"name": "mode", "dtype": "i32", "shape": [1]},
            {"name": "movement", "dtype": "f32", "shape": [3]},
            {"name": "facing", "dtype": "f32", "shape": [3]},
            {"name": "speed", "dtype": "f32", "shape": [1]},
            {"name": "height", "dtype": "f32", "shape": [1]},
        ]
        values: list[float | int] = [int(mode), *movement, *facing, speed, height]
        fmt = "<i3f3fff"
        if waypoints is not None:
            fields.extend(
                [
                    {
                        "name": "specific_target_positions",
                        "dtype": "f32",
                        "shape": [PLANNER_WAYPOINT_COUNT, 3],
                    },
                    {
                        "name": "specific_target_headings",
                        "dtype": "f32",
                        "shape": [PLANNER_WAYPOINT_COUNT],
                    },
                ]
            )
            values.extend(waypoints.positions)
            values.extend(waypoints.headings)
            fmt += "12f4f"
        if vr_positions is not None and vr_orientations is not None:
            fields.extend(
                [
                    {"name": "vr_position", "dtype": "f32", "shape": [9]},
                    {"name": "vr_orientation", "dtype": "f32", "shape": [12]},
                ]
            )
            values.extend(vr_positions)
            values.extend(vr_orientations)
            fmt += "9f12f"
        self.send("planner", fields, struct.pack(fmt, *values))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.socket.close()
        self.context.term()


class GearSonicInterface:
    """Continuously publish planar velocity commands to Gear Sonic.

    Configuration is supplied as one :class:`GearSonicConfig` so protocol and
    lifecycle tuning stay together.

    ``vx`` and ``vy`` passed to :meth:`set_velocity` are in the robot body
    frame, in m/s. ``omega`` is the desired facing-yaw rate in rad/s.
    """

    def __init__(
        self,
        config: GearSonicConfig | None = None,
    ) -> None:
        config = config or GearSonicConfig()
        endpoint = config.endpoint
        publish_hz = config.publish_hz
        locomotion_mode = config.locomotion_mode
        height = config.height
        connection_timeout = config.connection_timeout
        connection_delay = config.connection_delay
        command_repetitions = config.command_repetitions
        if not endpoint:
            raise ValueError("endpoint must not be empty")
        if not math.isfinite(publish_hz) or publish_hz <= 1.0:
            raise ValueError("publish_hz must be finite and greater than the manager's 1 Hz timeout")
        if not math.isfinite(height):
            raise ValueError("height must be finite")
        if not math.isfinite(connection_timeout) or connection_timeout <= 0.0:
            raise ValueError("connection_timeout must be finite and positive")
        if not math.isfinite(connection_delay) or connection_delay < 0.0:
            raise ValueError("connection_delay must be finite and non-negative")
        if command_repetitions < 1:
            raise ValueError("command_repetitions must be at least one")

        self.endpoint = endpoint
        self.publish_hz = float(publish_hz)
        self.height = float(height)
        self.connection_timeout = float(connection_timeout)
        self.connection_delay = float(connection_delay)
        self.command_repetitions = int(command_repetitions)
        self._locomotion_mode = None if locomotion_mode is None else LocomotionMode(locomotion_mode)

        self._state_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._velocity = (0.0, 0.0, 0.0)
        self._facing_yaw = 0.0
        self._direct_command: PlannerCommand | None = None
        self._vr_positions: tuple[float, ...] | None = None
        self._vr_orientations: tuple[float, ...] | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._running = False
        self._closed = False
        self._background_error: BaseException | None = None

    @property
    def is_running(self) -> bool:
        """Whether the publishing thread is active."""

        with self._lifecycle_lock:
            return self._running

    def start(self) -> None:
        """Bind the publisher, select planner mode, and start command streaming."""

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("interface is closed")
            if self._running:
                return

            self._velocity = (0.0, 0.0, 0.0)
            self._facing_yaw = 0.0
            self._direct_command = None
            self._vr_positions = None
            self._vr_orientations = None
            self._background_error = None
            self._stop_event = threading.Event()
            self._started_event = threading.Event()
            self._running = True
            self._thread = threading.Thread(
                target=self._publish_loop,
                name="gear-sonic-publisher",
                daemon=False,
            )
            self._thread.start()
            started_event = self._started_event

        startup_timeout = (
            self.connection_timeout + self.connection_delay + self.command_repetitions / self.publish_hz + 5.0
        )
        if not started_event.wait(startup_timeout):
            self.stop()
            raise TimeoutError("timed out while starting the Gear Sonic publisher")

        self._raise_background_error()

    def set_velocity(self, command: PlanarVelocity) -> None:
        """Set the body-frame planar velocity command.

        The latest command is retained and published continuously until it is
        replaced. Send ``PlanarVelocity()`` to remain under control
        while commanding idle; call :meth:`stop` to leave controller mode.
        """

        self._require_running()
        with self._state_lock:
            self._velocity = command.as_tuple()
            self._direct_command = None

    def set_planner_command(self, command: PlannerCommand) -> None:
        """Retain a direct world-frame planner command until it is replaced.

        This bypasses the body-twist adapter used by :meth:`set_velocity`.
        A command containing :class:`PlannerWaypoints` requires the matching
        C++ ZMQ waypoint extension; unmodified GEAR-SONIC deployments ignore
        those two optional wire fields.
        """

        if not isinstance(command, PlannerCommand):
            raise TypeError("command must be a PlannerCommand")
        self._require_running()
        with self._state_lock:
            self._direct_command = command

    def set_vr_poses(
        self,
        positions: Sequence[float],
        orientations: Sequence[float],
    ) -> None:
        """Attach waist-frame left, right, and head poses to planner commands.

        ``positions`` contains three consecutive xyz vectors and
        ``orientations`` contains three consecutive wxyz quaternions. Their
        order is left wrist, right wrist, then head. The latest poses are
        retained and included in every planner update until
        :meth:`clear_vr_poses` is called.
        """

        vr_positions = tuple(float(value) for value in positions)
        vr_orientations = tuple(float(value) for value in orientations)
        if len(vr_positions) != 9:
            raise ValueError("positions must contain three xyz poses (9 values)")
        if len(vr_orientations) != 12:
            raise ValueError("orientations must contain three wxyz quaternions (12 values)")
        if not all(math.isfinite(value) for value in (*vr_positions, *vr_orientations)):
            raise ValueError("VR pose values must be finite")
        for index in range(0, len(vr_orientations), 4):
            norm = math.sqrt(sum(value * value for value in vr_orientations[index : index + 4]))
            if norm <= _ZERO_TOLERANCE:
                raise ValueError("VR orientation quaternions must be non-zero")

        self._require_running()
        with self._state_lock:
            self._vr_positions = vr_positions
            self._vr_orientations = vr_orientations

    def clear_vr_poses(self) -> None:
        """Stop including VR pose fields in planner commands."""

        self._require_running()
        with self._state_lock:
            self._vr_positions = None
            self._vr_orientations = None

    def clear_waypoints(self, *, facing_yaw: float = 0.0) -> None:
        """Clear a waypoint horizon and command the planner's idle mode."""

        yaw = float(facing_yaw)
        if not math.isfinite(yaw):
            raise ValueError("facing_yaw must be finite")
        self.set_planner_command(
            PlannerCommand(
                mode=LocomotionMode.IDLE,
                movement=(0.0, 0.0, 0.0),
                facing=(math.cos(yaw), math.sin(yaw), 0.0),
                speed=-1.0,
                height=self.height,
            )
        )

    def set_locomotion_mode(self, mode: LocomotionMode | None) -> None:
        """Select a fixed locomotion mode, or ``None`` for speed-based selection."""

        with self._state_lock:
            self._locomotion_mode = mode

    def stop(self) -> None:
        """Command idle, pulse stop, and stop the publishing thread."""

        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_event.set()

        if thread is not threading.current_thread():
            thread.join(timeout=self.command_repetitions / self.publish_hz + 5.0)
            if thread.is_alive():
                raise TimeoutError("timed out while stopping the Gear Sonic publisher")

        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None
                self._running = False
        self._raise_background_error()

    def close(self) -> None:
        """Stop the interface and prevent it from being entered again."""

        with self._lifecycle_lock:
            self._closed = True
        self.stop()

    def __enter__(self) -> GearSonicInterface:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _require_running(self) -> None:
        with self._lifecycle_lock:
            if not self._running:
                raise RuntimeError("call start() before setting velocity")
        self._raise_background_error()

    def _raise_background_error(self) -> None:
        with self._lifecycle_lock:
            error = self._background_error
        if error is not None:
            raise RuntimeError("Gear Sonic publishing thread failed") from error

    @staticmethod
    def _automatic_mode(speed: float) -> LocomotionMode:
        if speed <= 0.8:
            return LocomotionMode.SLOW_WALK
        if speed <= 2.5:
            return LocomotionMode.WALK
        return LocomotionMode.RUN

    def _planner_command(self, dt: float) -> PlannerCommand:
        with self._state_lock:
            direct_command = self._direct_command
            if direct_command is not None:
                return direct_command
            vx, vy, omega = self._velocity
            self._facing_yaw = math.remainder(
                self._facing_yaw + omega * dt,
                2.0 * math.pi,
            )
            yaw = self._facing_yaw
            fixed_mode = self._locomotion_mode

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        movement_x = cos_yaw * vx - sin_yaw * vy
        movement_y = sin_yaw * vx + cos_yaw * vy
        speed = math.hypot(vx, vy)

        if speed > _ZERO_TOLERANCE:
            movement = (movement_x / speed, movement_y / speed, 0.0)
        else:
            movement = (0.0, 0.0, 0.0)

        facing = (cos_yaw, sin_yaw, 0.0)
        if speed <= _ZERO_TOLERANCE and abs(omega) <= _ZERO_TOLERANCE:
            mode = LocomotionMode.IDLE
        else:
            mode = fixed_mode or self._automatic_mode(speed)
        return PlannerCommand(
            mode=mode,
            movement=movement,
            facing=facing,
            speed=speed,
            height=self.height,
        )

    def _send_planner(self, publisher: PackedPublisher, dt: float) -> None:
        command = self._planner_command(dt)
        with self._state_lock:
            vr_positions = self._vr_positions
            vr_orientations = self._vr_orientations
        publisher.planner(
            mode=command.mode,
            movement=command.movement,
            facing=command.facing,
            speed=command.speed,
            height=command.height,
            waypoints=command.waypoints,
            vr_positions=vr_positions,
            vr_orientations=vr_orientations,
        )

    def _publish_loop(self) -> None:
        publisher: PackedPublisher | None = None
        start_sent = False
        period = 1.0 / self.publish_hz
        try:
            publisher = PackedPublisher(self.endpoint)
            connected = publisher.wait_for_subscribers(
                ("command", "planner"),
                self.connection_timeout,
                self._stop_event,
            )
            if not connected:
                if self._stop_event.is_set():
                    return
                raise TimeoutError(
                    f"timed out waiting for GEAR-SONIC command and planner subscribers at {self.endpoint}"
                )
            if self._stop_event.wait(self.connection_delay):
                return

            # Prime the planner stream before the pulse, then repeat the pulse
            # because PUB/SUB may discard early messages while connecting.
            for _ in range(self.command_repetitions):
                self._send_planner(publisher, 0.0)
                publisher.command(start=True)
                start_sent = True
                if self._stop_event.wait(period):
                    return

            self._started_event.set()
            last_update = time.monotonic()
            next_send = last_update
            while not self._stop_event.is_set():
                now = time.monotonic()
                dt = max(0.0, min(now - last_update, 1.0))
                last_update = now
                self._send_planner(publisher, dt)

                next_send += period
                wait_time = next_send - time.monotonic()
                if wait_time <= 0.0:
                    next_send = time.monotonic()
                    continue
                self._stop_event.wait(wait_time)
        except BaseException as error:
            with self._lifecycle_lock:
                self._background_error = error
        finally:
            if publisher is not None and start_sent:
                try:
                    with self._state_lock:
                        self._velocity = (0.0, 0.0, 0.0)
                        self._direct_command = None
                        self._vr_positions = None
                        self._vr_orientations = None
                    for _ in range(self.command_repetitions):
                        self._send_planner(publisher, 0.0)
                        publisher.command(stop=True)
                        time.sleep(period)
                except BaseException as error:
                    with self._lifecycle_lock:
                        if self._background_error is None:
                            self._background_error = error
            if publisher is not None:
                publisher.close()
            with self._lifecycle_lock:
                self._running = False
            self._started_event.set()


__all__ = [
    "DEFAULT_VR_ORIENTATIONS",
    "DEFAULT_VR_POSITIONS",
    "GearSonicInterface",
    "LocomotionMode",
    "PlannerCommand",
    "PlannerWaypoints",
]
