import argparse
import json
import select
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
import rerun as rr
from loop_rate_limiters import RateLimiter
from motion_tools.robot_gui import ReRunRobot

from rs_imle_policy.configs.g1_configs import G1IKConfig
from rs_imle_policy.g1_arm_ik import G1ReducedPinkIK


CONTROL_DT = 1 / 200
MIRROR_Y = np.diag([1.0, -1.0, 1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pink-only synchronized dual-hand waypoint recorder for G1.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/inference/g1_sim/startup_waypoints.json"),
        help="JSON file used to save recorded waypoints.",
    )
    parser.add_argument(
        "--driver",
        choices=("left", "right"),
        default="left",
        help="Hand target that drives the mirrored synchronized target.",
    )
    parser.add_argument("--side-x", type=float, default=0.18, help="Default side pose x position in meters.")
    parser.add_argument("--side-y", type=float, default=0.30, help="Default side pose lateral distance in meters.")
    parser.add_argument("--side-z", type=float, default=-0.05, help="Default side pose z position in meters.")
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Disable ReRun logging and only use the Pink visualizer.",
    )
    return parser.parse_args()


def clone_se3(transform: pin.SE3) -> pin.SE3:
    return pin.SE3(transform.rotation.copy(), transform.translation.copy())


def mirror_se3(transform: pin.SE3) -> pin.SE3:
    return pin.SE3(MIRROR_Y @ transform.rotation @ MIRROR_Y, MIRROR_Y @ transform.translation)


def make_side_target(base: pin.SE3, side_sign: float, x: float, y: float, z: float) -> pin.SE3:
    target = clone_se3(base)
    target.translation = np.array([x, side_sign * abs(y), z], dtype=float)
    return target


def se3_to_dict(transform: pin.SE3) -> dict[str, list[list[float]] | list[float]]:
    return {
        "translation": np.asarray(transform.translation, dtype=float).round(6).tolist(),
        "rotation": np.asarray(transform.rotation, dtype=float).round(6).tolist(),
        "matrix": np.asarray(transform.homogeneous, dtype=float).round(6).tolist(),
    }


def save_waypoints(path: Path, driver: str, waypoints: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "driver": driver,
        "waypoints": waypoints,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {len(waypoints)} waypoint(s) to {path}")


def maybe_read_command() -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None
    return sys.stdin.readline().strip().lower()


def setup_rerun(enabled: bool):
    if not enabled:
        return None, None, None

    rec = rr.RecordingStream("g1_sync_waypoint_recorder")
    rec.spawn()
    robot_gui = ReRunRobot.g1(rec, target_frame="pelvis")
    robot_sol = ReRunRobot.g1_debug(rec, target_frame="pelvis")
    robot_sol.apply_color([0, 1, 0, 0.5])
    return rec, robot_gui, robot_sol


def main() -> None:
    args = parse_args()

    ik = G1ReducedPinkIK(
        config=G1IKConfig(box_barrier_gain=0),
        visualize=True,
        spawn_visualizer=True,
    )
    rec, robot_gui, robot_sol = setup_rerun(not args.no_rerun)

    initial_targets = ik.get_targets_from_configuration()
    left_target = make_side_target(initial_targets.left, side_sign=1.0, x=args.side_x, y=args.side_y, z=args.side_z)
    right_target = mirror_se3(left_target)
    if args.driver == "right":
        right_target = make_side_target(
            initial_targets.right, side_sign=-1.0, x=args.side_x, y=args.side_y, z=args.side_z
        )
        left_target = mirror_se3(right_target)
    ik.set_targets(left=left_target, right=right_target)

    print("Interactive commands: [enter]/r record, p print, u undo, s save, q save+quit")
    print(f"Driver hand: {args.driver}. Move only the {args.driver} handle in the visualizer.")

    rate = RateLimiter(int(1 / CONTROL_DT), warn=False)
    waypoints: list[dict] = []
    full_q = np.zeros(29)

    while True:
        targets = ik.get_targets()
        driver_target = clone_se3(targets.left if args.driver == "left" else targets.right)
        synced_target = mirror_se3(driver_target)

        if args.driver == "left":
            left_target = driver_target
            right_target = synced_target
        else:
            left_target = synced_target
            right_target = driver_target

        ik.set_targets(left=left_target, right=right_target)
        q = ik.solve(dt=CONTROL_DT)
        ik.configuration.update(q.copy())
        ik.viz.display(q)

        if rec is not None:
            full_q[:] = 0.0
            full_q[15:29] = q
            robot_sol.log(full_q)
            robot_sol.log_pin_transform("left_ee", left_target)
            robot_sol.log_pin_transform("right_ee", right_target)
            robot_gui.log(full_q.copy())
            rec.log("state/arm_q", rr.Scalars(q.copy()))

        command = maybe_read_command()
        if command in {"", "r"}:
            waypoint = {
                "index": len(waypoints),
                "left": se3_to_dict(left_target),
                "right": se3_to_dict(right_target),
            }
            waypoints.append(waypoint)
            print(json.dumps(waypoint, indent=2))
        elif command == "p":
            print(json.dumps({"driver": args.driver, "waypoints": waypoints}, indent=2))
        elif command == "u":
            if waypoints:
                removed = waypoints.pop()
                print(f"Removed waypoint {removed['index']}")
            else:
                print("No waypoints to remove.")
        elif command == "s":
            save_waypoints(args.output, args.driver, waypoints)
        elif command == "q":
            save_waypoints(args.output, args.driver, waypoints)
            break

        rate.sleep()


if __name__ == "__main__":
    main()
