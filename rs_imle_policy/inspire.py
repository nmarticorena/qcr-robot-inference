import numpy as np
import threading
from multiprocessing import Process, Array
import time
from enum import IntEnum

import logging_mp
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber  # dds

from inspire_sdkpy import inspire_dds
import inspire_sdkpy.inspire_hand_defaut as inspire_hand_default

logger_mp = logging_mp.getLogger(__name__)

kTopicInspireFTPLeftCommand = "rt/inspire_hand/ctrl/l"
kTopicInspireFTPRightCommand = "rt/inspire_hand/ctrl/r"
kTopicInspireFTPLeftState = "rt/inspire_hand/state/l"
kTopicInspireFTPRightState = "rt/inspire_hand/state/r"

Inspire_Num_Motors = 6


class Inspire_Controller_FTP:
    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
    ):
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        # Initialize hand command publishers
        self.LeftHandCmd_publisher = ChannelPublisher(kTopicInspireFTPLeftCommand, inspire_dds.inspire_hand_ctrl)
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(kTopicInspireFTPRightCommand, inspire_dds.inspire_hand_ctrl)
        self.RightHandCmd_publisher.Init()

        # Initialize hand state subscribers
        self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireFTPLeftState, inspire_dds.inspire_hand_state)
        self.LeftHandState_subscriber.Init()  # Consider using callback if preferred: Init(callback_func, period_ms)
        self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireFTPRightState, inspire_dds.inspire_hand_state)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states ([0,1] normalized values)
        self.left_hand_state_array = Array("d", Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", Inspire_Num_Motors, lock=True)

        # Initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # Wait for initial DDS messages (optional, but good for ensuring connection)
        wait_count = 0
        while not (any(self.left_hand_state_array) or any(self.right_hand_state_array)):
            if wait_count % 100 == 0:  # Print every second
                logger_mp.info(
                    f"[Inspire_Controller_FTP] Waiting to subscribe to hand states from DDS (L: {any(self.left_hand_state_array)}, R: {any(self.right_hand_state_array)})..."
                )
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 500:  # Timeout after 5 seconds
                logger_mp.warning(
                    "[Inspire_Controller_FTP] Warning: Timeout waiting for initial hand states. Proceeding anyway."
                )
                break
        logger_mp.info("[Inspire_Controller_FTP] Initial hand states received or timeout.")

        hand_control_process = Process(
            target=self.control_process,
            args=(
                left_hand_array,
                right_hand_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
            ),
        )
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Inspire_Controller_FTP OK!\n")

    def _subscribe_hand_state(self):
        logger_mp.info("[Inspire_Controller_FTP] Subscribe thread started.")
        while True:
            # Left Hand
            left_state_msg = self.LeftHandState_subscriber.Read()
            if left_state_msg is not None:
                if hasattr(left_state_msg, "angle_act") and len(left_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.left_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.left_hand_state_array[i] = left_state_msg.angle_act[i] / 1000.0
                else:
                    logger_mp.warning(
                        f"[Inspire_Controller_FTP] Received left_state_msg but attributes are missing or incorrect. Type: {type(left_state_msg)}, Content: {str(left_state_msg)[:100]}"
                    )
            # Right Hand
            right_state_msg = self.RightHandState_subscriber.Read()
            if right_state_msg is not None:
                if hasattr(right_state_msg, "angle_act") and len(right_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.right_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.right_hand_state_array[i] = right_state_msg.angle_act[i] / 1000.0
                else:
                    logger_mp.warning(
                        f"[Inspire_Controller_FTP] Received right_state_msg but attributes are missing or incorrect. Type: {type(right_state_msg)}, Content: {str(right_state_msg)[:100]}"
                    )

            time.sleep(0.002)

    def _send_hand_command(self, left_angle_cmd_scaled, right_angle_cmd_scaled):
        """
        Send scaled angle commands [0-1000] to both hands.
        """
        # Left Hand Command
        left_cmd_msg = inspire_hand_default.get_inspire_hand_ctrl()
        left_cmd_msg.angle_set = left_angle_cmd_scaled
        left_cmd_msg.mode = 0b0001  # Mode 1: Angle control
        self.LeftHandCmd_publisher.Write(left_cmd_msg)

        # Right Hand Command
        right_cmd_msg = inspire_hand_default.get_inspire_hand_ctrl()
        right_cmd_msg.angle_set = right_angle_cmd_scaled
        right_cmd_msg.mode = 0b0001  # Mode 1: Angle control
        self.RightHandCmd_publisher.Write(right_cmd_msg)

        # Temporarily open the first N logs.
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0
        if self._debug_count < 50:
            logger_mp.info(
                f"[Inspire_Controller_FTP] Publish cmd L={left_angle_cmd_scaled} R={right_angle_cmd_scaled} "
            )
            self._debug_count += 1

    def dummy_hand(
        self,
        left_state: float,
        right_state: float,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        start_time = time.time()

        left_q_target = np.full(Inspire_Num_Motors, left_state)
        right_q_target = np.full(Inspire_Num_Motors, right_state)
        self._send_hand_command(left_q_target, right_q_target)
        current_time = time.time()
        time_elapsed = current_time - start_time
        sleep_time = max(0, (1 / self.fps) - time_elapsed)
        time.sleep(sleep_time)

    def policy_to_hand_command(self, left_policy_output, right_policy_output):
        """
        left/right_policy_output: np.array of shape (6,) in radians,
        same joint ordering as the collected data.
        """

        def normalize(val, min_val, max_val):
            return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

        angle_ranges = [
            (0.0, 1.7),  # idx 0
            (0.0, 1.7),  # idx 1
            (0.0, 1.7),  # idx 2
            (0.0, 1.7),  # idx 3
            (0.0, 0.5),  # idx 4
            (-0.1, 1.3),  # idx 5
        ]

        def process(q):
            normalized = np.array([normalize(q[i], *angle_ranges[i]) for i in range(Inspire_Num_Motors)])
            scaled = [int(np.clip(v * 1000, 0, 1000)) for v in normalized]
            return scaled

        scaled_left = process(left_policy_output)
        scaled_right = process(right_policy_output)
        self._send_hand_command(scaled_left, scaled_right)

    def control_process(
        self,
        left_hand_array,
        right_hand_array,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        logger_mp.info("[Inspire_Controller_FTP] Control process started.")
        self.running = True

        left_q_target = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        try:
            while self.running:
                start_time = time.time()
                # get dual hand state
                with left_hand_array.get_lock():
                    left_hand_data = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if not np.all(right_hand_data == 0.0) and not np.all(
                    left_hand_data[4] == np.array([-1.13, 0.3, 0.15])
                ):  # if hand data has been initialized.
                    ref_left_value = (
                        left_hand_data[self.hand_retargeting.left_indices[1, :]]
                        - left_hand_data[self.hand_retargeting.left_indices[0, :]]
                    )
                    ref_right_value = (
                        right_hand_data[self.hand_retargeting.right_indices[1, :]]
                        - right_hand_data[self.hand_retargeting.right_indices[0, :]]
                    )

                    left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[
                        self.hand_retargeting.left_dex_retargeting_to_hardware
                    ]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[
                        self.hand_retargeting.right_dex_retargeting_to_hardware
                    ]

                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(Inspire_Num_Motors):
                        if idx <= 3:
                            left_q_target[idx] = normalize(left_q_target[idx], 0.0, 1.7)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target[idx] = normalize(left_q_target[idx], 0.0, 0.5)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
                        elif idx == 5:
                            left_q_target[idx] = normalize(left_q_target[idx], -0.1, 1.3)
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)

                scaled_left_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in left_q_target]
                scaled_right_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in right_q_target]

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self._send_hand_command(scaled_left_cmd, scaled_right_cmd)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller_FTP has been closed.")


# Update hand state, according to the official documentation:
# 1. https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
# 2. https://support.unitree.com/home/en/G1_developer/inspire_ftp_dexterity_hand
# the state sequence is as shown in the table below
# ┌──────┬───────┬──────┬────────┬────────┬────────────┬────────────────┬───────┬──────┬────────┬────────┬────────────┬────────────────┐
# │ Id   │   0   │  1   │   2    │   3    │     4      │       5        │   6   │  7   │   8    │   9    │    10      │       11       │
# ├──────┼───────┼──────┼────────┼────────┼────────────┼────────────────┼───────┼──────┼────────┼────────┼────────────┼────────────────┤
# │      │                    Right Hand                                │                   Left Hand                                  │
# │Joint │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │
# └──────┴───────┴──────┴────────┴────────┴────────────┴────────────────┴───────┴──────┴────────┴────────┴────────────┴────────────────┘
class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5


class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11


INSPIRE_FTP_JOINT_MAP: dict[int, str] = {
    # ── Right hand ──────────────────────────────────────────
    Inspire_Right_Hand_JointIndex.kRightHandPinky: "right_little_1_joint",
    Inspire_Right_Hand_JointIndex.kRightHandRing: "right_ring_1_joint",
    Inspire_Right_Hand_JointIndex.kRightHandMiddle: "right_middle_1_joint",
    Inspire_Right_Hand_JointIndex.kRightHandIndex: "right_index_1_joint",
    Inspire_Right_Hand_JointIndex.kRightHandThumbBend: "right_thumb_2_joint",
    Inspire_Right_Hand_JointIndex.kRightHandThumbRotation: "right_thumb_1_joint",
    # ── Left hand ───────────────────────────────────────────
    Inspire_Left_Hand_JointIndex.kLeftHandPinky: "left_little_1_joint",
    Inspire_Left_Hand_JointIndex.kLeftHandRing: "left_ring_1_joint",
    Inspire_Left_Hand_JointIndex.kLeftHandMiddle: "left_middle_1_joint",
    Inspire_Left_Hand_JointIndex.kLeftHandIndex: "left_index_1_joint",
    Inspire_Left_Hand_JointIndex.kLeftHandThumbBend: "left_thumb_2_joint",
    Inspire_Left_Hand_JointIndex.kLeftHandThumbRotation: "left_thumb_1_joint",
}


INSPIRE_FTP_LEFT_JOINT_MAP: dict[int, str] = {
    Inspire_Left_Hand_JointIndex.kLeftHandPinky - 6: "left_little_1_joint",
    Inspire_Left_Hand_JointIndex.kLeftHandRing - 6: "left_ring_1_joint",
    Inspire_Left_Hand_JointIndex.kLeftHandMiddle - 6: "left_middle_1_joint",
    Inspire_Left_Hand_JointIndex.kLeftHandIndex - 6: "left_index_1_joint",
    Inspire_Left_Hand_JointIndex.kLeftHandThumbBend - 6: "left_thumb_2_joint",
    Inspire_Left_Hand_JointIndex.kLeftHandThumbRotation - 6: "left_thumb_1_joint",
}

INSPIRE_FTP_RIGHT_JOINT_MAP: dict[int, str] = {
    Inspire_Right_Hand_JointIndex.kRightHandPinky: "right_little_1_joint",
    Inspire_Right_Hand_JointIndex.kRightHandRing: "right_ring_1_joint",
    Inspire_Right_Hand_JointIndex.kRightHandMiddle: "right_middle_1_joint",
    Inspire_Right_Hand_JointIndex.kRightHandIndex: "right_index_1_joint",
    Inspire_Right_Hand_JointIndex.kRightHandThumbBend: "right_thumb_2_joint",
    Inspire_Right_Hand_JointIndex.kRightHandThumbRotation: "right_thumb_1_joint",
}


def ftp_to_urdf_angles(joint_pos: np.ndarray) -> np.ndarray:
    """
    FTP angle_act/1000 convention: 0=open, max=closed
    URDF convention:               0=closed, positive=open
    Ranges match the DFX normalize() bounds.
    """
    out = joint_pos.copy()

    out = unnormalize_inspire_angles(out)

    out[:4] = 1.7 - out[:4]  # fingers:        [0, 1.7]  → flip
    out[4] = 0.5 - out[4]  # thumb bend:     [0, 0.5]  → flip
    out[5] = 1.3 - out[5]  # thumb rotation: [-0.1, 1.3] → flip around midpoint
    return out


def hand_state_to_urdf_map(
    joint_pos: np.ndarray,
    joint_map: dict[int, str],
) -> dict[str, float]:
    """Map a 6-element hand state array to {urdf_name: position}."""
    assert len(joint_pos) == len(joint_map), f"Expected {len(joint_map)} joints, got {len(joint_pos)}"
    return {joint_map[i]: float(joint_pos[i]) for i in range(len(joint_pos))}


def unnormalize_inspire_angles(angles):
    angle_ranges = [
        (0.0, 1.7),  # idx 0
        (0.0, 1.7),  # idx 1
        (0.0, 1.7),  # idx 2
        (0.0, 1.7),  # idx 3
        (0, 0.5),  # idx 4
        (-0.1, 1.3),  # idx 5
    ]
    unnormalized = np.array(
        [angles[i] * (angle_ranges[i][0] - angle_ranges[i][1]) + angle_ranges[i][1] for i in range(Inspire_Num_Motors)]
    )
    return unnormalized
