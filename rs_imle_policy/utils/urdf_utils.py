import pinocchio as pin


def print_urdf_joints(urdf_path: str) -> None:
    """
    Load a URDF and print all joints.

    Args:
        urdf_path: Path to the URDF file
    """
    model = pin.buildModelFromUrdf(urdf_path)

    print(f"Robot name: {model.name}")
    print(f"Number of joints: {model.njoints}\n")

    for joint_id in range(model.njoints):
        joint = model.joints[joint_id]
        name = model.names[joint_id]

        print(f"Joint ID: {joint_id}")
        print(f"  Name: {name}")
        print(f"  Type: {joint.shortname()}")
        print(f"  nq: {joint.nq}")
        print(f"  nv: {joint.nv}")
        print()


def get_finger_joints(model: pin.Model):
    """
    Return all finger joint names from a Pinocchio model.
    """
    finger_keywords = ["index", "middle", "ring", "little", "thumb"]

    finger_joints = [
        model.names[i]
        for i in range(model.njoints)
        if any(k in model.names[i] for k in finger_keywords)
    ]

    return finger_joints


def print_finger_joints(model: pin.Model):
    joints = get_finger_joints(model)

    print(",\n".join(f'"{j}"' for j in joints))


# Lock every joint except the arms
BODY_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

DEX3_HAND_JOINTS = [
    # Dex3 Hand Joints
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]

INSPIRE_HAND_JOINTS = [
    # Inspire hand Joints
    "left_index_1_joint",
    "left_index_2_joint",
    "left_little_1_joint",
    "left_little_2_joint",
    "left_middle_1_joint",
    "left_middle_2_joint",
    "left_ring_1_joint",
    "left_ring_2_joint",
    "left_thumb_1_joint",
    "left_thumb_2_joint",
    "left_thumb_3_joint",
    "left_thumb_4_joint",
    "right_index_1_joint",
    "right_index_2_joint",
    "right_little_1_joint",
    "right_little_2_joint",
    "right_middle_1_joint",
    "right_middle_2_joint",
    "right_ring_1_joint",
    "right_ring_2_joint",
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_thumb_3_joint",
    "right_thumb_4_joint",
]
