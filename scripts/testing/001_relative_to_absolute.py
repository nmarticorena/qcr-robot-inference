import numpy as np
import rerun as rr
from rs_imle_policy.inference import RobotInferenceController
import spatialmath as sm

poses = []
poses.append([np.array([0.1, 0.0, 0.0]), np.eye(3)])
poses.append([np.array([0.0, 0.0, 0.1]), np.eye(3)])
poses.append([np.array([-0.1, 0.0, 0.0]), np.eye(3)])
poses.append([np.array([0.0, 0.0, -0.1]), np.eye(3)])

rr.init("relative_to_absolute_example", spawn=True)

for ix, pose in enumerate(poses):
    rr.log(
        f"test/{ix}",
        rr.Transform3D(
            translation=pose[0],
            mat3x3=pose[1],
        ),
        rr.TransformAxes3D(axis_length=0.1),
    )

trans = np.stack([p[0] for p in poses])
rots = np.stack([p[1] for p in poses])

abs_trans, abs_rots = RobotInferenceController.transform_action_to_absolute(trans, rots)

for ix in range(len(poses)):
    rr.log(
        f"absolute/{ix}",
        rr.Transform3D(translation=abs_trans[ix], mat3x3=abs_rots[ix]),
        rr.TransformAxes3D(axis_length=0.1),
    )

# Test with rotations
trans = np.repeat(np.array([0.1, 0, 0])[None, :], 8, axis=0)
rots = np.repeat(sm.SO3.Rz(np.pi / 4).A[None, :, :], 8, axis=0)

abs_trans, abs_rots = RobotInferenceController.transform_action_to_absolute(trans, rots)

for ix in range(len(abs_trans)):
    rr.log(
        f"absolute_rot/{ix}",
        rr.Transform3D(translation=abs_trans[ix], mat3x3=abs_rots[ix]),
        rr.TransformAxes3D(axis_length=0.1),
    )
