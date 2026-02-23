from numpy.typing import NDArray
from typing import Optional
import rerun as rr
import spatialgeometry as sg
import trimesh
import roboticstoolbox as rtb
import numpy as np


class ReRunRobot:
    def __init__(
        self, robot: rtb.Robot, prefix: str, alpha: Optional[float] = None
    ) -> None:
        self.robot = robot
        self.name = f"{prefix}/{robot.name}"
        self.alpha = alpha
        self._load_meshes()

    def log_robot_state(self, q: NDArray) -> None:
        """
        Log robot positions to rerun
        q: joint positions
        """
        transforms = self.robot.fkine_all(q)
        rr.log(self.name + "/q", rr.Scalars(q))
        for ix, link in enumerate(self.robot.links):
            O_T_ix: NDArray[np.float64] = transforms[ix + 1].A
            rr.log(
                self.name + "/" + link.name,
                rr.Transform3D(translation=O_T_ix[:3, 3], mat3x3=O_T_ix[:3, :3]),
            )

    def log_pose(
        self, position: NDArray, quaternion: NDArray, name: str = "pose"
    ) -> None:
        """
        Log a 3D pose to rerun
        position: [x, y, z]
        quaternion: [w, x, y, z]
        """
        quad = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]  # xyzw
        rr.log(
            f"{self.name}/{name}",
            rr.Transform3D(translation=position, rotation=rr.Quaternion(xyzw=quad)),
            rr.TransformAxes3D(axis_length=0.1),
        )

    def log_frame(self, image, camera_name):
        """
        Log a camera frame to rerun
        image: HxWx3 array
        """
        rr.log(f"{self.name}/{camera_name}", rr.Image(image).compress())

    def _apply_alpha(self, vertex_colour: NDArray) -> NDArray:
        if self.alpha is None:
            return vertex_colour
        vertex_colour = np.atleast_2d(vertex_colour)
        vertex_colour[:, -1] = int(self.alpha * 255)
        return vertex_colour

    def _load_meshes(self) -> None:
        for link in self.robot.links:
            if len(link.geometry) == 0:
                continue
            geom = link.geometry[0]
            if isinstance(geom, sg.Mesh):
                mesh = trimesh.load_mesh(geom.filename, process=False)
                vertex_colour = self._apply_alpha(mesh.visual.to_color().vertex_colors)
                rr.log(
                    f"{self.name}/{link.name}",
                    rr.Mesh3D(
                        vertex_positions=mesh.vertices,
                        triangle_indices=mesh.faces,
                        vertex_normals=mesh.vertex_normals,
                        vertex_colors=vertex_colour,
                    ),
                    static=True,
                )


if __name__ == "__main__":
    robot = rtb.models.Panda()
    rr.init("urdf_test", recording_id="testing", spawn=True)
    rr.log(f"/{robot.name}", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rerun_robot = ReRunRobot(robot, "test")
    rerun_robot_2 = ReRunRobot(robot, "transparent", alpha=0.1)
    for _ in range(20):
        rerun_robot.log_robot_state(rerun_robot.robot.random_q())
        rerun_robot_2.log_robot_state(rerun_robot_2.robot.random_q())
