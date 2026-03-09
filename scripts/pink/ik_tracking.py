#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev
#
# /// script
# dependencies = ["daqp", "loop-rate-limiters", "meshcat", "pin-pink",
# "qpsolvers", "robot_descriptions">=1.21]
# ///

"""G1 humanoid squat by regulating CoM"""

import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter
import viser
from scipy.spatial.transform import Rotation


import pink
from pink import solve_ik
from pink.tasks import ComTask, FrameTask, PostureTask
from pink.barriers import SelfCollisionBarrier
from pink.visualization import start_viser_visualizer
from pink.utils import process_collision_pairs

import xml.etree.ElementTree as ET


def parse_srdf_default_disabled_links(srdf_path: str) -> set[str]:
    root = ET.parse(srdf_path).getroot()
    return {e.attrib["link"] for e in root.findall(".//disable_default_collisions")}


def apply_disable_default_collisions(model, geom_model, srdf_path: str):
    disabled_links = parse_srdf_default_disabled_links(srdf_path)

    removed = 0
    # iterate backwards so indices stay valid when popping
    for i in range(len(geom_model.collisionPairs) - 1, -1, -1):
        pair = geom_model.collisionPairs[i]
        go1 = geom_model.geometryObjects[pair.first]
        go2 = geom_model.geometryObjects[pair.second]
        l1 = geom_parent_link_name(model, go1)
        l2 = geom_parent_link_name(model, go2)

        if (l1 in disabled_links) or (l2 in disabled_links):
            geom_model.collisionPairs.pop(i)
            removed += 1

    return removed, len(disabled_links)


def add_handle(task: FrameTask, scale: float):
    pose = task.transform_target_to_world.np
    frame = task.frame

    handle = viz.viewer.scene.add_transform_controls(
        "/" + frame,
        position=pose[:3, 3],
        wxyz=Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True),
        scale=scale,
        line_width=3.0,
    )

    @handle.on_update
    def _(evt: viser.TransformControlsEvent, task=task):
        T = task.transform_target_to_world
        T.translation = evt.target.position
        T.rotation = Rotation.from_quat(evt.target.wxyz, scalar_first=True).as_matrix()

    return handle


def loadG1() -> pin.RobotWrapper:
    robot = pin.RobotWrapper.BuildFromURDF(
        "assets/g1.urdf", ["assets/"], pin.JointModelFreeFlyer()
    )

    robot.collision_model.addAllCollisionPairs()
    print("Collision pairs:", len(robot.collision_model.collisionPairs))

    srdf = "/home/nmarticorena/tools/unitree_moveit_config3/config/g1.srdf"

    converted = force_convex_collision_geometry(robot)
    print(f"Converted {converted} collision meshes to convex.")

    # Collision pairs
    robot.collision_model.removeAllCollisionPairs()
    robot.collision_model.addAllCollisionPairs()
    pin.removeCollisionPairs(robot.model, robot.collision_model, srdf)
    robot.collision_data = process_collision_pairs(
        robot.model, robot.collision_model, srdf
    )
    print(
        "Collision pairs after processing:", len(robot.collision_model.collisionPairs)
    )

    def inspect_collision_types(robot):
        for go in robot.collision_model.geometryObjects:
            print(go.name, type(go.geometry).__name__)

    inspect_collision_types(robot)
    return robot


def geom_parent_link_name(model: pin.Model, geom_obj: pin.GeometryObject) -> str:
    # geom_obj.parentFrame is a Pinocchio frame index
    return model.frames[geom_obj.parentFrame].name


def force_convex_collision_geometry(robot: pin.RobotWrapper) -> int:
    converted = 0

    for go in robot.collision_model.geometryObjects:
        geom = go.geometry
        geom_type = type(geom).__name__

        # Only convert mesh BVHs
        if geom_type.startswith("BVHModel"):
            geom.buildConvexRepresentation(True)

            # Pinocchio example replaces the geometry by geom.convex
            if hasattr(geom, "convex") and geom.convex is not None:
                go.geometry = geom.convex
                converted += 1
            else:
                print(f"[WARN] Could not build convex hull for {go.name}")

    # IMPORTANT: rebuild collision data after modifying collision geometries
    robot.collision_data = pin.GeometryData(robot.collision_model)
    return converted


def print_closest_pairs(robot, q, top_k=20):
    pin.computeDistances(
        robot.model, robot.data, robot.collision_model, robot.collision_data, q
    )

    rows = []
    for k, pair in enumerate(robot.collision_model.collisionPairs):
        res = robot.collision_data.distanceResults[k]
        go1 = robot.collision_model.geometryObjects[pair.first]
        go2 = robot.collision_model.geometryObjects[pair.second]

        l1 = geom_parent_link_name(robot.model, go1)
        l2 = geom_parent_link_name(robot.model, go2)

        rows.append((res.min_distance, k, l1, l2, go1.name, go2.name))

    rows.sort(key=lambda x: x[0])

    print("\n=== Closest collision pairs ===")
    for d, k, l1, l2, g1, g2 in rows[:top_k]:
        print(f"[{k:03d}] d={d: .6f}  links=({l1}, {l2})  geoms=({g1}, {g2})")

    print("\nMinimum distance:", rows[0][0])


def add_debug_marker(scene, name, position, radius=0.02, color=(255, 0, 0)):
    # Try sphere-like primitives first
    if hasattr(scene, "add_icosphere"):
        return scene.add_icosphere(
            name,
            radius=radius,
            position=position,
            color=color,
        )

    if hasattr(scene, "add_sphere"):
        return scene.add_sphere(
            name,
            radius=radius,
            position=position,
            color=color,
        )

    # Fallback: tiny frame marker
    return scene.add_frame(
        name,
        position=position,
        wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        axes_length=radius * 2.0,
        axes_radius=radius * 0.2,
    )


class CollisionDebugViewer:
    def __init__(self, viz, max_pairs=10, radius=0.02):
        self.viz = viz
        self.max_pairs = max_pairs
        self.radius = radius
        self.handles = []
        off = np.array([0.0, 0.0, -100.0])

        scene = viz.viewer.scene
        for i in range(max_pairs):
            p1 = add_debug_marker(
                scene,
                f"/collision_debug/pair_{i}/p1",
                position=off,
                radius=radius,
                color=(255, 0, 0),
            )
            p2 = add_debug_marker(
                scene,
                f"/collision_debug/pair_{i}/p2",
                position=off,
                radius=radius,
                color=(0, 0, 255),
            )

            label = None
            if hasattr(scene, "add_label"):
                label = scene.add_label(
                    f"/collision_debug/pair_{i}/label",
                    text=f"pair_{i}",
                    position=off,
                )

            self.handles.append({"p1": p1, "p2": p2, "label": label})

    def update(self, rows):
        off = np.array([0.0, 0.0, -100.0])

        for i, h in enumerate(self.handles):
            if (
                i < len(rows)
                and rows[i]["p1"] is not None
                and rows[i]["p2"] is not None
            ):
                row = rows[i]
                p1 = row["p1"]
                p2 = row["p2"]
                mid = 0.5 * (p1 + p2)

                # red if penetrating, orange if near
                if row["distance"] < 0.0:
                    color1 = (255, 0, 0)
                    color2 = (255, 0, 0)
                else:
                    color1 = (255, 165, 0)
                    color2 = (255, 215, 0)

                if hasattr(h["p1"], "position"):
                    h["p1"].position = p1
                if hasattr(h["p2"], "position"):
                    h["p2"].position = p2
                if hasattr(h["p1"], "color"):
                    h["p1"].color = color1
                if hasattr(h["p2"], "color"):
                    h["p2"].color = color2
                if hasattr(h["p1"], "visible"):
                    h["p1"].visible = True
                if hasattr(h["p2"], "visible"):
                    h["p2"].visible = True

                if h["label"] is not None:
                    if hasattr(h["label"], "position"):
                        h["label"].position = mid
                    if hasattr(h["label"], "text"):
                        h["label"].text = (
                            f"[{row['pair_index']}] "
                            f"{row['link1']} ↔ {row['link2']}  "
                            f"d={row['distance']:.4f}"
                        )
                    if hasattr(h["label"], "visible"):
                        h["label"].visible = True
            else:
                if hasattr(h["p1"], "position"):
                    h["p1"].position = off
                if hasattr(h["p2"], "position"):
                    h["p2"].position = off
                if hasattr(h["p1"], "visible"):
                    h["p1"].visible = False
                if hasattr(h["p2"], "visible"):
                    h["p2"].visible = False
                if h["label"] is not None:
                    if hasattr(h["label"], "position"):
                        h["label"].position = off
                    if hasattr(h["label"], "visible"):
                        h["label"].visible = False


if __name__ == "__main__":
    robot = loadG1()

    # Initialize visualization
    viz = start_viser_visualizer(robot)
    debug_viewer = CollisionDebugViewer(viz, max_pairs=8, radius=0.025)

    q_ref = np.zeros(robot.nq)
    q_ref[2] = 0.72
    q_ref[6] = 1.0

    configuration = pink.Configuration(
        robot.model,
        robot.data,
        q_ref,
        collision_model=robot.collision_model,
        collision_data=robot.collision_data,
    )
    print_closest_pairs(robot, configuration.q, top_k=30)

    pelvis_orientation_task = FrameTask(
        "pelvis",
        position_cost=0.0,  # [cost] / [m]
        orientation_cost=10.0,  # [cost] / [rad]
    )

    com_task = ComTask(cost=200.0)
    com_task.set_target_from_configuration(configuration)

    posture_task = PostureTask(
        cost=1e-1,  # [cost] / [rad]
    )

    tasks = [pelvis_orientation_task, posture_task, com_task]

    for foot in ["right_ankle_roll_link", "left_ankle_roll_link"]:
        task = FrameTask(
            frame=foot,
            position_cost=[2.0, 2.0, 200.0],  # [cost] / [m]
            orientation_cost=10.0,  # [cost] / [rad]
        )
        tasks.append(task)

    for arm_points in ["right_wrist_yaw_link", "left_wrist_yaw_link"]:
        task = FrameTask(
            frame=arm_points,
            position_cost=[20.0],  # [cost] / [m]
            orientation_cost=10.0,  # [cost] / [rad]
        )
        tasks.append(task)

    interactive_frames = [
        "right_ankle_roll_link",
        "left_ankle_roll_link",
        "right_wrist_yaw_link",
        "left_wrist_yaw_link",
    ]

    for task in tasks:
        task.set_target_from_configuration(configuration)
        if isinstance(task, FrameTask):
            target = task.transform_target_to_world
            if task.frame in ["right_wrist_yaw_link", "left_wrist_yaw_link"]:
                # target.translation += np.array([-0.1, 0.0, -0.2])
                task.set_target(target)
            if task.frame in interactive_frames:
                _ = add_handle(task, 0.1)

    print(len(robot.collision_model.collisionPairs))
    # Pink Barriers
    collision_barrier = SelfCollisionBarrier(
        n_collision_pairs=10,
        gain=5.0,
        safe_displacement_gain=0.01,
        d_min=0.002,
    )

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    period = 2
    omega = 2 * np.pi / period

    while True:
        pin.centerOfMass(robot.model, robot.data, configuration.q)
        com = robot.data.com[0]
        # Update CoM target
        Az = 0.05
        desired_com = np.zeros(3)
        desired_com[2] = 0.55 + Az * np.sin(omega * t)
        com_task.set_target(desired_com)

        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            damping=0.01,
            safety_break=False,
            barriers=[collision_barrier],
        )
        configuration.integrate_inplace(velocity, dt)
        viz.display(configuration.q)
        # rows = collect_closest_collision_pairs(
        #     robot,
        #     configuration.q,
        #     top_k=8,
        #     max_distance=0.05,   # only visualize pairs within 5 cm
        # )
        # debug_viewer.update(rows)
        #
        # if rows:
        #     worst = rows[0]
        #     print(
        #         f"worst pair [{worst['pair_index']}] "
        #         f"{worst['link1']} <-> {worst['link2']} "
        #         f"d={worst['distance']:.6f}"
        #     )

        rate.sleep()
        t += dt
