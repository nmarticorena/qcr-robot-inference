from loop_rate_limiters import RateLimiter

from rs_imle_policy.g1_arm_ik import G1ReducedPinkIK

ik = G1ReducedPinkIK(
    urdf_path="assets/g1.urdf",
    mesh_dirs=["assets/"],
    srdf_path="assets/g1.srdf",
    visualize=True,
    spawn_visualizer=True,
    enable_self_collision=True,
)

targets = ik.get_targets_from_configuration()
ik.set_targets(targets.left, targets.right)

rate = RateLimiter(int(1 / CONTROL_DT), warn=True)
while True:
    rate.sleep()
    q = ik.solve(dt=CONTROL_DT)
    ik.configuration.update(q.copy())
    ik.viz.display(q)
