import pinocchio as pin


def force_convex_collision_geometry(robot: pin.RobotWrapper) -> int:
    """
    Force convex collision geometry for the robot.
    This method loads the collision geometries and convertst them
    to convex hulls, which can be used for collision checking and distance

    Important this operation is in place

    Args:
        robot:pin.RobotWrapper: Pinnochio robot wrapper, which contains the collision model to be converted.
    Returns:
        converted: int
            The number of geometries that are converted to convex hulls.
    """
    converted = 0
    for go in robot.collision_model.geometryObjects:
        geom = go.geometry
        if type(geom).__name__.startswith("BVHModel"):
            geom.buildConvexRepresentation(True)
            if hasattr(geom, "convex") and geom.convex is not None:
                go.geometry = geom.convex
                converted += 1
    robot.collision_data = pin.GeometryData(robot.collision_model)
    return converted
