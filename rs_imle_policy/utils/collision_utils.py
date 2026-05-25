import pinocchio as pin


def force_convex_collision_geometry(robot: pin.RobotWrapper, inflation: float = 0.0) -> int:
    """
    Force convex collision geometry for the robot.
    This method loads the collision geometries and convert them
    to convex hulls, which can be used for collision checking and distance

    Args:
        robot:pin.RobotWrapper: pinocchio robot wrapper, which contains the collision model to be converted.
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
                if inflation > 0.0:
                    go.geometry.setSweptSphereRadius(inflation)
                converted += 1
        if type(geom).__name__.startswith("Convex"):
            go.geometry.setSweptSphereRadius(inflation)
    robot.collision_data = pin.GeometryData(robot.collision_model)
    return converted
