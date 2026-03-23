Our ik solution is based in pink.


## Adding a new robot
We assume that you already have available a .urdf of the robot describing both the robot kinematics and dynamics.
if self collision is desired you also required to have collision geometries in addition to a sematic robot description file.
### Collision geometries
For collision detection is often used an approximation of the robot visual geometry, this can be done by the combination of geometric primitives, spherification (often used in GPU accelerated pipelines) or convex hulls.
The easier approach I found when using pinocchio is using convex hull and then forcing collision checking to be convex by either adding the convex flag to the urdf, or with our function in the collision utils. [1]

```xml
<link name="right_shoulder_roll_link">
   <...>
   <collision_checking>
      <convex name="right_shoulder_roll_link"/>
    </collision_checking>
  </link>
```


For implementing an additional robot you required to add a urdf file in adition to a .srdf file.


The .srdf file can be obtained from

[1] I still havent figure it out if the convex meshes need to have a minimal number of vertices to be more efficient, I know that simulators use this capability but unaware of this scenario for pin
