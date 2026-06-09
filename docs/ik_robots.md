Our ik solution is based in **P**ython **in**verse **k**inematics for articulated robot models, aka [pink](https://github.com/stephane-caron/pink).

## Adding a new robot
We assume that you already have available a .urdf of the robot describing both the robot kinematics and dynamics if self collision is desired you also required to have collision geometries in addition to a semantic robot description file (i.e, .srdf).
### Collision geometries
For collision detection is often used an approximation of the robot visual geometry, this can be done by the combination of geometric primitives, spherification (often used in GPU accelerated pipelines) or convex hulls. The easier approach I found when using pinocchio is using convex hull and then forcing collision checking to be convex by either adding the convex flag to the urdf, or with our function in the collision utils. [1]

```xml
<link name="right_shoulder_roll_link">
   <...>
   <collision_checking>
      <convex name="right_shoulder_roll_link"/>
    </collision_checking>
  </link>
```

### Semantic robot description format (SRDF)

The inclusion of this file allows for excluding collision-pairs between two links if there are adjacent or there is not a single joint configuration in which this links are in collision. The easier approach to get the srdf is using [moveit-setup](https://moveit.picknik.ai/main/doc/examples/setup_assistant/setup_assistant_tutorial.html), but also scripts that traverse a urdf is also possible [urdf_to_disable_collision.py](https://gist.github.com/awesomebytes/18fe75b808c4c644bd3d)

 
[1] I still haven't figure it out if the convex meshes need to have a minimal number of vertices to be more efficient, I know that simulators use this capability but unaware of this scenario for pin
