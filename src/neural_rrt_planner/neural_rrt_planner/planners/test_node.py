from neural_rrt_planner.planners.rrt_node import RRTNode

node = RRTNode(
    1.0,
    0.5,
    -0.8
)

print(node)

print(node.position())