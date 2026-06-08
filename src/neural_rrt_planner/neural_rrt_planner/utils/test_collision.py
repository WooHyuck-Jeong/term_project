from neural_rrt_planner.utils.collision_checker import CollisionChecker

checker = CollisionChecker()

point = [1.0, 1.0, -0.5]

print(
    checker.is_valid_point(point)
)

start = [0.4, 0.4, -0.5]
goal = [4.6, 1.6, -1.0]

print(
    checker.is_collision_free(
        start,
        goal
    )
)