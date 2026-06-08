import math

from neural_rrt_planner.config import MAP_CONFIG


class CollisionChecker:

    def __init__(self):
        self.bounds = MAP_CONFIG["bounds"]
        self.depth_limit = MAP_CONFIG["depth_limit"]
        self.obstacles = MAP_CONFIG["obstacles"]

        self.safety_margin = MAP_CONFIG["safety_margin"]

        self.check_resolution = (
            MAP_CONFIG["rrt_params"]["collision_check_resolution"]
        )

    # --------------------------------------------------
    # Boundary Check
    # --------------------------------------------------
    def is_inside_bounds(self, point):

        x, y, z = point

        return (
            self.bounds["x"][0] <= x <= self.bounds["x"][1]
            and
            self.bounds["y"][0] <= y <= self.bounds["y"][1]
            and
            self.bounds["z"][0] <= z <= self.bounds["z"][1]
        )

    # --------------------------------------------------
    # Depth Constraint Check
    # --------------------------------------------------
    def is_depth_valid(self, point):

        z = point[2]

        return (
            self.depth_limit["z_min"]
            <= z
            <=
            self.depth_limit["z_max"]
        )

    # --------------------------------------------------
    # Cylinder Obstacle Collision
    # --------------------------------------------------
    def is_point_in_obstacle(self, point):

        x, y, z = point

        for obs in self.obstacles:

            if obs["type"] != "cylinder":
                continue

            cx, cy, cz = obs["center"]

            radius = (
                obs["radius"]
                + self.safety_margin
            )

            height = (
                obs["height"]
                + 2.0 * self.safety_margin
            )

            dx = x - cx
            dy = y - cy

            radial_distance = math.sqrt(
                dx**2 + dy**2
            )

            z_min = cz - height / 2.0
            z_max = cz + height / 2.0

            if (
                radial_distance <= radius
                and
                z_min <= z <= z_max
            ):
                return True

        return False

    # --------------------------------------------------
    # Point Validity
    # --------------------------------------------------
    def is_valid_point(self, point):

        if not self.is_inside_bounds(point):
            return False

        if not self.is_depth_valid(point):
            return False

        if self.is_point_in_obstacle(point):
            return False

        return True

    # --------------------------------------------------
    # Edge Collision Check
    # --------------------------------------------------
    def is_collision_free(
        self,
        start_point,
        end_point
    ):

        distance = math.sqrt(
            (end_point[0] - start_point[0]) ** 2
            +
            (end_point[1] - start_point[1]) ** 2
            +
            (end_point[2] - start_point[2]) ** 2
        )

        num_steps = max(
            2,
            int(distance / self.check_resolution)
        )

        for i in range(num_steps + 1):

            t = i / num_steps

            x = (
                start_point[0]
                +
                t * (end_point[0] - start_point[0])
            )

            y = (
                start_point[1]
                +
                t * (end_point[1] - start_point[1])
            )

            z = (
                start_point[2]
                +
                t * (end_point[2] - start_point[2])
            )

            point = [x, y, z]

            if not self.is_valid_point(point):
                return False

        return True