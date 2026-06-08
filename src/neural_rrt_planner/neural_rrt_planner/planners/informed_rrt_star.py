import math
import random
import numpy as np

from neural_rrt_planner.planners.rrt_star import RRTStar
from neural_rrt_planner.planners.rrt_node import RRTNode


class InformedRRTStar(RRTStar):

    def __init__(self):
        super().__init__()

        self.c_min = self.distance(
            self.start_node,
            self.goal_node
        )

        self.x_center = np.array([
            (self.start_node.x + self.goal_node.x) / 2.0,
            (self.start_node.y + self.goal_node.y) / 2.0,
            (self.start_node.z + self.goal_node.z) / 2.0,
        ])

    # --------------------------------------------------
    # Override Sampling Function
    # --------------------------------------------------
    def sample_random_node(self):

        if self.best_goal_node is None:
            return self.sample_uniform_node()

        if random.random() < self.goal_sample_rate:
            return RRTNode(
                self.goal_node.x,
                self.goal_node.y,
                self.goal_node.z
            )

        return self.sample_informed_node()

    # --------------------------------------------------
    # Uniform Sampling
    # --------------------------------------------------
    def sample_uniform_node(self):
        bounds = self.map_config["bounds"]

        while True:
            x = random.uniform(
                bounds["x"][0],
                bounds["x"][1]
            )
            y = random.uniform(
                bounds["y"][0],
                bounds["y"][1]
            )
            z = random.uniform(
                bounds["z"][0],
                bounds["z"][1]
            )

            node = RRTNode(x, y, z)

            if self.collision_checker.is_valid_point(
                node.position()
            ):
                return node

    # --------------------------------------------------
    # Sample inside unit ball
    # --------------------------------------------------
    def sample_unit_ball(self):
        while True:
            point = np.array([
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
            ])

            if np.linalg.norm(point) <= 1.0:
                return point

    # --------------------------------------------------
    # Rotation Matrix
    # x-axis -> start-goal direction
    # --------------------------------------------------
    def get_rotation_matrix(self):
        start = np.array(
            self.start_node.position()
        )

        goal = np.array(
            self.goal_node.position()
        )

        direction = goal - start
        norm = np.linalg.norm(direction)

        if norm < 1e-9:
            return np.eye(3)

        e1 = direction / norm
        x_axis = np.array([1.0, 0.0, 0.0])

        v = np.cross(x_axis, e1)
        c = np.dot(x_axis, e1)

        if np.linalg.norm(v) < 1e-9:
            if c > 0:
                return np.eye(3)
            return np.array([
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ])

        vx = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])

        rotation = (
            np.eye(3)
            +
            vx
            +
            vx @ vx
            *
            ((1.0 - c) / (np.linalg.norm(v) ** 2))
        )

        return rotation

    # --------------------------------------------------
    # Informed Sampling
    # --------------------------------------------------
    def sample_informed_node(self):
        c_best = self.best_goal_node.cost

        if c_best < self.c_min:
            return self.sample_uniform_node()

        a1 = c_best / 2.0

        minor_term = max(
            c_best ** 2 - self.c_min ** 2,
            0.0
        )

        a2 = math.sqrt(minor_term) / 2.0
        a3 = a2

        L = np.diag([
            a1,
            a2,
            a3
        ])

        C = self.get_rotation_matrix()

        for _ in range(100):
            x_ball = self.sample_unit_ball()

            x_rand = (
                C @ L @ x_ball
                +
                self.x_center
            )

            node = RRTNode(
                x_rand[0],
                x_rand[1],
                x_rand[2]
            )

            if self.collision_checker.is_valid_point(
                node.position()
            ):
                return node

        return self.sample_uniform_node()

    # --------------------------------------------------
    # Ellipsoid Information for Visualization
    # --------------------------------------------------
    def get_ellipsoid_info(self):

        if self.best_goal_node is None:
            return None

        c_best = self.best_goal_node.cost

        a1 = c_best / 2.0

        minor_term = max(
            c_best ** 2 - self.c_min ** 2,
            0.0
        )

        a2 = math.sqrt(minor_term) / 2.0
        a3 = a2

        direction = np.array(
            self.goal_node.position()
        ) - np.array(
            self.start_node.position()
        )

        return {
            "center": self.x_center,
            "axes": [a1, a2, a3],
            "direction": direction,
        }