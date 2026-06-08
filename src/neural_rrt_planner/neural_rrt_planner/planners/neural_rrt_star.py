#! /usr/bin/env python3

import random

from neural_rrt_planner.config import MAP_CONFIG
from neural_rrt_planner.planners.rrt_node import RRTNode
from neural_rrt_planner.planners.rrt_star import RRTStar
from neural_rrt_planner.learning.neural_sampler import NeuralSampler

"""
Neural-Guided RRT* Planner.

RRTStar를 상속하여 sample_random_node()만 오버라이드한다.

Sampling 전략:
    - goal_sample_rate      : goal 직접 샘플링 (RRTStar와 동일)
    - neural_sample_rate    : NeuralSampler 사용 (nearest node 기반)
    - random_sample_rate    : uniform random 샘플링 (탐색 다양성 확보)

config.py neural_params:
    neural_sample_rate  : 0.80
    random_sample_rate  : 0.20
    sample_distance     : 0.35
"""


class NeuralRRTStar(RRTStar):

    def __init__(self):
        super().__init__()

        self.neural_params: dict = MAP_CONFIG["neural_params"]
        self.neural_sample_rate: float = self.neural_params["neural_sample_rate"]
        self.random_sample_rate: float = self.neural_params["random_sample_rate"]

        self.neural_sampler: NeuralSampler = NeuralSampler()

    # --------------------------------------------------
    # Override: sample_random_node
    # --------------------------------------------------
    def sample_random_node(self) -> RRTNode:
        """
        세 가지 전략을 확률적으로 선택하여 샘플 노드를 반환한다.

        1. goal_sample_rate  확률 → goal 직접 샘플링
        2. neural_sample_rate 확률 → NeuralSampler 기반 샘플링
        3. random_sample_rate 확률 → uniform random 샘플링

        Returns:
            RRTNode: 샘플링된 노드
        """

        # 1. Goal sampling
        if random.random() < self.goal_sample_rate:
            return RRTNode(
                self.goal_node.x,
                self.goal_node.y,
                self.goal_node.z,
            )

        # 2. Neural sampling vs Random sampling
        use_neural = random.random() < (
            self.neural_sample_rate
            /
            (self.neural_sample_rate + self.random_sample_rate)
        )

        if use_neural:
            return self.sample_neural_node()
        else:
            return self.sample_uniform_node()

    # --------------------------------------------------
    # Neural Sampling
    # --------------------------------------------------
    def sample_neural_node(self) -> RRTNode:
        """
        현재 트리의 nearest node를 기준으로
        NeuralSampler를 이용해 goal 방향으로 편향된 샘플을 생성한다.

        nearest node를 current로 사용하는 이유:
            트리의 frontier(탐색 경계)에서 goal 방향으로
            확장하는 것이 가장 효과적이기 때문.

        Returns:
            RRTNode: neural-guided 샘플 노드
        """
        nearest = self._get_nearest_to_goal()

        sample_point = self.neural_sampler.sample(
            nearest.position()
        )

        return RRTNode(
            sample_point[0],
            sample_point[1],
            sample_point[2],
        )

    # --------------------------------------------------
    # Uniform Sampling (fallback)
    # --------------------------------------------------
    def sample_uniform_node(self) -> RRTNode:
        """
        map boundary 안에서 uniform random 샘플링.
        collision check를 통과한 유효한 노드만 반환한다.

        Returns:
            RRTNode: uniform random 샘플 노드
        """
        bounds = self.map_config["bounds"]

        while True:
            x = random.uniform(bounds["x"][0], bounds["x"][1])
            y = random.uniform(bounds["y"][0], bounds["y"][1])
            z = random.uniform(bounds["z"][0], bounds["z"][1])

            node = RRTNode(x, y, z)

            if self.collision_checker.is_valid_point(node.position()):
                return node

    # --------------------------------------------------
    # Helper: goal에 가장 가까운 트리 노드 반환
    # --------------------------------------------------
    def _get_nearest_to_goal(self) -> RRTNode:
        """
        현재 트리에서 goal에 가장 가까운 노드를 반환한다.
        NeuralSampler의 current 입력으로 사용된다.

        Returns:
            RRTNode: goal과 가장 가까운 트리 노드
        """
        return min(
            self.nodes,
            key=lambda node: self.distance(node, self.goal_node),
        )