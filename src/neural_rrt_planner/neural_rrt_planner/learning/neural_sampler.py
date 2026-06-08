#! /usr/bin/env python3

import math
import os
import random
from typing import TypeAlias

import torch
import torch.nn as nn

from neural_rrt_planner.config import MAP_CONFIG

"""
Neural Sampler for Neural-Guided RRT*.

학습된 SamplingMLP와 Normalizer를 로드하고,
현재 위치와 장애물 거리를 입력받아
goal 방향으로 편향된 샘플 포인트를 반환한다.

Interface:
    sampler = NeuralSampler()
    point = sampler.sample(current_position)  # [x, y, z]
"""

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Point3D: TypeAlias = list[float]
FloatTensor: TypeAlias = torch.Tensor

# ---------------------------------------------------------------------------
# Paths (train_mlp.py와 동일한 경로)
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.path.expanduser("~/term_project/models/sampling_mlp.pth")
NORMALIZER_PATH: str = os.path.expanduser("~/term_project/models/normalizer.pth")

# ---------------------------------------------------------------------------
# Architecture (train_mlp.py와 반드시 동일해야 함)
# ---------------------------------------------------------------------------
INPUT_DIM: int = 7
OUTPUT_DIM: int = 3
HIDDEN_DIMS: list[int] = [128, 128, 64] #[256, 256, 128, 64]


# ---------------------------------------------------------------------------
# Model (train_mlp.py와 동일한 구조)
# ---------------------------------------------------------------------------
class SamplingMLP(nn.Module):

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] = HIDDEN_DIMS,
        output_dim: int = OUTPUT_DIM,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim: int = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.LeakyReLU(negative_slope=0.1))  # ReLU → LeakyReLU
            # layers.append(nn.Dropout(p=0.2))                  # Dropout 추가
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: FloatTensor) -> FloatTensor:
        out = self.net(x)
        out = nn.functional.normalize(out, p=2, dim=1)
        return out


# ---------------------------------------------------------------------------
# NeuralSampler
# ---------------------------------------------------------------------------
class NeuralSampler:
    """
    학습된 MLP를 이용한 goal-biased 샘플러.

    sample() 호출 시:
        1. 현재 위치에서 가장 가까운 장애물까지의 거리 계산
        2. MLP 추론 → goal 방향 단위벡터 예측
        3. 현재 위치에서 sample_distance만큼 이동한 포인트 반환
        4. map boundary / depth limit 초과 시 클램핑
    """

    def __init__(self) -> None:
        self.map_config: dict = MAP_CONFIG
        self.bounds: dict = MAP_CONFIG["bounds"]
        self.goal: Point3D = MAP_CONFIG["goal"]
        self.obstacles: list[dict] = MAP_CONFIG["obstacles"]
        self.depth_limit: dict = MAP_CONFIG["depth_limit"]
        self.safety_margin: float = MAP_CONFIG["safety_margin"]
        self.sample_distance: float = MAP_CONFIG["neural_params"]["sample_distance"]

        self.device = torch.device("cpu")

        self.model: SamplingMLP = self._load_model()
        self.mean: FloatTensor
        self.std: FloatTensor
        self._load_normalizer()

        print("NeuralSampler initialized.")

    # --------------------------------------------------
    # Load
    # --------------------------------------------------
    def _load_model(self) -> SamplingMLP:
        model = SamplingMLP()
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device)
        )
        model.eval()
        print(f"Model loaded: {MODEL_PATH}")
        return model

    def _load_normalizer(self) -> None:
        stats = torch.load(NORMALIZER_PATH, map_location=self.device)
        self.mean = stats["mean"]
        self.std = stats["std"]
        print(f"Normalizer loaded: {NORMALIZER_PATH}")

    # --------------------------------------------------
    # Obstacle Distance (generate_sampling_dataset.py와 동일한 로직)
    # --------------------------------------------------
    def _compute_nearest_obstacle_distance(self, point: Point3D) -> float:
        min_distance: float = float("inf")

        px, py, pz = point[0], point[1], point[2]

        for obstacle in self.obstacles:
            if obstacle["type"] != "cylinder":
                continue

            cx: float = obstacle["center"][0]
            cy: float = obstacle["center"][1]
            cz: float = obstacle["center"][2]

            radius: float = obstacle["radius"] + self.safety_margin
            height: float = obstacle["height"] + 2.0 * self.safety_margin

            dx: float = px - cx
            dy: float = py - cy

            horizontal_distance: float = (
                math.sqrt(dx ** 2 + dy ** 2) - radius
            )

            z_min: float = cz - height / 2.0
            z_max: float = cz + height / 2.0

            if pz < z_min:
                vertical_distance: float = z_min - pz
            elif pz > z_max:
                vertical_distance = pz - z_max
            else:
                vertical_distance = 0.0

            horizontal_distance = max(horizontal_distance, 0.0)

            distance: float = math.sqrt(
                horizontal_distance ** 2 + vertical_distance ** 2
            )

            min_distance = min(min_distance, distance)

        if min_distance == float("inf"):
            return 10.0

        return min_distance

    # --------------------------------------------------
    # Normalize (train_mlp.py의 Normalizer.transform()과 동일)
    # --------------------------------------------------
    def _normalize(self, x: FloatTensor) -> FloatTensor:
        return (x - self.mean) / self.std

    # --------------------------------------------------
    # Clamp to valid map region
    # --------------------------------------------------
    def _clamp(self, point: Point3D) -> Point3D:
        x = max(self.bounds["x"][0], min(self.bounds["x"][1], point[0]))
        y = max(self.bounds["y"][0], min(self.bounds["y"][1], point[1]))
        z = max(self.depth_limit["z_min"], min(self.depth_limit["z_max"], point[2]))
        return [x, y, z]

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    def _predict_direction(self, current: Point3D) -> Point3D:
        """
        MLP 추론 → goal 방향 단위벡터 반환.

        Args:
            current (Point3D): 현재 위치 [x, y, z]

        Returns:
            Point3D: 예측된 방향 단위벡터 [dx, dy, dz]
        """
        nearest_obstacle_distance = self._compute_nearest_obstacle_distance(current)

        input_vec: FloatTensor = torch.tensor(
            [[
                current[0],
                current[1],
                current[2],
                self.goal[0],
                self.goal[1],
                self.goal[2],
                nearest_obstacle_distance,
            ]],
            dtype=torch.float32,
        )

        input_normalized = self._normalize(input_vec)

        with torch.no_grad():
            direction: FloatTensor = self.model(input_normalized)

        d = direction[0].tolist()
        return d

    # --------------------------------------------------
    # Sample
    # --------------------------------------------------
    def sample(self, current: Point3D) -> Point3D:
        """
        현재 위치를 기반으로 neural-guided 샘플 포인트 반환.

        현재 위치에서 MLP가 예측한 방향으로
        sample_distance만큼 이동한 포인트를 반환한다.
        map boundary / depth limit을 벗어나면 클램핑한다.

        Args:
            current (Point3D): 현재 위치 [x, y, z]

        Returns:
            Point3D: 샘플링된 포인트 [x, y, z]
        """
        direction: Point3D = self._predict_direction(current)

        sample_x = current[0] + direction[0] * self.sample_distance
        sample_y = current[1] + direction[1] * self.sample_distance
        sample_z = current[2] + direction[2] * self.sample_distance

        sample_point: Point3D = self._clamp([sample_x, sample_y, sample_z])

        return sample_point


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sampler = NeuralSampler()

    test_points: list[Point3D] = [
        MAP_CONFIG["start"],
        [1.0, 0.5, -0.5],
        [2.5, 1.0, -0.8],
        [3.5, 1.5, -1.0],
    ]

    print("\n--- NeuralSampler test ---")
    print(f"Goal: {MAP_CONFIG['goal']}")
    print()

    for current in test_points:
        sample = sampler.sample(current)
        print(f"Current: {[round(v, 3) for v in current]}"
              f"  →  Sample: {[round(v, 3) for v in sample]}")