#! /usr/bin/env python3
"""
Neural Sampler v2 for Neural-Guided RRT*.

neural_sampler.py와 완전히 동일한 구조이며
v2 모델 경로와 아키텍처만 다르다.

    v1 (neural_sampler.py)  : sampling_mlp.pth,    [128, 128, 64],      ReLU
    v2 (neural_sampler_v2.py): sampling_mlp_v2.pth, [256, 256, 128, 64], LeakyReLU + Dropout

Interface:
    sampler = NeuralSamplerV2()
    point = sampler.sample(current_position)  # [x, y, z]
"""

import math
import os
from typing import TypeAlias

import torch
import torch.nn as nn

from neural_rrt_planner.config import MAP_CONFIG

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Point3D: TypeAlias = list[float]
FloatTensor: TypeAlias = torch.Tensor

# ---------------------------------------------------------------------------
# Paths (train_mlp_v2.py와 동일한 경로)
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.path.expanduser("~/term_project/models/sampling_mlp_v2.pth")
NORMALIZER_PATH: str = os.path.expanduser("~/term_project/models/normalizer_v2.pth")

# ---------------------------------------------------------------------------
# Architecture (train_mlp_v2.py와 반드시 동일해야 함)
# ---------------------------------------------------------------------------
INPUT_DIM: int = 7
OUTPUT_DIM: int = 3
HIDDEN_DIMS: list[int] = [256, 256, 128, 64]
DROPOUT_RATE: float = 0.2


# ---------------------------------------------------------------------------
# Model (train_mlp_v2.py와 동일한 구조)
# ---------------------------------------------------------------------------
class SamplingMLP(nn.Module):

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] = HIDDEN_DIMS,
        output_dim: int = OUTPUT_DIM,
        dropout_rate: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim: int = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: FloatTensor) -> FloatTensor:
        out = self.net(x)
        out = nn.functional.normalize(out, p=2.0, dim=1)
        return out


# ---------------------------------------------------------------------------
# NeuralSamplerV2
# ---------------------------------------------------------------------------
class NeuralSamplerV2:
    """
    학습된 MLP v2를 이용한 goal-biased 샘플러.

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

        print("NeuralSamplerV2 initialized.")

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
    # Obstacle Distance
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
    # Normalize
    # --------------------------------------------------
    def _normalize(self, x: FloatTensor) -> FloatTensor:
        return (x - self.mean) / self.std

    # --------------------------------------------------
    # Clamp
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
        nearest_obstacle_distance = (
            self._compute_nearest_obstacle_distance(current)
        )

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

        return direction[0].tolist()

    # --------------------------------------------------
    # Sample
    # --------------------------------------------------
    def sample(self, current: Point3D) -> Point3D:
        direction: Point3D = self._predict_direction(current)

        sample_x = current[0] + direction[0] * self.sample_distance
        sample_y = current[1] + direction[1] * self.sample_distance
        sample_z = current[2] + direction[2] * self.sample_distance

        return self._clamp([sample_x, sample_y, sample_z])


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sampler = NeuralSamplerV2()

    test_points: list[Point3D] = [
        MAP_CONFIG["start"],
        [1.0, 0.5, -0.5],
        [2.5, 1.0, -0.8],
        [3.5, 1.5, -1.0],
    ]

    print("\n--- NeuralSamplerV2 test ---")
    print(f"Goal: {MAP_CONFIG['goal']}")
    print()

    for current in test_points:
        sample = sampler.sample(current)
        print(
            f"Current: {[round(v, 3) for v in current]}"
            f"  →  Sample: {[round(v, 3) for v in sample]}"
        )

# #! /usr/bin/env python3
# """
# Improved Neural Sampler (v2) using TorchScript model.

# neural_sampler.py 대비 개선사항:
#     - TorchScript 모델 로드 (sampling_mlp_v2_scripted.pt)
#     - torch.no_grad() 컨텍스트 유지
#     - SamplingMLP 클래스 정의 불필요 (scripted.pt에 구조 포함)

# Interface (기존과 동일):
#     sampler = NeuralSamplerV2()
#     point = sampler.sample(current_position)  # [x, y, z]
# """

# import math
# import os
# from typing import TypeAlias

# import torch

# from neural_rrt_planner.config import MAP_CONFIG

# # ---------------------------------------------------------------------------
# # Type aliases
# # ---------------------------------------------------------------------------
# Point3D: TypeAlias = list[float]
# FloatTensor: TypeAlias = torch.Tensor

# # ---------------------------------------------------------------------------
# # Paths
# # ---------------------------------------------------------------------------
# SCRIPTED_MODEL_PATH: str = os.path.expanduser(
#     "~/term_project/models/sampling_mlp_v2_scripted.pt"
# )
# NORMALIZER_PATH: str = os.path.expanduser(
#     "~/term_project/models/normalizer_v2.pth"
# )


# # ---------------------------------------------------------------------------
# # NeuralSamplerV2
# # ---------------------------------------------------------------------------
# class NeuralSamplerV2:
#     """
#     TorchScript 모델을 이용한 goal-biased 샘플러.

#     neural_sampler.py와 동일한 인터페이스를 유지하면서
#     TorchScript 변환으로 inference 속도를 개선한다.
#     """

#     def __init__(self) -> None:
#         self.map_config: dict = MAP_CONFIG
#         self.bounds: dict = MAP_CONFIG["bounds"]
#         self.goal: Point3D = MAP_CONFIG["goal"]
#         self.obstacles: list[dict] = MAP_CONFIG["obstacles"]
#         self.depth_limit: dict = MAP_CONFIG["depth_limit"]
#         self.safety_margin: float = MAP_CONFIG["safety_margin"]
#         self.sample_distance: float = MAP_CONFIG["neural_params"]["sample_distance"]

#         self.device = torch.device("cpu")

#         self.model = self._load_scripted_model()
#         self.mean: FloatTensor
#         self.std: FloatTensor
#         self._load_normalizer()

#         print("NeuralSamplerV2 initialized.")

#     # --------------------------------------------------
#     # Load
#     # --------------------------------------------------
#     def _load_scripted_model(self) -> torch.jit.ScriptModule:
#         """
#         TorchScript 모델 로드.
#         ScriptModule은 eval() 변환 시 이미 inference 모드로 고정되어
#         별도의 model.eval() 호출 불필요.
#         """
#         model = torch.jit.load(SCRIPTED_MODEL_PATH, map_location=self.device)
#         model.eval()
#         print(f"Scripted model loaded: {SCRIPTED_MODEL_PATH}")
#         return model

#     def _load_normalizer(self) -> None:
#         stats = torch.load(NORMALIZER_PATH, map_location=self.device)
#         self.mean = stats["mean"]
#         self.std = stats["std"]
#         print(f"Normalizer loaded: {NORMALIZER_PATH}")

#     # --------------------------------------------------
#     # Obstacle Distance
#     # --------------------------------------------------
#     def _compute_nearest_obstacle_distance(self, point: Point3D) -> float:
#         min_distance: float = float("inf")

#         px, py, pz = point[0], point[1], point[2]

#         for obstacle in self.obstacles:
#             if obstacle["type"] != "cylinder":
#                 continue

#             cx: float = obstacle["center"][0]
#             cy: float = obstacle["center"][1]
#             cz: float = obstacle["center"][2]

#             radius: float = obstacle["radius"] + self.safety_margin
#             height: float = obstacle["height"] + 2.0 * self.safety_margin

#             dx: float = px - cx
#             dy: float = py - cy

#             horizontal_distance: float = (
#                 math.sqrt(dx ** 2 + dy ** 2) - radius
#             )

#             z_min: float = cz - height / 2.0
#             z_max: float = cz + height / 2.0

#             if pz < z_min:
#                 vertical_distance: float = z_min - pz
#             elif pz > z_max:
#                 vertical_distance = pz - z_max
#             else:
#                 vertical_distance = 0.0

#             horizontal_distance = max(horizontal_distance, 0.0)

#             distance: float = math.sqrt(
#                 horizontal_distance ** 2 + vertical_distance ** 2
#             )

#             min_distance = min(min_distance, distance)

#         if min_distance == float("inf"):
#             return 10.0

#         return min_distance

#     # --------------------------------------------------
#     # Normalize
#     # --------------------------------------------------
#     def _normalize(self, x: FloatTensor) -> FloatTensor:
#         return (x - self.mean) / self.std

#     # --------------------------------------------------
#     # Clamp
#     # --------------------------------------------------
#     def _clamp(self, point: Point3D) -> Point3D:
#         x = max(self.bounds["x"][0], min(self.bounds["x"][1], point[0]))
#         y = max(self.bounds["y"][0], min(self.bounds["y"][1], point[1]))
#         z = max(self.depth_limit["z_min"], min(self.depth_limit["z_max"], point[2]))
#         return [x, y, z]

#     # --------------------------------------------------
#     # Inference
#     # --------------------------------------------------
#     def _predict_direction(self, current: Point3D) -> Point3D:
#         nearest_obstacle_distance = (
#             self._compute_nearest_obstacle_distance(current)
#         )

#         input_vec: FloatTensor = torch.tensor(
#             [[
#                 current[0],
#                 current[1],
#                 current[2],
#                 self.goal[0],
#                 self.goal[1],
#                 self.goal[2],
#                 nearest_obstacle_distance,
#             ]],
#             dtype=torch.float32,
#         )

#         input_normalized = self._normalize(input_vec)

#         with torch.no_grad():
#             direction: FloatTensor = self.model(input_normalized)

#         return direction[0].tolist()

#     # --------------------------------------------------
#     # Sample (기존과 동일한 인터페이스)
#     # --------------------------------------------------
#     def sample(self, current: Point3D) -> Point3D:
#         direction: Point3D = self._predict_direction(current)

#         sample_x = current[0] + direction[0] * self.sample_distance
#         sample_y = current[1] + direction[1] * self.sample_distance
#         sample_z = current[2] + direction[2] * self.sample_distance

#         return self._clamp([sample_x, sample_y, sample_z])


# # ---------------------------------------------------------------------------
# # Quick test
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     sampler = NeuralSamplerV2()

#     test_points: list[Point3D] = [
#         MAP_CONFIG["start"],
#         [1.0, 0.5, -0.5],
#         [2.5, 1.0, -0.8],
#         [3.5, 1.5, -1.0],
#     ]

#     print("\n--- NeuralSamplerV2 test ---")
#     print(f"Goal: {MAP_CONFIG['goal']}")
#     print()

#     for current in test_points:
#         sample = sampler.sample(current)
#         print(
#             f"Current: {[round(v, 3) for v in current]}"
#             f"  →  Sample: {[round(v, 3) for v in sample]}"
#         )