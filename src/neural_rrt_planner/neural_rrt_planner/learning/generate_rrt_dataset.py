#! /usr/bin/env python3


import argparse
import csv
import math
import os
from typing import TypeAlias

from neural_rrt_planner.config import MAP_CONFIG
from neural_rrt_planner.planners.rrt_star import RRTStar

Point3D: TypeAlias = list[float]
Direction3D: TypeAlias = list[float]

"""
Improved Sampling Dataset Generator using RRT* planned paths.

기존 방식과의 차이:
    기존: current 위치를 random 샘플링 → 타겟 = current → goal 단위벡터
    개선: RRT*를 N번 실행 → 실제 경로의 waypoint 쌍에서 샘플링
          타겟 = current → next_waypoint 단위벡터
          → 장애물을 실제로 우회한 방향을 학습

Dataset columns (동일하게 유지):
    Input  (7): current_x, current_y, current_z,
                goal_x, goal_y, goal_z,
                nearest_obstacle_distance
    Output (3): target_dx, target_dy, target_dz

실행:
    python3 generate_rrt_dataset.py
    python3 generate_rrt_dataset.py --runs 200 --output rrt_dataset.csv
"""


class RRTDatasetGenerator:
    """
    RRT* 플래닝 결과 기반 데이터셋 생성기.

    RRT*를 num_runs번 실행하고, 각 실행에서 얻은 경로의
    연속된 waypoint 쌍 (current, next)으로부터 학습 데이터를 생성한다.

    타겟이 실제 장애물 우회 경로에서 나오기 때문에
    기존 random 샘플링 방식보다 장애물 주변에서의
    방향 예측 품질이 향상된다.
    """

    def __init__(self) -> None:
        self.map_config: dict = MAP_CONFIG
        self.obstacles: list[dict] = MAP_CONFIG["obstacles"]
        self.goal: Point3D = MAP_CONFIG["goal"]
        self.safety_margin: float = MAP_CONFIG["safety_margin"]
        self.max_iter: int = MAP_CONFIG["rrt_params"]["max_iter"]

        self.dataset_dir: str = os.path.expanduser(
            "~/term_project/datasets"
        )
        os.makedirs(self.dataset_dir, exist_ok=True)

    # --------------------------------------------------
    # Obstacle distance (generate_sampling_dataset.py와 동일)
    # --------------------------------------------------
    def _compute_nearest_obstacle_distance(
        self,
        point: Point3D,
    ) -> float:
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
    # Unit direction vector
    # --------------------------------------------------
    def _compute_direction(
        self,
        current: Point3D,
        target: Point3D,
    ) -> Direction3D:
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        dz = target[2] - current[2]

        norm = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if norm < 1e-9:
            return [0.0, 0.0, 0.0]

        return [dx / norm, dy / norm, dz / norm]

    # --------------------------------------------------
    # Extract rows from a single RRT* path
    # --------------------------------------------------
    def _extract_rows_from_path(
        self,
        path: list[Point3D],
    ) -> list[list[float]]:
        """
        경로의 연속된 waypoint 쌍에서 데이터 행을 생성한다.

        waypoint[i]를 current로,
        waypoint[i+1]을 next로 사용하여
        타겟 방향 = current → next 단위벡터를 계산한다.

        Args:
            path: RRT* 경로 waypoint 리스트

        Returns:
            list of CSV rows
        """
        rows: list[list[float]] = []

        for i in range(len(path) - 1):
            current: Point3D = path[i]
            next_wp: Point3D = path[i + 1]

            direction: Direction3D = self._compute_direction(
                current, next_wp
            )

            # 방향이 0벡터면 스킵
            if direction == [0.0, 0.0, 0.0]:
                continue

            nearest_obstacle_distance = (
                self._compute_nearest_obstacle_distance(current)
            )

            row: list[float] = [
                current[0],
                current[1],
                current[2],
                self.goal[0],
                self.goal[1],
                self.goal[2],
                nearest_obstacle_distance,
                direction[0],
                direction[1],
                direction[2],
            ]

            rows.append(row)

        return rows

    # --------------------------------------------------
    # Generate
    # --------------------------------------------------
    def generate(
        self,
        num_runs: int = 200,
        output_filename: str = "rrt_dataset.csv",
    ) -> None:
        """
        RRT*를 num_runs번 실행하고
        각 실행의 경로에서 데이터를 수집하여 CSV로 저장한다.

        Args:
            num_runs:        RRT* 실행 횟수
            output_filename: 저장할 CSV 파일명
        """
        output_path = os.path.join(self.dataset_dir, output_filename)

        header: list[str] = [
            "current_x", "current_y", "current_z",
            "goal_x", "goal_y", "goal_z",
            "nearest_obstacle_distance",
            "target_dx", "target_dy", "target_dz",
        ]

        total_rows: int = 0
        success_runs: int = 0
        failed_runs: int = 0

        with open(output_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)

            for run_idx in range(1, num_runs + 1):
                planner = RRTStar()

                for _ in range(self.max_iter):
                    planner.expand_once()

                if planner.best_goal_node is None:
                    failed_runs += 1
                    if run_idx % 20 == 0:
                        print(
                            f"Run {run_idx:>4} / {num_runs} | "
                            f"FAILED | "
                            f"Total rows: {total_rows}"
                        )
                    continue

                success_runs += 1
                rows = self._extract_rows_from_path(planner.final_path)

                for row in rows:
                    writer.writerow(row)

                total_rows += len(rows)

                if run_idx % 20 == 0:
                    print(
                        f"Run {run_idx:>4} / {num_runs} | "
                        f"Path cost: {planner.best_goal_node.cost:.3f} | "
                        f"Waypoints: {len(planner.final_path):>3} | "
                        f"Total rows: {total_rows}"
                    )

        print()
        print(f"Dataset generation complete.")
        print(f"  Success runs  : {success_runs} / {num_runs}")
        print(f"  Failed runs   : {failed_runs} / {num_runs}")
        print(f"  Total rows    : {total_rows}")
        print(f"  Saved to      : {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RRT* path-based sampling dataset"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of RRT* runs (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rrt_dataset.csv",
        help="Output CSV filename (default: rrt_dataset.csv)",
    )
    args = parser.parse_args()

    generator = RRTDatasetGenerator()
    generator.generate(
        num_runs=args.runs,
        output_filename=args.output,
    )


if __name__ == "__main__":
    main()