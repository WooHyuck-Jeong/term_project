#! /usr/bin/env python3
"""
Benchmark: Neural RRT*

실행:
    python3 benchmark_neural_rrt_star.py
    python3 benchmark_neural_rrt_star.py --runs 20
    python3 benchmark_neural_rrt_star.py --runs 20 --max-iter 3000

출력:
    ~/term_project/benchmark/benchmark_neural_rrt_star_raw.csv
    ~/term_project/benchmark/benchmark_neural_rrt_star_summary.csv
"""

from neural_rrt_planner.planners.neural_rrt_star_v2 import NeuralRRTStar
from neural_rrt_planner.learning.benchmark_base import make_parser, run_benchmark


def main() -> None:
    args = make_parser("Neural RRT*").parse_args()

    # 모델 로드 시간이 첫 번째 run에만 포함되지 않도록 미리 워밍업
    print("Warming up Neural RRT* (model load)...")
    _ = NeuralRRTStar()
    print("Done.\n")

    run_benchmark(
        planner_name="Neural RRT* v2",
        planner_tag="neural_rrt_star_v2",
        planner_factory=NeuralRRTStar,
        runs=args.runs,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()