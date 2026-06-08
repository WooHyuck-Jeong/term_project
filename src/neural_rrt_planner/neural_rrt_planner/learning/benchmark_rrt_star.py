#! /usr/bin/env python3
"""
Benchmark: RRT*

실행:
    python3 benchmark_rrt_star.py
    python3 benchmark_rrt_star.py --runs 20
    python3 benchmark_rrt_star.py --runs 20 --max-iter 3000

출력:
    ~/term_project/benchmark/benchmark_rrt_star_raw.csv
    ~/term_project/benchmark/benchmark_rrt_star_summary.csv
"""

from neural_rrt_planner.planners.rrt_star import RRTStar
from neural_rrt_planner.learning.benchmark_base import make_parser, run_benchmark


def main() -> None:
    args = make_parser("RRT*").parse_args()

    run_benchmark(
        planner_name="RRT*",
        planner_tag="rrt_star",
        planner_factory=RRTStar,
        runs=args.runs,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()