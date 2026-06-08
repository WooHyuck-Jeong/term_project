#! /usr/bin/env python3
"""
Benchmark: Informed RRT*

실행:
    python3 benchmark_informed_rrt_star.py
    python3 benchmark_informed_rrt_star.py --runs 20
    python3 benchmark_informed_rrt_star.py --runs 20 --max-iter 3000

출력:
    ~/term_project/benchmark/benchmark_informed_rrt_star_raw.csv
    ~/term_project/benchmark/benchmark_informed_rrt_star_summary.csv
"""

from neural_rrt_planner.planners.informed_rrt_star import InformedRRTStar
from neural_rrt_planner.learning.benchmark_base import make_parser, run_benchmark


def main() -> None:
    args = make_parser("Informed RRT*").parse_args()

    run_benchmark(
        planner_name="Informed RRT*",
        planner_tag="informed_rrt_star",
        planner_factory=InformedRRTStar,
        runs=args.runs,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()