#! /usr/bin/env python3
"""
Benchmark Script: RRT* vs Informed RRT* vs Neural RRT*

비교 지표:
    - 첫 경로 발견까지 걸린 iteration 수
    - 최종 경로 cost
    - 최종 노드 수

실행 방법:
    python3 benchmark.py
    python3 benchmark.py --runs 20
    python3 benchmark.py --runs 20 --max-iter 3000
"""

import argparse
import csv
import os
import statistics
import time

from neural_rrt_planner.config import MAP_CONFIG
from neural_rrt_planner.planners.rrt_star import RRTStar
from neural_rrt_planner.planners.informed_rrt_star import InformedRRTStar
from neural_rrt_planner.planners.neural_rrt_star import NeuralRRTStar


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
class RunResult:
    def __init__(
        self,
        planner_name: str,
        run_index: int,
        first_path_iter: int | None,
        final_cost: float | None,
        final_nodes: int,
        elapsed_sec: float,
    ) -> None:
        self.planner_name = planner_name
        self.run_index = run_index
        self.first_path_iter = first_path_iter
        self.final_cost = final_cost
        self.final_nodes = final_nodes
        self.elapsed_sec = elapsed_sec


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_once(
    planner_name: str,
    run_index: int,
    max_iter: int,
) -> RunResult:

    if planner_name == "RRT*":
        planner = RRTStar()
    elif planner_name == "Informed RRT*":
        planner = InformedRRTStar()
    elif planner_name == "Neural RRT*":
        planner = NeuralRRTStar()
    else:
        raise ValueError(f"Unknown planner: {planner_name}")

    first_path_iter: int | None = None
    start_time = time.perf_counter()

    for i in range(1, max_iter + 1):
        planner.expand_once()

        if first_path_iter is None and planner.best_goal_node is not None:
            first_path_iter = i

    elapsed_sec = time.perf_counter() - start_time

    final_cost = (
        planner.best_goal_node.cost
        if planner.best_goal_node is not None
        else None
    )

    return RunResult(
        planner_name=planner_name,
        run_index=run_index,
        first_path_iter=first_path_iter,
        final_cost=final_cost,
        final_nodes=len(planner.nodes),
        elapsed_sec=elapsed_sec,
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(results: list[RunResult]) -> dict:
    planner_name = results[0].planner_name
    total_runs = len(results)

    # 경로를 찾은 run만 따로
    found = [r for r in results if r.first_path_iter is not None]
    success_rate = len(found) / total_runs * 100.0

    def stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    first_iter_stats = stats(
        [r.first_path_iter for r in found]
    )
    cost_stats = stats(
        [r.final_cost for r in found if r.final_cost is not None]
    )
    nodes_stats = stats(
        [float(r.final_nodes) for r in results]
    )
    time_stats = stats(
        [r.elapsed_sec for r in results]
    )

    return {
        "planner": planner_name,
        "total_runs": total_runs,
        "success_rate": success_rate,
        "first_path_iter": first_iter_stats,
        "final_cost": cost_stats,
        "final_nodes": nodes_stats,
        "elapsed_sec": time_stats,
    }


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
def print_summary(summary: dict) -> None:
    def fmt(stats: dict, unit: str = "", precision: int = 2) -> str:
        if stats["mean"] is None:
            return "N/A"
        return (
            f"{stats['mean']:.{precision}f}{unit} "
            f"± {stats['std']:.{precision}f} "
            f"[{stats['min']:.{precision}f} ~ {stats['max']:.{precision}f}]"
        )

    print(f"\n{'=' * 60}")
    print(f"  {summary['planner']}")
    print(f"{'=' * 60}")
    print(f"  Runs          : {summary['total_runs']}")
    print(f"  Success rate  : {summary['success_rate']:.1f}%")
    print(f"  First path    : {fmt(summary['first_path_iter'], ' iter', 1)}")
    print(f"  Final cost    : {fmt(summary['final_cost'], '', 4)}")
    print(f"  Final nodes   : {fmt(summary['final_nodes'], '', 1)}")
    print(f"  Elapsed       : {fmt(summary['elapsed_sec'], ' s', 3)}")


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
def save_csv(
    all_results: list[RunResult],
    all_summaries: list[dict],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Raw results
    raw_path = os.path.join(output_dir, "benchmark_raw.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "planner", "run",
            "first_path_iter", "final_cost",
            "final_nodes", "elapsed_sec",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "planner": r.planner_name,
                "run": r.run_index,
                "first_path_iter": r.first_path_iter if r.first_path_iter is not None else "N/A",
                "final_cost": f"{r.final_cost:.6f}" if r.final_cost is not None else "N/A",
                "final_nodes": r.final_nodes,
                "elapsed_sec": f"{r.elapsed_sec:.4f}",
            })
    print(f"\nRaw results saved : {raw_path}")

    # Summary
    summary_path = os.path.join(output_dir, "benchmark_summary.csv")

    def mean_or_na(stats: dict, precision: int = 4) -> str:
        if stats["mean"] is None:
            return "N/A"
        return f"{stats['mean']:.{precision}f}"

    def std_or_na(stats: dict, precision: int = 4) -> str:
        if stats["std"] is None:
            return "N/A"
        return f"{stats['std']:.{precision}f}"

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "planner", "runs", "success_rate",
            "first_path_iter_mean", "first_path_iter_std",
            "final_cost_mean", "final_cost_std",
            "final_nodes_mean", "final_nodes_std",
            "elapsed_sec_mean", "elapsed_sec_std",
        ])
        writer.writeheader()
        for s in all_summaries:
            writer.writerow({
                "planner": s["planner"],
                "runs": s["total_runs"],
                "success_rate": f"{s['success_rate']:.1f}",
                "first_path_iter_mean": mean_or_na(s["first_path_iter"], 1),
                "first_path_iter_std": std_or_na(s["first_path_iter"], 1),
                "final_cost_mean": mean_or_na(s["final_cost"], 4),
                "final_cost_std": std_or_na(s["final_cost"], 4),
                "final_nodes_mean": mean_or_na(s["final_nodes"], 1),
                "final_nodes_std": std_or_na(s["final_nodes"], 1),
                "elapsed_sec_mean": mean_or_na(s["elapsed_sec"], 3),
                "elapsed_sec_std": std_or_na(s["elapsed_sec"], 3),
            })
    print(f"Summary saved     : {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RRT* vs Informed RRT* vs Neural RRT*"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per planner (default: 10)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=MAP_CONFIG["rrt_params"]["max_iter"],
        help=f"Max iterations per run (default: {MAP_CONFIG['rrt_params']['max_iter']})",
    )
    args = parser.parse_args()

    planners = ["RRT*", "Informed RRT*", "Neural RRT*"]

    print(f"\nBenchmark started")
    print(f"  Planners  : {', '.join(planners)}")
    print(f"  Runs      : {args.runs} per planner")
    print(f"  Max iter  : {args.max_iter}")

    # Neural RRT*는 NeuralSampler 초기화 시 모델을 로드함
    # 첫 run에서 로드 시간이 포함되지 않도록 미리 워밍업
    print("\nWarming up Neural RRT* (model load)...")
    _ = NeuralRRTStar()
    print("Done.\n")

    all_results: list[RunResult] = []
    all_summaries: list[dict] = []

    for planner_name in planners:
        print(f"\n[{planner_name}] Running {args.runs} trials...")
        results: list[RunResult] = []

        for i in range(1, args.runs + 1):
            result = run_once(planner_name, i, args.max_iter)
            results.append(result)
            all_results.append(result)

            status = (
                f"iter={result.first_path_iter}"
                if result.first_path_iter is not None
                else "NOT FOUND"
            )
            cost = (
                f"{result.final_cost:.4f}"
                if result.final_cost is not None
                else "N/A"
            )
            print(
                f"  Run {i:>3} | "
                f"first_path={status:>12} | "
                f"cost={cost:>8} | "
                f"nodes={result.final_nodes:>5} | "
                f"time={result.elapsed_sec:.2f}s"
            )

        summary = summarize(results)
        all_summaries.append(summary)
        print_summary(summary)

    # 전체 비교 요약
    print(f"\n{'=' * 60}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"  {'Planner':<16} | "
        f"{'Success':>8} | "
        f"{'1st iter':>10} | "
        f"{'Cost':>10} | "
        f"{'Nodes':>8} | "
        f"{'Time(s)':>8}"
    )
    print(f"  {'-' * 16}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}")

    for s in all_summaries:
        def mfmt(stats, p=2):
            return f"{stats['mean']:.{p}f}" if stats["mean"] is not None else "N/A"

        print(
            f"  {s['planner']:<16} | "
            f"  {s['success_rate']:>5.1f}% | "
            f"{mfmt(s['first_path_iter'], 1):>10} | "
            f"{mfmt(s['final_cost'], 4):>10} | "
            f"{mfmt(s['final_nodes'], 1):>8} | "
            f"{mfmt(s['elapsed_sec'], 2):>8}"
        )

    output_dir = os.path.expanduser("~/term_project/benchmark")
    save_csv(all_results, all_summaries, output_dir)


if __name__ == "__main__":
    main()