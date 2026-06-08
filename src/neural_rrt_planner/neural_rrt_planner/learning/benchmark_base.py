#! /usr/bin/env python3
"""
Benchmark 공통 로직.

benchmark_rrt_star.py, benchmark_informed_rrt_star.py,
benchmark_neural_rrt_star.py 에서 공통으로 사용한다.
"""

import argparse
import csv
import os
import statistics
import time
from typing import Callable

from neural_rrt_planner.config import MAP_CONFIG


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
    planner_factory: Callable,
    run_index: int,
    max_iter: int,
) -> RunResult:

    planner = planner_factory()

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

    return {
        "planner": planner_name,
        "total_runs": total_runs,
        "success_rate": success_rate,
        "first_path_iter": stats([r.first_path_iter for r in found]),
        "final_cost": stats([r.final_cost for r in found if r.final_cost is not None]),
        "final_nodes": stats([float(r.final_nodes) for r in results]),
        "elapsed_sec": stats([r.elapsed_sec for r in results]),
    }


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
def print_summary(summary: dict) -> None:
    def fmt(s: dict, unit: str = "", precision: int = 2) -> str:
        if s["mean"] is None:
            return "N/A"
        return (
            f"{s['mean']:.{precision}f}{unit} "
            f"± {s['std']:.{precision}f} "
            f"[{s['min']:.{precision}f} ~ {s['max']:.{precision}f}]"
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
    results: list[RunResult],
    summary: dict,
    output_dir: str,
    planner_tag: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Raw
    raw_path = os.path.join(output_dir, f"benchmark_{planner_tag}_raw.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "planner", "run",
            "first_path_iter", "final_cost",
            "final_nodes", "elapsed_sec",
        ])
        writer.writeheader()
        for r in results:
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
    summary_path = os.path.join(output_dir, f"benchmark_{planner_tag}_summary.csv")

    def m(s: dict, p: int = 4) -> str:
        return f"{s['mean']:.{p}f}" if s["mean"] is not None else "N/A"

    def sd(s: dict, p: int = 4) -> str:
        return f"{s['std']:.{p}f}" if s["std"] is not None else "N/A"

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "planner", "runs", "success_rate",
            "first_path_iter_mean", "first_path_iter_std",
            "final_cost_mean", "final_cost_std",
            "final_nodes_mean", "final_nodes_std",
            "elapsed_sec_mean", "elapsed_sec_std",
        ])
        writer.writeheader()
        writer.writerow({
            "planner": summary["planner"],
            "runs": summary["total_runs"],
            "success_rate": f"{summary['success_rate']:.1f}",
            "first_path_iter_mean": m(summary["first_path_iter"], 1),
            "first_path_iter_std": sd(summary["first_path_iter"], 1),
            "final_cost_mean": m(summary["final_cost"], 4),
            "final_cost_std": sd(summary["final_cost"], 4),
            "final_nodes_mean": m(summary["final_nodes"], 1),
            "final_nodes_std": sd(summary["final_nodes"], 1),
            "elapsed_sec_mean": m(summary["elapsed_sec"], 3),
            "elapsed_sec_std": sd(summary["elapsed_sec"], 3),
        })
    print(f"Summary saved     : {summary_path}")


# ---------------------------------------------------------------------------
# Common argument parser
# ---------------------------------------------------------------------------
def make_parser(planner_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Benchmark {planner_name}"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs (default: 10)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=MAP_CONFIG["rrt_params"]["max_iter"],
        help=f"Max iterations per run (default: {MAP_CONFIG['rrt_params']['max_iter']})",
    )
    return parser


# ---------------------------------------------------------------------------
# Common run loop
# ---------------------------------------------------------------------------
def run_benchmark(
    planner_name: str,
    planner_tag: str,
    planner_factory: Callable,
    runs: int,
    max_iter: int,
) -> None:

    print(f"\nBenchmark: {planner_name}")
    print(f"  Runs     : {runs}")
    print(f"  Max iter : {max_iter}")
    print()

    results: list[RunResult] = []

    for i in range(1, runs + 1):
        result = run_once(planner_name, planner_factory, i, max_iter)
        results.append(result)

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
    print_summary(summary)

    output_dir = os.path.expanduser("~/term_project/benchmark")
    save_csv(results, summary, output_dir, planner_tag)