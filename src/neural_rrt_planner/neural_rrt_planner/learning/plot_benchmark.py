#!/usr/bin/env python3
"""
벤치마크 결과 시각화 스크립트.

5개 플래너의 3가지 지표를 matplotlib으로 플롯하여
PNG 이미지로 저장한다.

Output:
    ~/term_project/benchmark/plot_first_path.png
    ~/term_project/benchmark/plot_final_cost.png
    ~/term_project/benchmark/plot_final_nodes.png
    ~/term_project/benchmark/plot_combined.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
PLANNERS = ["RRT*", "Informed\nRRT*", "Neural\nRRT* v1", "Neural\nRRT* v2", "Neural\nRRT* v2+TS"]
PLANNERS_SHORT = ["RRT*", "Informed RRT*", "Neural v1", "Neural v2", "Neural v2+TS"]

DATA = {
    "first_path": {
        "mean":  [103.2, 278.7, 126.9, 26.5,  61.7],
        "std":   [27.6,  257.0, 98.5,  6.2,   46.5],
        "label": "First Path Found (iter, lower is better)",
        "unit":  "iter",
    },
    "final_cost": {
        "mean":  [4.6851, 4.5842, 4.6269, 4.4868, 4.5408],
        "std":   [0.0610, 0.0449, 0.0370, 0.0270, 0.0739],
        "label": "Final Path Cost (lower is better)",
        "unit":  "cost",
    },
    "final_nodes": {
        "mean":  [1986.4, 1987.4, 1909.9, 1996.2, 1963.7],
        "std":   [10.3,   7.3,    84.9,   5.8,    41.4],
        "label": "Final Node Count (lower is better)",
        "unit":  "nodes",
    },
}

# 플래너별 색상
COLORS = ["#065A82", "#1C7293", "#8B5CF6", "#10B981", "#0ABFBC"]

OUTPUT_DIR = os.path.expanduser("~/term_project/benchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.color":        "#E2E8F0",
    "grid.linewidth":    0.8,
    "axes.facecolor":    "#FAFAFA",
    "figure.facecolor":  "white",
})

# ---------------------------------------------------------------------------
# Helper: single bar chart
# ---------------------------------------------------------------------------
def plot_single(
    key: str,
    filename: str,
    figsize: tuple = (8, 5),
    ymin: float | None = None,
) -> None:
    d = DATA[key]
    means = d["mean"]
    stds  = d["std"]
    label = d["label"]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(PLANNERS))
    bars = ax.bar(
        x, means,
        color=COLORS,
        width=0.55,
        zorder=3,
        edgecolor="white",
        linewidth=1.2,
    )

    # error bars
    ax.errorbar(
        x, means, yerr=stds,
        fmt="none",
        color="#334155",
        capsize=5,
        capthick=1.5,
        linewidth=1.5,
        zorder=4,
    )

    # value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + (max(means) * 0.015),
            f"{mean:.4f}" if key == "final_cost" else f"{mean:.1f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color="#1E293B",
        )

    # best marker (최솟값)
    best_idx = int(np.argmin(means))
    ax.text(
        x[best_idx],
        -max(means) * 0.08,
        "★ Best",
        ha="center", va="top",
        fontsize=9, color=COLORS[best_idx],
        fontweight="bold",
        transform=ax.get_xaxis_transform(),
    )

    if ymin is not None:
        ax.set_ylim(bottom=ymin)

    ax.set_xticks(x)
    ax.set_xticklabels(PLANNERS, fontsize=11)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(label, fontsize=14, fontweight="bold", pad=12, color="#1E293B")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Combined 3-panel figure
# ---------------------------------------------------------------------------
def plot_combined() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        "Benchmark Results Comparison  (mean of 10 runs, max_iter=2000)",
        fontsize=14, fontweight="bold", color="#1E293B", y=1.02
    )

    configs = [
        ("first_path",  None,   True),
        ("final_cost",  4.3,    True),
        ("final_nodes", 1800.0, False),
    ]

    for ax, (key, ymin, show_best) in zip(axes, configs):
        d = DATA[key]
        means = np.array(d["mean"])
        stds  = np.array(d["std"])

        x = np.arange(len(PLANNERS_SHORT))
        bars = ax.bar(
            x, means,
            color=COLORS,
            width=0.58,
            zorder=3,
            edgecolor="white",
            linewidth=1.0,
        )

        ax.errorbar(
            x, means, yerr=stds,
            fmt="none",
            color="#334155",
            capsize=4,
            capthick=1.2,
            linewidth=1.2,
            zorder=4,
        )

        # value labels
        for bar, mean, std in zip(bars, means, stds):
            offset = std + (means.max() - (ymin or 0)) * 0.018
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{mean:.4f}" if key == "final_cost" else f"{mean:.1f}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
                color="#1E293B",
            )

        # best highlight
        if show_best:
            best_idx = int(np.argmin(means))
            bars[best_idx].set_edgecolor(COLORS[best_idx])
            bars[best_idx].set_linewidth(2.5)
            ax.text(
                x[best_idx],
                0.01,
                "★",
                ha="center", va="bottom",
                fontsize=11, color=COLORS[best_idx],
                fontweight="bold",
                transform=ax.get_xaxis_transform(),
            )

        if ymin is not None:
            ax.set_ylim(bottom=ymin)

        ax.set_xticks(x)
        ax.set_xticklabels(PLANNERS_SHORT, fontsize=8.5, rotation=15, ha="right")
        ax.set_title(d["label"], fontsize=12, fontweight="bold",
                     pad=10, color="#1E293B")
        ax.set_ylabel(d["unit"] if d["unit"] else "value", fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "plot_combined.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plot_single("first_path",  "plot_first_path.png",  ymin=0)
    plot_single("final_cost",  "plot_final_cost.png",  ymin=4.3)
    plot_single("final_nodes", "plot_final_nodes.png", ymin=1800)
    plot_combined()
    print("\nAll plots saved.")