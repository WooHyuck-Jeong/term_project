import os
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

dataDir = "/Users/woohyuck/Documents/02_workspace/04_Lecture/01_Mobility/MyProject/term_project/benchmark"
dataList = glob(os.path.join(dataDir, "*.csv"))

# rawDataIdx = [2, 4, 0, 1]
rawDataIdx = [2, 4, 0]
rawDataList = [dataList[i] for i in rawDataIdx]

rawData = [pd.read_csv(f) for f in rawDataList]

rrt_star = rawData[0].copy()
informed_rrt_star = rawData[1].copy()
neural_rrt_v1 = rawData[2].copy()
# neural_rrt_v2 = rawData[3].copy()

save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

algorithms = [
    (rrt_star, "rrt*", "red", "+"),
    (informed_rrt_star, "Informed RRT*", "blue", "^"),
    (neural_rrt_v1, "Neural RRT* v1", "green", "d"),
    # (neural_rrt_v2, "Neural RRT* v2", "orange", "o"),
]


def save_plot(y_col, title, filename, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    for df, label, color, marker in algorithms:
        df.plot(
            kind="line",
            x="run",
            y=y_col,
            ax=ax,
            label=label,
            c=color,
            marker=marker,
            grid=True
        )

    ax.set_title(title)
    ax.legend(loc="upper right")

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    fig.savefig(
        os.path.join(save_dir, filename),
        dpi=500,
        bbox_inches="tight"
    )
    plt.close(fig)


# 1. first_path_iter
save_plot(
    y_col="first_path_iter",
    title="First Path Iter",
    filename="first_path_iter_v1.png",
    ylim=(0, 500)
)

# 2. final_cost
save_plot(
    y_col="final_cost",
    title="Final Cost",
    filename="final_cost_v1.png"
)

# 3. final_nodes
save_plot(
    y_col="final_nodes",
    title="Final Nodes",
    filename="final_nodes_v1.png",
    ylim=(1700, 2100)
)

# 4. elapsed_sec
save_plot(
    y_col="elapsed_sec",
    title="Elapsed Time [s]",
    filename="elapsed_sec_v1.png"
)

print("All plots saved.")