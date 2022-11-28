import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.config import (  # pylint: disable=no-name-in-module, import-error
    CMAP,
    GOAL_STATE_MAP,
    TERM_STATE_MAP,
)


def visualize_policy_frozen_lake(policy, shape, name, title=None):
    """Visualize the policy"""
    M = shape[0]
    N = shape[1]
    actions = np.asarray(policy).reshape(shape)
    mapping = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    arr = np.zeros(shape)
    for i in range(M):
        for j in range(N):
            if N * i + j in TERM_STATE_MAP[name]:
                arr[i, j] = 0.25
            elif N * i + j in GOAL_STATE_MAP[name]:
                arr[i, j] = 1.0
    _, ax = plt.subplots(figsize=(10, 10))
    _ = ax.imshow(arr, cmap=CMAP)
    ax.set_xticks(np.arange(M))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(np.arange(M))
    ax.set_yticklabels(np.arange(N))
    ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="w", linewidth=2)

    for i in range(M):
        for j in range(N):
            if N * i + j in TERM_STATE_MAP[name]:
                ax.text(j, i, "H", ha="center", va="center", color="k", size=18)
            elif N * i + j in GOAL_STATE_MAP[name]:
                ax.text(j, i, "G", ha="center", va="center", color="k", size=18)
            else:
                ax.text(
                    j,
                    i,
                    mapping[actions[i, j]],
                    ha="center",
                    va="center",
                    color="k",
                    size=18,
                )
    # fig.tight_layout()
    if title:
        ax.set_title(title)
    return plt


def visualize_forest(policy, problem_size, title="Forest Management"):
    rows = int(np.sqrt(problem_size))
    cols = int(np.sqrt(problem_size))

    colors = {0: "g", 1: "k"}

    labels = {
        0: "W",
        1: "C",
    }

    # reshape policy array to be 2-D - assumes 500 states...
    policy = np.array(list(policy)).reshape(rows, cols)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, xlim=(-0.01, cols + 0.01), ylim=(-0.01, rows + 0.01))
    plt.title(title, fontsize=16, weight="bold", y=1.01)

    for i in range(rows):
        for j in range(cols):
            y = cols - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1, linewidth=1, edgecolor="k")
            p.set_facecolor(colors[policy[i, j]])
            ax.add_patch(p)

            _ = ax.text(
                x + 0.5,
                y + 0.5,
                labels[policy[i, j]],
                horizontalalignment="center",
                size=10,
                verticalalignment="center",
                color="w",
            )

    plt.axis("off")
    plt.savefig("./reports/figures/" + title + ".png", dpi=400)


def visualize_statistics_frozen_lake(file_path, algorithm):

    statistics = pd.read_csv(file_path)
    statistics = statistics[statistics["method"] == algorithm]
    _, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(statistics["gamma"], statistics["time"], "o-", color="r")
    axes[0, 0].set_title("Gamma vs Wall Time")

    axes[0, 1].plot(
        statistics["gamma"],
        statistics["iterations"],
        "o-",
    )
    axes[0, 1].set_title("Gamma vs Number of Iterations")

    axes[1, 0].plot(
        statistics["gamma"],
        statistics["reward"],
        "o-",
        color="k",
    )
    axes[1, 0].set_title("Gamma vs Rewards")

    axes[1, 1].plot(
        statistics["gamma"],
        statistics["success_rate"],
        "o-",
        color="b",
    )
    axes[1, 1].set_title("Gamma vs Success Rate")

    return plt


def visualize_statistics_forest(file_path, algorithm):

    statistics = pd.read_csv(file_path)
    statistics = statistics[statistics["method"] == algorithm]
    _, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].plot(statistics["gamma"], statistics["time"], "o-", color="r")
    axes[0].set_title("Gamma vs Wall Time")

    axes[1].plot(
        statistics["gamma"],
        statistics["iterations"],
        "o-",
    )
    axes[1].set_title("Gamma vs Number of Iterations")

    axes[2].plot(
        statistics["gamma"],
        statistics["reward"],
        "o-",
        color="k",
    )
    axes[2].set_title("Gamma vs Rewards")

    return plt
