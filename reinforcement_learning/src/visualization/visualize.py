import matplotlib.pyplot as plt
import numpy as np

from src.config.config import (  # pylint: disable=no-name-in-module
    CMAP,
    GOAL_STATE_MAP,
    TERM_STATE_MAP,
)


def visualize_policy(policy, shape, name, title=None):
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
