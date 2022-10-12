import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_curves(problem_name, fitness_score_map):
    """
    Plot 4 fitness curves and 4 wall time curves for solving the given problem using
    random hillclimbing, simulated annealing, genetic algorithm and MIMIC
    """
    _, axes = plt.subplots(1, 4, figsize=(15, 15))

    # plot fitness scores vs iterations for 4 different algorithms
    for i, (algorithm, fitness_vals) in enumerate(fitness_score_map.items()):
        axes[i].plot(np.arange(0, len(fitness_vals)), fitness_vals)
        axes[i].set_title(f"{algorithm}")  # type: ignore
        axes[i].set_xlabel("Number of Iterations")  # type: ignore
        axes[i].set_ylabel("Fitness")  # type: ignore

    plt.suptitle(f"Fitness Curves for Solving {problem_name}", fontsize=20)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig(f"../reports/figures/{problem_name}_fitness_curves.jpg", dpi=150)


def plot_walltime_chart(problem_name, walltime_map):
    """
    Plot 4 fitness curves and 4 wall time curves for solving the given problem using
    random hillclimbing, simulated annealing, genetic algorithm and MIMIC
    """
    algorithms = list(walltime_map.keys())
    times = list(walltime_map.values())

    # Create bars
    plt.bar(algorithms, times)

    # Create names on the axis and set title
    plt.xlabel("Algorithm")
    plt.ylabel("Wall Time (s)")
    plt.title(f"Wall Time for Solving {problem_name}")

    # save figure
    plt.savefig(f"../reports/figures/{problem_name}_wall_times.jpg", dpi=150)
