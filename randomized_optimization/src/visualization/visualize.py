import matplotlib.pyplot as plt
import numpy as np


def plot_discrete_problem_fitness_curves(fitness_score_map, axes):
    """
    Plot 4 fitness curves for solving the given problem using
    random hillclimbing, simulated annealing, genetic algorithm and MIMIC
    """

    # plot fitness scores vs iterations for 4 different algorithms
    for i, (algorithm, fitness_vals) in enumerate(fitness_score_map.items()):
        axes[i].plot(fitness_vals[:, 1], fitness_vals[:, 0])
        axes[i].set_title(f"{algorithm}")
        axes[i].set_xlabel("Number of Iterations")
        axes[i].set_ylabel("Fitness")

    return plt


def plot_neural_network_fitness_curve(fitness_score_map):
    """
    Plot 3 fitness curves for solving the given problem using
    random hillclimbing, simulated annealing, genetic algorithm
    """
    _, axes = plt.subplots(1, 3, figsize=(15, 10))

    # plot fitness scores vs iterations for 4 different algorithms
    for i, (algorithm, fitness_vals) in enumerate(fitness_score_map.items()):
        axes[i].plot(np.arange(0, len(fitness_vals)), fitness_vals)
        axes[i].set_title(f"{algorithm}")
        axes[i].set_xlabel("Number of Iterations")
        axes[i].set_ylabel("Fitness")

    plt.suptitle(
        "Fitness Curves for Finding Neural Network Weights with Random Optimization",
        fontsize=20,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig("../reports/figures/nn_fitness_curves.jpg", dpi=150)


def plot_walltime_chart(walltime_map, axes):
    """
    Plot 4 fitness curves for solving the given problem using
    random hillclimbing, simulated annealing, genetic algorithm and MIMIC
    """
    algorithms = list(walltime_map.keys())
    times = list(walltime_map.values())

    # Create bars
    bars = axes.bar(algorithms, times)

    # Create names on the axis and set title
    axes.set_xlabel("Algorithm")
    axes.set_ylabel("Wall Time (s)")
    axes.xticks(rotation=45)
    axes.bar_label(bars, padding=0.5)

    return plt
