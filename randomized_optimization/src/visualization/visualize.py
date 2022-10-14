import matplotlib.pyplot as plt
import numpy as np


def plot_discrete_problem_fitness_curves(fitness_score_map, axes):
    """
    Plot 4 fitness curves for solving the given problem using
    random hillclimbing, simulated annealing, genetic algorithm and MIMIC
    """
    # plot fitness scores vs iterations for 4 different algorithms
    axes = axes.flatten()
    for i, (algorithm, fitness_vals_for_algo) in enumerate(fitness_score_map.items()):
        for params_set, fitness_vals in fitness_vals_for_algo.items():
            # create plot using the corresponding parameters value
            params_name, params_val = (
                params_set.split(":")[0][2:-1],
                params_set.split(":")[1][1:-1],
            )
            axes[i].plot(
                fitness_vals[:, 0],
                label=f"{params_name}={params_val}",
            )
        axes[i].set_title(f"{algorithm}")
        axes[i].set_xlabel("Number of Iterations")
        axes[i].set_ylabel("Fitness")
        axes[i].legend()
        plt.sca(axes[i])
    return axes


def plot_discrete_problem_scalability(size_walltimes_map):
    """
    Plot problem size vs wall time curve
    """
    for algorithm, size_walltime in size_walltimes_map.items():
        plt.plot(np.arange(5, 51, 5), size_walltime, label=f"{algorithm}")
    plt.xlabel("Problem Size (n)")
    plt.ylabel("Wall Time (s)")
    plt.legend()
    return plt


def plot_neural_network_fitness_curve(fitness_score_map, axes):
    """
    Plot 3 fitness curves for solving the given problem using
    random hillclimbing, simulated annealing, genetic algorithm
    """
    # plot fitness scores vs iterations for 4 different algorithms
    for algorithm, fitness_vals in fitness_score_map.items():
        fitness_vals = (
            fitness_vals[:, 0] if algorithm != "gradient_descent" else -fitness_vals
        )
        axes.plot(fitness_vals, label=f"{algorithm}")
    axes.set_xlabel("Number of Iterations")
    axes.set_ylabel("Loss")
    axes.set_title("Fitness Curves")
    axes.legend()
    return plt


def plot_neural_network_walltime_chart(walltime_map, axes):
    """
    Plot 4 fitness curves for solving the given problem using
    random hillclimbing, simulated annealing, genetic algorithm and MIMIC
    """
    algorithms = list(walltime_map.keys())
    times = list(map(lambda x: round(x, 2), walltime_map.values()))

    # Create bars
    bars = axes.bar(algorithms, times)

    # Create names on the axis and set title
    axes.set_xlabel("Algorithm")
    axes.set_ylabel("Wall Time (s)")
    axes.tick_params(axis="x", rotation=30)
    axes.bar_label(bars, padding=0.5)
    axes.set_title("Wall Time")

    return plt


def plot_neural_network_accuracy_chart(
    train_accuracy_score_map, test_accuracy_score_map, axes
):
    """
    Plot 4 fitness curves for solving the given problem using
    random hillclimbing, simulated annealing, genetic algorithm and MIMIC
    """
    algorithms = list(train_accuracy_score_map.keys())
    train_accuracy_score = list(
        map(lambda x: round(x, 2), train_accuracy_score_map.values())
    )
    test_accuracy_score_map = list(
        map(lambda x: round(x, 2), test_accuracy_score_map.values())
    )

    # Create bars
    X_axis = np.arange(len(algorithms))
    bars1 = axes.bar(X_axis - 0.2, train_accuracy_score, 0.4, label="Train Accuracy")
    bars2 = axes.bar(X_axis + 0.2, test_accuracy_score_map, 0.4, label="Test Accuracy")

    # Create names on the axis and set title
    plt.xticks(X_axis, algorithms)
    axes.set_xlabel("Algorithm")
    axes.set_ylabel("Accuracy")
    axes.tick_params(axis="x", rotation=30)
    axes.bar_label(bars1, padding=0.5)
    axes.bar_label(bars2, padding=0.5)
    axes.set_title("Accuracy of Train and Test")

    return plt
