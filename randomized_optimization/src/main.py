import time

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.config.config import (
    ALGORITHM_MAPPING,
    NN_OPT_ALGORITHMS,
    PROBLEM_PARAMS_MAPPING,
    RANDOM_SEED,
)
from src.data.nba_dataset import NBADataset
from src.models.discrete_problems import (
    knapsack_problem,
    multi_queens_problem,
    solver,
    traveling_salesman_problem,
)
from src.models.neural_network import neural_network
from src.visualization.visualize import (
    plot_discrete_problem_fitness_curves,
    plot_walltime_chart,
)


def discrete_problem_analysis(problem_name):
    """Perform analysis and generate graphs for the discrete problem given the name"""
    PROBLEM_NAME_MAPPING = {
        "Traveling Salesman Problem": traveling_salesman_problem,
        "Knapsack Problem": knapsack_problem,
        "N-Queens Problem": multi_queens_problem,
    }
    # define a traveling salesman problem with 20 cities
    problem = PROBLEM_NAME_MAPPING[problem_name](**PROBLEM_PARAMS_MAPPING[problem_name])
    walltime_map_default = {}
    fitness_score_map_default = {}
    walltime_map_tuned = {}
    fitness_curve_map_tuned = {}

    # store fitness curves and wall times for the 4 algorithms
    for algorithm in ALGORITHM_MAPPING:
        # run the solver and get the fitness scores for default parameters set
        start_time = time.perf_counter()
        _, _, fitness_curve = solver(problem, algorithm, params_set="default")
        end_time = time.perf_counter()
        # store fitness curve and wall times
        fitness_score_map_default[algorithm] = fitness_curve
        walltime_map_default[algorithm] = end_time - start_time

        # run the solver and get the fitness scores for default parameters set
        start_time = time.perf_counter()
        _, _, fitness_curve = solver(problem, algorithm, params_set="tuned")
        end_time = time.perf_counter()
        # store fitness curve and wall times
        fitness_curve_map_tuned[algorithm] = fitness_curve
        walltime_map_tuned[algorithm] = end_time - start_time

    # plot the fitness scores
    _, axes = plt.subplots(4, 2, figsize=(10, 20))
    plot_discrete_problem_fitness_curves(fitness_score_map_default, axes[:, 0])  # type: ignore
    plot_discrete_problem_fitness_curves(fitness_curve_map_tuned, axes[:, 1])  # type: ignore
    plt.suptitle(f"Fitness Curves for Solving {problem_name}", fontsize=20)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig(f"./reports/figures/{problem_name}_fitness_curves.jpg", dpi=150)

    # plot the wall time the walltime bar chart
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_walltime_chart(walltime_map_default, ax1)
    plot_walltime_chart(walltime_map_tuned, ax2)
    plt.suptitle(f"Wall Time for Solving {problem_name}", fontsize=20)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig(f"./reports/figures/{problem_name}_wall_times.jpg", dpi=150)


def neural_network_analysis():
    """
    Perform analysis and generate graphs for predicting NBA players career duration
    using neural network, with the optimization algorithms subsituted with the
    4 randomized optimization algorithms
    """
    walltime_map = {}
    fitness_score_map = {}
    accuracy_score_map = {}

    # get training and test sets
    nba_dataset = NBADataset()
    X, y = nba_dataset.build_training_test_set()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    # store fitness curves and wall times for the 4 algorithms
    for algorithm in NN_OPT_ALGORITHMS:
        start_time = time.perf_counter()
        # run the fitted neural network and get fitness curves
        nn = neural_network(X_train, y_train, algorithm)
        end_time = time.perf_counter()
        # store fitness curve and wall times
        fitness_score_map[algorithm] = nn.fitness_curve
        walltime_map[algorithm] = end_time - start_time
        # get the accuracy score on test
        y_pred = nn.predict(X_test)
        accuracy_score_map[algorithm] = accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    discrete_problem_analysis("Traveling Salesman Problem")
