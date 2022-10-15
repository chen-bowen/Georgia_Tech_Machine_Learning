import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.config.config import (
    ALGORITHM_HYPERPARAMS_MAPPING,
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
from src.models.neural_network import feature_transformer, neural_network
from src.visualization.visualize import (
    plot_discrete_problem_evals_per_iter_chart,
    plot_discrete_problem_fitness_curves,
    plot_discrete_problem_scalability,
    plot_neural_network_accuracy_chart,
    plot_neural_network_fitness_curve,
    plot_neural_network_walltime_chart,
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
    walltime_map = defaultdict(list)
    fitness_score_map = defaultdict(dict)
    evals_per_iter_map = defaultdict()

    # store fitness curves and wall times for the 4 algorithms
    for algorithm in ALGORITHM_MAPPING:
        for params_set in ALGORITHM_HYPERPARAMS_MAPPING[algorithm]:
            # run the solver and get the fitness scores for default parameters set
            _, _, fitness_curve = solver(problem, algorithm, params_set)
            # store fitness curve and wall times
            fitness_score_map[algorithm][str(params_set)] = fitness_curve

        for problem_size in np.arange(5, 51, 5):
            # start counter and get the wall times for algorithm
            start_time = time.perf_counter()
            problem = PROBLEM_NAME_MAPPING[problem_name](problem_size)
            _, _, _ = solver(
                problem,
                algorithm,
                params_set=ALGORITHM_HYPERPARAMS_MAPPING[algorithm][1],
            )
            end_time = time.perf_counter()
            walltime_map[algorithm].append(end_time - start_time)

        # get evals per iteration
        evals_per_iter_map[algorithm] = round(
            fitness_curve[-1, 1] / len(fitness_curve[:, 1])
        )

    # plot the fitness scores
    _, axes = plt.subplots(2, 2, figsize=(20, 20))
    plot_discrete_problem_fitness_curves(fitness_score_map, axes)  # type: ignore
    plt.suptitle(f"Fitness Curves for Solving {problem_name}", fontsize=20)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig(f"./reports/figures/{problem_name}_fitness_curves.jpg", dpi=150)

    # plot the wall time the walltime bar chart
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plot_discrete_problem_scalability(walltime_map, ax1)
    plot_discrete_problem_evals_per_iter_chart(evals_per_iter_map, ax2)
    plt.title(f"Scalability for Solving {problem_name}")
    plt.savefig(f"./reports/figures/{problem_name}_scalability.jpg", dpi=150)


def neural_network_analysis():
    """
    Perform analysis and generate graphs for predicting NBA players career duration
    using neural network, with the optimization algorithms subsituted with the
    4 randomized optimization algorithms
    """
    walltime_map = {}
    fitness_score_map = {}
    train_accuracy_score_map = {}
    test_accuracy_score_map = {}

    # get training and test sets
    nba_dataset = NBADataset()
    X, y = nba_dataset.build_training_test_set()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    # preprocessing
    features = feature_transformer()
    X_train = features.fit_transform(X_train)
    X_test = features.transform(X_test)
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
        y_pred_train = nn.predict(X_train)
        y_pred_test = nn.predict(X_test)
        train_accuracy_score_map[algorithm] = accuracy_score(y_train, y_pred_train)
        test_accuracy_score_map[algorithm] = accuracy_score(y_test, y_pred_test)

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 22))
    plot_neural_network_fitness_curve(fitness_score_map, ax1)
    plot_neural_network_walltime_chart(walltime_map, ax2)
    plot_neural_network_accuracy_chart(
        train_accuracy_score_map, test_accuracy_score_map, ax3
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    plt.suptitle(
        "Neural Network Fitted With Random Optimization vs Gradient Descent",
        fontsize=20,
    )
    plt.savefig("./reports/figures/neural_network_curves.jpg", dpi=150)


if __name__ == "__main__":
    discrete_problem_analysis("Traveling Salesman Problem")
    discrete_problem_analysis("Knapsack Problem")
    discrete_problem_analysis("N-Queens Problem")
    neural_network_analysis()
