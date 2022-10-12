import time

from src.config.config import (
    ALGORITHM_MAPPING,
    PROBLEM_NAME_MAPPING,
    PROBLEM_PARAMS_MAPPING,
)
from src.models.discrete_problems import solver
from src.visualization.visualize import plot_fitness_curves, plot_walltime_chart


def discrete_problem_analysis(problem_name):
    """Perform analysis and generate graphs for the discrete problem given the name"""
    # define a traveling salesman problem with 20 cities
    problem = PROBLEM_NAME_MAPPING[problem_name](**PROBLEM_PARAMS_MAPPING[problem_name])
    walltime_map = {}
    fitness_score_map = {}

    # store fitness curves and wall times for the 4 algorithms
    for algorithm in ALGORITHM_MAPPING:
        start_time = time.perf_counter()
        # run the solver and get the fitness scores
        _, _, fitness_curve = solver(problem, algorithm)
        end_time = time.perf_counter()
        # store fitness curve and wall times
        fitness_score_map[algorithm] = fitness_curve
        walltime_map[algorithm] = end_time - start_time

    # plot the fitness scores and the walltime bar chart and save to figures
    plot_fitness_curves(problem_name, fitness_score_map)
    plot_walltime_chart(problem_name, walltime_map)
