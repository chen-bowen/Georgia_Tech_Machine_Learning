import time

from src.config.config import ALGORITHM_MAPPING
from src.models.discrete_problems import solver, traveling_salesman_problem
from src.visualization.visualize import plot_fitness_curves, plot_walltime_chart


def traveling_salesman_problem_analysis():
    # define a traveling salesman problem with 20 cities
    problem = traveling_salesman_problem(number_of_cities=20)
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
    plot_fitness_curves("Traveling Salesman", fitness_score_map)
    plot_walltime_chart("Traveling Salesman", walltime_map)
