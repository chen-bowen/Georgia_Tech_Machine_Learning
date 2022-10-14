import mlrose_hiive as mlrose
import numpy as np
from src.config.config import ALGORITHM_MAPPING, RANDOM_SEED


def traveling_salesman_problem(number_of_cities=20):
    """
    Generate a traveling salesmane problem given the number of cities
    With the traveling salesman problem generator, the coordinates of the cities are
        [(175, 68),
        (196, 176),
        (25, 127),
        (246, 135),
        (67, 172),
        (211, 0),
        (151, 75),
        (103, 55),
        (92, 6),
        (185, 19),
        (142, 188),
        (23, 44),
        (72, 191),
        (89, 69),
        (110, 56),
        (42, 152),
        (218, 183),
        (136, 181),
        (167, 112),
        (230, 189)]
    """
    # define problem and objects
    problem_obj = mlrose.TSPGenerator().generate(RANDOM_SEED, number_of_cities)
    return problem_obj


def multi_queens_problem(number_of_queens=10):
    """
    generate an n queens problem given the number of queens
    With the n-queens problem generator, the row positions of each queen initially are
    [9, 6, 7, 3, 4, 3, 7, 9, 7, 8]
    """
    # define problem and objects
    problem_obj = mlrose.QueensGenerator().generate(RANDOM_SEED, number_of_queens)

    return problem_obj


def knapsack_problem(max_item_count=5):
    """
    Generate a knapsack problem given the parameters
    """
    number_of_items_types = 10
    max_weight_per_item = 25
    max_value_per_item = 10
    max_weight_pct = 0.7
    # get the knapsack problem
    weights = 1 + np.random.randint(max_weight_per_item, size=number_of_items_types)
    values = 1 + np.random.randint(max_value_per_item, size=number_of_items_types)
    problem_obj = mlrose.KnapsackOpt(
        length=number_of_items_types,
        maximize=True,
        max_val=max_item_count,
        weights=weights,
        values=values,
        max_weight_pct=max_weight_pct,
    )
    return problem_obj


def solver(problem, algorithm, params_set):
    """
    Solve the optimization problem using the given algorithm
    and return the problem name, best state, best fitnees and fitness curve vs iterations

    problem: (problem name, problem object)
    algorithm: (str)
    """
    # solve the problem using the given algorithm
    problem.reset()
    best_state, best_fitnes, fitness_curve = ALGORITHM_MAPPING[algorithm](
        problem,
        random_state=RANDOM_SEED,
        curve=True,
        max_iters=700,
        max_attempts=700,
        **params_set,
    )

    return best_state, best_fitnes, fitness_curve
