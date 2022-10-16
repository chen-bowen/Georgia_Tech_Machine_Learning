import mlrose_hiive as mlrose
from src.config.config import ALGORITHM_MAPPING, RANDOM_SEED


def traveling_salesman_problem(number_of_cities=20):
    """
    Generate a traveling salesmane problem given the number of cities
    """
    # define problem and objects
    problem_obj = mlrose.TSPGenerator().generate(RANDOM_SEED, number_of_cities)
    return problem_obj


def n_queens_problem(number_of_queens=10):
    """
    generate an n queens problem given the number of queens
    """
    # define problem and objects
    problem_obj = mlrose.QueensGenerator().generate(RANDOM_SEED, number_of_queens)

    return problem_obj


def knapsack_problem(max_item_count=10):
    """
    Generate a knapsack problem given the parameters
    """
    problem_obj = mlrose.KnapsackGenerator().generate(RANDOM_SEED, max_item_count)
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
