import mlrose_hiive as mlrose
from src.models.discrete_problems import (  # pylint: disable=import-error
    knapsack_problem,
    multi_queens_problem,
    traveling_salesman_problem,
)

ALGORITHM_MAPPING = {
    "Random Hill Climb": mlrose.random_hill_climb,
    "Simulated Annealing": mlrose.simulated_annealing,
    "Genetic Algorithm": mlrose.genetic_alg,
    "Mimic": mlrose.mimic,
}
ALGORITHM_HYPERPARAMS_MAPPING = {  # type: ignore
    "Random Hill Climb": dict(restarts=3),
    "Simulated Annealing": dict(schedule=mlrose.ExpDecay(exp_const=0.005)),
    "Genetic Algorithm": dict(pop_size=500, mutation_prob=0.2),
    "Mimic": dict(pop_size=500, keep_pct=0.2),
}
PROBLEM_NAME_MAPPING = {
    "Traveling Salesman Problem": traveling_salesman_problem,
    "Knapsack Problem": knapsack_problem,
    "N-Queens Problem": multi_queens_problem,
}
PROBLEM_PARAMS_MAPPING = {
    "Traveling Salesman Problem": dict(number_of_cities=20),
    "Knapsack Problem": dict(
        number_of_items_types=10,
        max_item_count=5,
        max_weight_per_item=25,
        max_value_per_item=10,
        max_weight_pct=0.7,
    ),
    "N-Queens Problem": dict(number_of_queens=10),
}
RANDOM_SEED = 7
ALL_POSITIONS = ["PG", "SG", "SF", "PF", "C"]
ALL_TEAMS = [
    "WAS",
    "ATL",
    "DET",
    "MIN",
    "MIA",
    "CLE",
    "MEM",
    "GSW",
    "TOR",
    "SAC",
    "PHI",
    "DEN",
    "NYK",
    "BOS",
    "TOT",
    "ORL",
    "HOU",
    "UTA",
    "DAL",
    "MIL",
    "CHI",
    "LAC",
    "LAL",
    "PHO",
    "OKC",
    "IND",
    "SAS",
    "CHA",
    "POR",
    "NOP",
    "BRK",
]
