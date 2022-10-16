import mlrose_hiive as mlrose
import numpy as np

ALGORITHM_MAPPING = {
    "Random Hill Climb": mlrose.random_hill_climb,
    "Simulated Annealing": mlrose.simulated_annealing,
    "Genetic Algorithm": mlrose.genetic_alg,
    "Mimic": mlrose.mimic,
}
ALGORITHM_HYPERPARAMS_MAPPING = {  # type: ignore
    "Random Hill Climb": [dict(restarts=i) for i in np.arange(1, 11, 3)],
    "Simulated Annealing": [
        dict(schedule=mlrose.ExpDecay(exp_const=i))
        for i in [0.0005, 0.001, 0.005, 0.01]
    ],
    "Genetic Algorithm": [dict(mutation_prob=i) for i in [0.05, 0.2, 0.5, 0.8]],
    "Mimic": [dict(keep_pct=i) for i in [0.05, 0.2, 0.5, 0.8]],
}

PROBLEM_PARAMS_MAPPING = {
    "Traveling Salesman Problem": dict(number_of_cities=20),
    "Knapsack Problem": dict(max_item_count=10),
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
NN_OPT_ALGORITHMS = [
    "random_hill_climb",
    "simulated_annealing",
    "genetic_alg",
    "gradient_descent",
]
NN_LEARNING_RATE_MAP = {
    "random_hill_climb": 0.1,
    "simulated_annealing": 0.1,
    "genetic_alg": 0.001,
    "gradient_descent": 0.0001,
}
NN_DEFAULT_PARAMS = dict(
    hidden_nodes=[32],
    activation="relu",
    max_iters=1000,
    bias=True,
    is_classifier=True,
    clip_max=5,
    max_attempts=1000,
)
