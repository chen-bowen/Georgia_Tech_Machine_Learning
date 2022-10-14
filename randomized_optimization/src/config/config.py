import mlrose_hiive as mlrose

ALGORITHM_MAPPING = {
    "Random Hill Climb": mlrose.random_hill_climb,
    "Simulated Annealing": mlrose.simulated_annealing,
    "Genetic Algorithm": mlrose.genetic_alg,
    "Mimic": mlrose.mimic,
}
ALGORITHM_HYPERPARAMS_DEFAULT_MAPPING = {  # type: ignore
    "Random Hill Climb": dict(restarts=3, max_attempts=500, max_iters=1000),
    "Simulated Annealing": dict(
        schedule=mlrose.ExpDecay(exp_const=0.005), max_attempts=500, max_iters=1000
    ),
    "Genetic Algorithm": dict(
        pop_size=500, mutation_prob=0.2, max_attempts=500, max_iters=1000
    ),
    "Mimic": dict(pop_size=500, keep_pct=0.2, max_attempts=500, max_iters=1000),
}
ALGORITHM_HYPERPARAMS_TUNED_MAPPING = {  # type: ignore
    "Random Hill Climb": dict(restarts=10, max_attempts=500, max_iters=1000),
    "Simulated Annealing": dict(
        schedule=mlrose.ExpDecay(exp_const=0.05), max_attempts=500, max_iters=1000
    ),
    "Genetic Algorithm": dict(
        pop_size=500, mutation_prob=0.5, max_attempts=500, max_iters=1000
    ),
    "Mimic": dict(pop_size=500, keep_pct=0.5, max_attempts=500, max_iters=1000),
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
NN_OPT_ALGORITHMS = [
    "random_hill_climb",
    "simulated_annealing",
    "genetic_alg",
    "gradient_descent",
]
NN_DEFAULT_PARAMS = dict(
    hidden_nodes=[32],
    activation="relu",
    max_iters=1000,
    bias=True,
    is_classifier=True,
    learning_rate=0.0001,
    early_stopping=True,
    clip_max=5,
    max_attempts=100,
)
NN_TUNED_PARAMS = dict(
    hidden_nodes=[32, 16],
    activation="relu",
    max_iters=1000,
    bias=True,
    is_classifier=True,
    learning_rate=0.0001,
    early_stopping=True,
    clip_max=5,
    max_attempts=100,
)  # type: ignore
