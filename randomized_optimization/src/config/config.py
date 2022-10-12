import mlrose_hiive as mlrose

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
RANDOM_SEED = 7
