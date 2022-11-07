import numpy as np

RANDOM_SEED = 100
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

NUM_CLUSTERS_LIST = list(np.arange(3, 15, 1)) + [20, 25, 30, 35, 40, 45, 50]
THRESHOLD_MAP = {
    "PCA": 0.85,
    "ICA": 5,
    "SVD": 0.85,
    "Random Projection": 0.3,
}
CLUSTERING_MODEL_NAMES = ["K-means", "Expectation Maximization"]
DIM_REDUCE_MODEL_NAMES = ["PCA", "ICA", "SVD", "Random Projection"]
