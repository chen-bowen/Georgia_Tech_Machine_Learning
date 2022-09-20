from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_MAPPING = {
    "decision tree": {"model": DecisionTreeClassifier, "params": {"ccp_alpha": 0.0}},
    "neural network": {
        "model": MLPClassifier,
        "params": {"hidden_layer_sizes": (50, 10), "max_iter": 500},
    },
    "adaboost": {"model": AdaBoostClassifier, "params": {"n_estimators": 50}},
    "svc": {"model": SVC, "params": {"kernel": "rbf"}},
    "knn": {"model": KNeighborsClassifier, "params": {"n_neighbors": 5}},
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
