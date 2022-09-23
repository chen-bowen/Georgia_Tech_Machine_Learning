# pylint: disable=import-error
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_MAPPING = {
    "Decision Tree": {
        "model": DecisionTreeClassifier,
        "params": {"ccp_alpha": 0.0},
        "default_value": 0.0,
        "actual_params_name": "ccp_alpha",
    },
    "Neural Network": {
        "model": MLPClassifier,
        "params": {"hidden_layer_sizes": (32,), "max_iter": 2000},
        "default_value": (32,),
        "actual_params_name": "hidden_layer_sizes",
    },
    "AdaBoost": {
        "model": AdaBoostClassifier,
        "params": {"n_estimators": 50},
        "default_value": 50,
        "actual_params_name": "n_estimators",
    },
    "SVC": {
        "model": SVC,
        "params": {"kernel": "linear"},
        "default_value": "linear",
        "actual_params_name": "kernel",
    },
    "KNN": {
        "model": KNeighborsClassifier,
        "params": {"n_neighbors": 5},
        "default_value": 5,
        "actual_params_name": "n_neighbors",
    },
}

MODEL_PARAMS_SPACE = {
    "Decision Tree": {"ccp_alpha": 0.08},
    "Neural Network": {"hidden_layer_sizes": (32, 16), "max_iter": 2000},
    "AdaBoost": {"n_estimators": 50},
    "SVC": {"kernel": "rbf"},
    "KNN": {"n_neighbors": 20},
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
