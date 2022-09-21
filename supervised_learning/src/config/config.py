# pylint: disable=import-error
from scipy.stats import randint, uniform
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
        "tuned_params_name": "classifier__ccp_alpha",
    },
    "Neural Network": {
        "model": MLPClassifier,
        "params": {"hidden_layer_sizes": (50,), "max_iter": 500},
        "default_value": (50,),
        "actual_params_name": "hidden_layer_sizes",
        "tuned_params_name": "classifier__hidden_layer_sizes",
    },
    "AdaBoost": {
        "model": AdaBoostClassifier,
        "params": {"n_estimators": 50},
        "default_value": 50,
        "actual_params_name": "n_estimators",
        "tuned_params_name": "classifier__n_estimators",
    },
    "SVC": {
        "model": SVC,
        "params": {"kernel": "rbf"},
        "default_value":  "rbf",
        "actual_params_name": "kernel",
        "tuned_params_name": "classifier__kernel",
    },
    "KNN": {
        "model": KNeighborsClassifier,
        "params": {"n_neighbors": 5},
         "default_value":  5,
        "actual_params_name": "n_neighbors",
        "tuned_params_name": "classifier__n_neighbors",
    },
}

MODEL_PARAMS_SPACE = {
    "Decision Tree": {"classifier__ccp_alpha": uniform(0, 1)},
    "Neural Network": {
        "classifier__hidden_layer_sizes": [(20, 5), (50, 10), (50,)],
    },
    "AdaBoost": {"classifier__n_estimators": randint(5, 100)},
    "SVC": {"classifier__kernel": ["linear", "poly", "rbf", "sigmoid"]},
    "KNN": {"classifier__n_neighbors": randint(2, 20)},
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
