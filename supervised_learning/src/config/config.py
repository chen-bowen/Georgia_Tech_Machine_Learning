from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_MAPPING = {
    "decision_tree": {"model": DecisionTreeClassifier, "params": {}},
    "neural_network": {"model": MLPClassifier, "params": {}},
    "adaboost": {"model": AdaBoostClassifier, "params": {}},
    "svc": {"model": SVC, "params": {}},
    "knn": {"model": KNeighborsClassifier, "params": {}},
}
