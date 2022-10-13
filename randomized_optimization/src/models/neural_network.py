import mlrose_hiive as mlrose
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config.config import NN_DEFAULT_PARAMS, RANDOM_SEED
from src.features.nba_features import PlayerPositionEncoder, TeamsEncoder


def feature_transformer():
    """build feature transformation pipeline"""
    return Pipeline(
        [
            ("player_position", PlayerPositionEncoder("pos")),
            ("team", TeamsEncoder("tm")),
            ("scaler", StandardScaler()),
        ]
    )


def neural_network(X_train, y_train, algorithm):
    """
    Constructs the career duration model pipeline that switches between different models
    """

    # fit the network
    nn_model1 = mlrose.NeuralNetwork(
        algorithm=algorithm, random_state=RANDOM_SEED, curve=True, **NN_DEFAULT_PARAMS
    )
    nn_model1.fit(X_train, y_train)

    return nn_model1
