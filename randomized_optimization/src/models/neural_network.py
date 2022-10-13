import warnings

import mlrose_hiive as mlrose
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.features.nba_features import PlayerPositionEncoder, TeamsEncoder

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")


def build_career_duration_model(params):
    """
    Constructs the career duration model pipeline that switches between different models
    """
    # build pipeline
    pipeline = Pipeline(
        [
            ("player_position", PlayerPositionEncoder("pos")),
            ("team", TeamsEncoder("tm")),
            ("scaler", StandardScaler()),
            ("classifier", mlrose.NeuralNetwork(**params)),
        ]
    )
    return pipeline
