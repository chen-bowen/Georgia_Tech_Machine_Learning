import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config.config import MODEL_MAPPING
from src.features.nba_features import PlayerPositionEncoder, TeamsEncoder

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")


def build_career_duration_model(model_type, params):
    """
    Constructs the career duration model pipeline that switches between different models
    """
    pipeline = Pipeline(
        [
            ("player_position", PlayerPositionEncoder("pos")),
            ("team", TeamsEncoder("tm")),
            ("scaler", StandardScaler()),
            ("classifier", MODEL_MAPPING[model_type]["model"](**params)),
        ]
    )
    return pipeline
