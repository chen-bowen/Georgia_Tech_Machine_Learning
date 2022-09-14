from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config.config import MODEL_MAPPING
from src.features.nba_features import PlayerPositionEncoder, TeamsEncoder


def build_tweet_sentiment_model(model_type):
    """
    Constructs the tweet sentiment model pipeline that switch between different models
    """
    pipeline = Pipeline(
        [
            ("player_position", PlayerPositionEncoder("pos")),
            ("team", TeamsEncoder("tm")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                MODEL_MAPPING[model_type]["model"](
                    **MODEL_MAPPING[model_type]["params"]
                ),
            ),
        ]
    )
    return pipeline
