from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from src.config.config import MODEL_MAPPING
from src.features.twitter_features import TweetPreprocessor


def build_tweet_sentiment_model(model_type):
    """
    Constructs the tweet sentiment model pipeline that switch between different models
    """
    pipeline = Pipeline(
        [
            ("tweet_preprocessor", TweetPreprocessor("tweet_content")),
            ("vec", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            (
                "classifier",
                MODEL_MAPPING[model_type]["model"](
                    **MODEL_MAPPING[model_type]["params"]
                ),
            ),
        ]
    )
    return pipeline
