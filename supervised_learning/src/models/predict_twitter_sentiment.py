import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from src.config.config import MODEL_MAPPING
from src.features.twitter_features import TweetPreprocessor

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")


def build_tweet_sentiment_model(model_type, params):
    """
    Constructs the tweet sentiment model pipeline that switches between different models
    """
    pipeline = Pipeline(
        [
            ("tweet_preprocessor", TweetPreprocessor("tweet_content")),
            ("vec", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("classifier", MODEL_MAPPING[model_type]["model"](**params),),
        ]
    )
    return pipeline
