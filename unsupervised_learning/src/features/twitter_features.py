import re
import warnings

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


class TweetPreprocessor(BaseEstimator):
    """
    Perform the following steps for preprocessing the tweets
    1) Remove Additional Letter such as @
    2) Remove Stop Words
    3) Stemming
    4) TF-IDF
    """

    def __init__(self, tweets_col):
        self.tweets_col = tweets_col
        self.set_nltk_resource()

    def set_nltk_resource(self):
        # download nltk resources
        self.english_stop_words = stopwords.words("english")

    def remove_additional_chars(self, df):
        """Remove additional characters such as @ from tweets"""
        REPLACE_WITH_SPACE = re.compile("(@)")
        SPACE = " "
        df[self.tweets_col] = df[self.tweets_col].apply(
            lambda row: REPLACE_WITH_SPACE.sub(SPACE, row.lower())
        )
        return df

    def remove_stop_words(self, df):
        """Remove stop words from tweets"""
        df[self.tweets_col] = df[self.tweets_col].apply(
            lambda row: " ".join(
                [word for word in row.split() if word not in self.english_stop_words]
            )
        )
        return df

    def stem_tweets(self, df):
        """Stemming on Tweets"""
        stemmer = PorterStemmer()
        df[self.tweets_col] = df[self.tweets_col].apply(
            lambda row: " ".join([stemmer.stem(word) for word in row.split()])
        )
        return df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Apply all the transformations on the tweets"""
        X = self.remove_additional_chars(X)
        X = self.remove_stop_words(X)
        X = self.stem_tweets(X)
        return X[self.tweets_col].values.tolist()


warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")


def preprocess_tweets(X):
    """
    Constructs the tweet data preprocessing pipeline
    """
    pipeline = Pipeline(
        [
            ("tweet_preprocessor", TweetPreprocessor("tweet_content")),
            ("vec", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
        ]
    )
    pipeline.fit(X)
    return pipeline.transform(X)
