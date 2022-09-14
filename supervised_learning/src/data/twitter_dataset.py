from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class TwitterDataset:
    """Class that generates the Twitter sentiment dataset"""

    def __init__(self):
        self.test_data = pd.read_csv(
            str(Path(__file__).parent.parent.parent)
            + "/data/Twitter_sentiment/twitter_validation.csv",
            header=None,
        )
        self.train_data = pd.read_csv(
            str(Path(__file__).parent.parent.parent)
            + "/data/Twitter_sentiment/twitter_training.csv",
            header=None,
        )
        self.label_encode_mapping = {
            "Negative": 0,
            "Positive": 1,
            "Neutral": 2,
            "Irrelevant": 3,
        }
        self.build_training_test_set()

    def build_training_test_set(self):
        "Split the players data attrtibute into train_X, test_X, train_Y and test_Y"
        # rename columns for both train and test data
        self.train_data.columns = ["tweet_id", "entity", "sentiment", "tweet_content"]
        self.test_data.columns = ["tweet_id", "entity", "sentiment", "tweet_content"]

        # select only postive and negative tweets
        self.train_data = self.train_data[
            self.train_data["sentiment"].isin(["Positive", "Negative"])
        ]
        self.test_data = self.test_data[
            self.test_data["sentiment"].isin(["Positive", "Negative"])
        ]

        # remove NA for both train and test
        self.train_data = self.train_data.dropna(subset=["tweet_content"])
        self.test_data = self.test_data.dropna(subset=["tweet_content"])

        # get the training, validation and test data and save to X, y attributes
        self.y_test = self.test_data["sentiment"].map(self.label_encode_mapping)
        self.X_test = self.test_data.drop(["sentiment"], axis=1)
        y_train_val = self.train_data["sentiment"].map(self.label_encode_mapping)
        X_train_val = self.train_data.drop(["sentiment"], axis=1)

        # split to train and validation, test set and save them to attrtibutes
        (
            self.X_train,
            self.X_val,
            self.y_val,
            self.y_test,
        ) = train_test_split(X_train_val, y_train_val, test_size=0.10, random_state=7)
