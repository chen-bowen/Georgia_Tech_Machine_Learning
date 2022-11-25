from pathlib import Path

import pandas as pd


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
        self.train_data.columns = ["tweet_id", "entity", "sentiment", "tweet_content"]  # type: ignore # pylint: disable=line-too-long
        self.test_data.columns = ["tweet_id", "entity", "sentiment", "tweet_content"]  # type: ignore # pylint: disable=line-too-long

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

        # concat to get all data
        data = pd.concat([self.train_data, self.test_data])

        # sample to get a smaller dataset
        data = data.groupby("tweet_id").head(1)
        data = data[
            data["entity"].isin(
                [
                    "FIFA",
                    "Amazon",
                    "NBA2K",
                    "MaddenNFL",
                    "Google",
                    "Xbox(Xseries)",
                    "Facebook",
                    "Overwatch",
                ]
            )
        ]
        # get the training, validation and test data and save to X, y attributes
        y = data["sentiment"].map(self.label_encode_mapping)
        X = data.drop(["sentiment"], axis=1)

        return X, y
