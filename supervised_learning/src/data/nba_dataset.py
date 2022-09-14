from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class NBADataset:
    """Class that generates the NBA player career duration dataset"""

    def __init__(self):
        self.player_careers = pd.read_csv(
            str(Path(__file__).parent.parent.parent)
            + "/data/NBA_all_seasons/Player Career Info.csv"
        )
        self.player_stats = pd.read_csv(
            str(Path(__file__).parent.parent.parent)
            + "/data/NBA_all_seasons/Player Per Game.csv"
        )
        self.player_advanced = pd.read_csv(
            str(Path(__file__).parent.parent.parent)
            + "/data/NBA_all_seasons/Advanced.csv"
        )
        self.process_data()
        self.build_training_test_set()

    def process_data(self):
        """
        Process and join data into the one dataframe
        """
        # replace NA with nan
        self.player_stats = self.player_stats.replace("NA", None)

        # filter out players drafted after 2017
        player_careers_draft_before_2017 = self.player_careers[
            self.player_careers["first_seas"] <= 2017
        ]
        # create target variable by defining players career longer than 5 years
        player_careers_draft_before_2017[
            "long_career"
        ] = player_careers_draft_before_2017.apply(
            lambda row: 1 if row["num_seasons"] >= 5 else 0, axis=1
        )
        # merge and drop missing data (untracked statisitcs for earlier years)
        self.players_data_full = pd.merge(
            player_careers_draft_before_2017.drop(["player", "birth_year"], axis=1),
            self.player_stats.drop(["birth_year"], axis=1),
            how="left",
            on="player_id",
        ).dropna()

        # get all players third year or earlier stats, one player per row
        players_rookie_contracts = self.players_data_full[
            self.players_data_full["experience"] <= 3
        ]
        self.players_data = (
            players_rookie_contracts.sort_values("experience", ascending=False)
            .groupby("player_id")
            .head(1)
        )

        # join advanced stats
        self.players_data = pd.merge(
            self.players_data,
            self.player_advanced.drop(
                [
                    "player",
                    "birth_year",
                    "season",
                    "pos",
                    "age",
                    "experience",
                    "lg",
                    "tm",
                    "g",
                    "mp",
                ],
                axis=1,
            ),
            on=["player_id", "seas_id"],
        )

    def build_training_test_set(self):
        "Split the players data attrtibute into train_X, test_X, train_Y and test_Y"
        y = self.players_data["long_career"]
        X = self.players_data.drop(
            [
                "player_id",
                "hof",
                "num_seasons",
                "first_seas",
                "last_seas",
                "long_career",
                "seas_id",
                "season",
                "player",
                "long_career",
            ],
            axis=1,
        )

        # split to train and validation, test set and save them to attrtibutes
        (
            X_train_val,
            self.X_test,
            y_train_val,
            self.y_test,
        ) = train_test_split(X, y, test_size=0.10, random_state=7)

        (self.X_train, self.X_val, self.y_train, self.y_val) = train_test_split(
            X_train_val, y_train_val, test_size=0.10, random_state=7
        )
