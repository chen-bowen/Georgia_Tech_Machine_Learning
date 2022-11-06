import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config.config import ALL_POSITIONS, ALL_TEAMS

# from sklearn.preprocessing import LabelBinarizer


class TeamsEncoder(BaseEstimator):
    """Encode the 30 teams into one hot columns"""

    def __init__(self, team_variable=None):
        self.team_variable = team_variable
        self.teams_map = {
            "NJN": "BRK",
            "SEA": "OKC",
            "WSB": "WAS",
            "CHH": "CHA",
            "NOH": "NOP",
            "VAN": "MEM",
            "CHO": "CHA",
            "KCK": "SAC",
            "SDC": "LAC",
            "NOK": "NOP",
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """only 30 teams in the league and expected no unknown values, so use pd.get_dummies"""
        X[self.team_variable] = (
            X[self.team_variable]
            .map(self.teams_map)
            .fillna(X[self.team_variable])
            .astype(pd.CategoricalDtype(categories=ALL_TEAMS))
        )
        X = X.drop([self.team_variable], axis=1).join(
            pd.get_dummies(X[self.team_variable])
        )
        return X


class PlayerPositionEncoder(BaseEstimator):
    """Encode the 5 player positions into one hot columns"""

    def __init__(self, position_variable=None):
        self.position_variable = position_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """only 5 positions and expected no unknown values, so use pd.get_dummies"""
        # select the players' first position to reduce the variations
        X[self.position_variable] = (
            X[self.position_variable]
            .str.split("-")
            .apply(lambda x: x[0])
            .astype(pd.CategoricalDtype(categories=ALL_POSITIONS))
        )
        X = X.drop([self.position_variable], axis=1).join(
            pd.get_dummies(X[self.position_variable])
        )
        return X


def preprocess_nba_players_data(X):
    """
    Constructs the nba data preprocessing pipeline
    """
    pipeline = Pipeline(
        [
            ("player_position", PlayerPositionEncoder("pos")),
            ("team", TeamsEncoder("tm")),
            ("scaler", StandardScaler()),
        ]
    )
    pipeline.fit(X)
    return pipeline.transform(X)
