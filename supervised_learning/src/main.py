from sklearn.model_selection import ShuffleSplit

from src.config.config import RANDOM_SEED
from src.data.nba_dataset import NBADataset
from src.models.predict_career_duration import build_career_duration_model
from src.visualization.visualize import plot_learning_curve


def analyze_nba_career_duration_model():
    """
    Plot the learning curve for all 5 different models,
    return the performance (accuracy) of the test set for each model (in dict)
    """
    # get training and test sets
    nba_dataset = NBADataset()
    X, y = nba_dataset.build_training_test_set()

    for model_type in ["decision tree", "neural network", "adaboost", "svc", "knn"]:
        # build model pipeline
        model = build_career_duration_model(model_type)

        # create cross validation object
        cv = ShuffleSplit(n_splits=50, test_size=0.1, random_state=RANDOM_SEED)

        # plot learning curve
        plot_learning_curve(
            model_type, "NBA", "Default", model, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4
        )
