import time

import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from src.config.config import MODEL_MAPPING, MODEL_PARAMS_SPACE, RANDOM_SEED
from src.data.nba_dataset import NBADataset
from src.data.twitter_dataset import TwitterDataset
from src.models.predict_career_duration import build_career_duration_model
from src.models.predict_twitter_sentiment import build_tweet_sentiment_model
from src.visualization.visualize import plot_learning_curve


def generate_model_analysis_plot(X, y, dataset_name):
    """
    Plot the learning curve for all 5 different models,
    return the performance (accuracy) of the test set for each model (in dict)
    Generate analysis plots given the model name and dataset name.
    """

    ANALYSIS_DATASET_MAP = {
        "NBA": build_career_duration_model,
        "Twitter": build_tweet_sentiment_model,
    }

    for model_type in ["Decision Tree", "Neural Network", "AdaBoost", "SVC", "KNN"]:
        # start timer
        start = time.perf_counter()
        # build model pipeline
        model_default = ANALYSIS_DATASET_MAP[dataset_name](
            model_type, MODEL_MAPPING[model_type]["params"]
        )
        model_variant = ANALYSIS_DATASET_MAP[dataset_name](
            model_type, MODEL_PARAMS_SPACE[model_type]
        )

        # create cross validation object
        cv = ShuffleSplit(n_splits=15, test_size=0.1, random_state=RANDOM_SEED)

        # initialize figure
        _, axes = plt.subplots(4, 2, figsize=(10, 20))

        # plot learning curve with default parameters
        plot_learning_curve(
            model_type,
            MODEL_MAPPING[model_type]["default_value"],
            model_default,
            X,
            y,
            axes=axes[:, 0],  # type: ignore
            ylim=(0.5, 1.01),
            cv=cv,
            n_jobs=4,
        )

        # plot learning curve with the best parameters
        plot_learning_curve(
            model_type,
            MODEL_PARAMS_SPACE[model_type][
                MODEL_MAPPING[model_type]["actual_params_name"]
            ],
            model_variant,
            X,
            y,
            axes=axes[:, 1],  # type: ignore
            ylim=(0.5, 1.01),
            cv=cv,
            n_jobs=4,
        )

        # save figure
        plt.suptitle(
            f"{model_type} Model with Default and Best Parameters", fontsize=20
        )
        plt.tight_layout(rect=[0, 0.01, 1, 0.99])
        plt.savefig(f"../reports/figures/{model_type}_{dataset_name}.jpg", dpi=150)

        # end timer
        end = time.perf_counter()
        print(f"Analyzing {model_type} completed in {end - start:0.4f} seconds")


def analyze_nba_career_duration_model():
    """
    Analyze the 5 different models using the NBA dataset
    """
    # get training and test sets
    nba_dataset = NBADataset()
    X, y = nba_dataset.build_training_test_set()

    # analyze model
    generate_model_analysis_plot(X, y, "NBA")


def analyze_twitter_sentiment_model():
    """
    Analyze the 5 different models using the Twitter Sentiment dataset
    """
    # get training and test sets
    twitter_dataset = TwitterDataset()
    X, y = twitter_dataset.build_training_test_set()

    # analyze model
    generate_model_analysis_plot(X, y, "Twitter")


if __name__ == "__main__":
    analyze_nba_career_duration_model()
    analyze_twitter_sentiment_model()
