import time
from warnings import simplefilter

import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from src.config.config import MODEL_MAPPING, MODEL_PARAMS_SPACE, RANDOM_SEED
from src.data.nba_dataset import NBADataset
from src.models.predict_career_duration import build_career_duration_model
from src.visualization.visualize import plot_learning_curve


def analyze_nba_career_duration_model():
    """
    Plot the learning curve for all 5 different models,
    return the performance (accuracy) of the test set for each model (in dict)
    """
    simplefilter("ignore", category=ConvergenceWarning)
    # get training and test sets
    nba_dataset = NBADataset()
    X, y = nba_dataset.build_training_test_set()

    for model_type in ["Decision Tree", "Neural Network", "AdaBoost", "SVC", "KNN"]:
        # start timer
        start = time.perf_counter()
        # build model pipeline
        model_default = build_career_duration_model(
            model_type, MODEL_MAPPING[model_type]["params"]
        )

        # create cross validation object
        cv = ShuffleSplit(n_splits=50, test_size=0.1, random_state=RANDOM_SEED)
        # perform random search for the optimal parameters
        params_search = RandomizedSearchCV(
            model_default,
            MODEL_PARAMS_SPACE[model_type],
            n_iter=100,
            cv=5,
            random_state=RANDOM_SEED,
        )
        tuned_model = params_search.fit(X, y)
        best_params_res = tuned_model.best_estimator_.get_params()  # type: ignore


        # creat model with the best parameters
        best_params = {
            MODEL_MAPPING[model_type]["actual_params_name"]: best_params_res[
                MODEL_MAPPING[model_type]["tuned_params_name"]
            ]
        }
        # add iteration to 500 if it's neural network
        if model_type == "Neural Network":
            best_params["max_iter"] = 500

        model_best = build_career_duration_model(model_type, best_params)

        # initialize figure
        _, axes = plt.subplots(4, 2, figsize=(10, 20))

        # plot learning curve with default parameters
        plot_learning_curve(
            "Default",
            model_default,
            X,
            y,
            axes=axes[:, 0],  # type: ignore
            ylim=(0.7, 1.01),
            cv=cv,
            n_jobs=4,
        )

        # plot learning curve with the best parameters
        plot_learning_curve(
            "Best",
            model_best,
            X,
            y,
            axes=axes[:, 1],  # type: ignore
            ylim=(0.7, 1.01),
            cv=cv,
            n_jobs=4,
        )

        # save figure
        plt.suptitle(f"{model_type} Model with Default and Best Parameters")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"../reports/figures/{model_type}_NBA.jpg", dpi=150)

        # end timer
        end = time.perf_counter()
        print(f"Analyzing {model_type} completed in {end - start:0.4f} seconds")
