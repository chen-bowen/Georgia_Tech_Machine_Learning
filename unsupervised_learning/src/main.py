from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from src.config.config import (
    CLUSTERING_MODEL_NAMES,
    DIM_REDUCE_MODEL_NAMES,
    RANDOM_SEED,
    THRESHOLD_MAP,
)
from src.data.nba_dataset import NBADataset
from src.data.twitter_dataset import TwitterDataset
from src.features.nba_features import preprocess_nba_players_data
from src.features.twitter_features import preprocess_tweets
from src.models.clustering import (
    expectation_maximization_clustering,
    expectation_maximization_experiment,
    k_means_clustering,
    k_means_clustering_experiment,
)
from src.models.dimensionality_reduction import (
    reduce_by_ica,
    reduce_by_lda,
    reduce_by_pca,
    reduce_by_random_projection,
)
from src.visualization.visualize import (
    expectation_maximization_visuals,
    explained_variance_plot,
    kmeans_visuals,
    kurtosis_plot,
    mean_squared_error_plot,
    plot_learning_curve,
)


def clustering_analysis(data, model_name, dataset_name):
    """
    Performs clustering analysis given the dataset and model name and save the visualizations
    data - the dataframe that contains the data to be clustered, either NBA or Twitter, full
            or reduced
    model_name - the string name of the clustering model
    dataset_name - the string name of the dataset
    """
    # get the metrics tuple
    metrics = CLUSTERING_EXPERIMENT_MAPPING[model_name](data)
    # make plots
    CLUSTERING_VISUAL_FUNC_MAPPING[model_name](**metrics, dataset_name=dataset_name)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.suptitle(
        f"Clustering using {model_name} Model For {dataset_name} Dataset",
        fontsize=20,
    )
    plt.savefig(f"./reports/figures/{model_name}_{dataset_name}.jpg", dpi=150)


def dimensionality_reduction_analysis(data, model_name, dataset_name):
    """
    Performs dimensionality reduction analysis given the dataset and model name
    and save the visualizations
    data - the dataframe that contains the data to be clustered, either NBA or Twitter, full
            or reduced
    model_name - the string name of the clustering model
    dataset_name - the string name of the dataset
    """
    # get the reduced data
    if model_name == "LDA":
        X, y = data.drop("target", axis=1), data["target"]
        reduced_data, _ = DIM_REDUCE_MODEL_MAPPING[model_name](X, y)
    else:
        reduced_data, _ = DIM_REDUCE_MODEL_MAPPING[model_name](data)
    # perform visualizations on the reduced data, get the number of components
    num_components, _ = DIM_REDUCE_MODEL_MAPPING[model_name](
        reduced_data, threshold=THRESHOLD_MAP
    )
    print(
        f"The numnber of components for reducing the {dataset_name} using {model_name} is {num_components}"  # pylint: disable=line-too-long
    )
    # save figures
    plt.suptitle(
        f"Dimensionality reduction using {model_name} Model For {dataset_name} Dataset",
        fontsize=20,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig(f"../reports/figures/{model_name}_{dataset_name}.jpg", dpi=150)

    return reduced_data


def all_analysis(dataset_name):
    """Perform all the analyses given the dataset name"""
    # preprocessing the data
    dataset = DATASET_MAP[dataset_name]
    X = dataset.build_training_test_set()
    X = PREPROCESS_FUNC_MAP[dataset_name](X)

    # perform clustering
    for cluster_model_name in CLUSTERING_MODEL_NAMES:
        clustering_analysis(X, cluster_model_name, dataset_name)
        breakpoint()
    # perform dimensionality reduction and save reduced dataset
    reduced_data_map = defaultdict()
    for dim_reduction_model_name in DIM_REDUCE_MODEL_NAMES:
        reduced_data_map[dim_reduction_model_name] = dimensionality_reduction_analysis(
            X, cluster_model_name, dataset_name
        )

    # perform clustering again on the reduced dataset
    cluster_data_map = defaultdict(defaultdict())
    optimal_num_clusters = {"K-means": 5, "Expectation Maximization": 5}
    for cluster_model_name in CLUSTERING_MODEL_NAMES:
        for dim_reduction_model_name, reduced_data in reduced_data_map.items():
            cluster_data_map[cluster_model_name][
                dim_reduction_model_name
            ] = CLUSTERING_MODEL_MAPPING[cluster_model_name](
                reduced_data, optimal_num_clusters[cluster_model_name]
            )

    # run neural network with dimension reduced features on NBA dataset only
    if dataset_name == "NBA":
        _, axes = plt.subplots(4, 4, figsize=(20, 20))
        for i, dim_reduction_model_name, reduced_data in enumerate(
            reduced_data_map.items()
        ):
            neural_network = MLPClassifier(hidden_layer_sizes=(32,))
            y = reduced_data["target"]
            X = reduced_data.drop("target", axis=1)
            cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=RANDOM_SEED)
            # plot learning curves cluster on the corresponding axes
            plot_learning_curve(
                dim_reduction_model_name,
                neural_network,
                X,
                y,
                ylim=(0.5, 1.01),
                axes=axes[:, i],
                cv=cv,
                n_jobs=4,
            )
        # save figure
        plt.suptitle(
            f"Neural Network with Dimensionality Reductions For {dataset_name} Dataset",
            fontsize=20,
        )
        plt.tight_layout(rect=[0, 0.01, 1, 0.99])
        plt.savefig("../reports/figures/nn_dim_reduction.jpg", dpi=150)

        # run neural network with cluster label added as features
        for cluster_model_name in CLUSTERING_MODEL_NAMES:
            _, axes = plt.subplots(4, 4, figsize=(20, 20))
            for i, dim_reduction_model_name, reduced_data in enumerate(
                reduced_data_map.items()
            ):
                neural_network = MLPClassifier(hidden_layer_sizes=(32,))
                y = reduced_data["target"]
                X = reduced_data.drop("target", axis=1)
                cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=RANDOM_SEED)
                # plot learning curves cluster on the corresponding axes
                plot_learning_curve(
                    dim_reduction_model_name,
                    neural_network,
                    X,
                    y,
                    ylim=(0.5, 1.01),
                    axes=axes[:, i],
                    cv=cv,
                    n_jobs=4,
                )
            # save figure
            plt.suptitle(
                f"Neural Network with {cluster_model_name} and {dim_reduction_model_name} For {dataset_name} Dataset",
                fontsize=20,
            )
            plt.tight_layout(rect=[0, 0.01, 1, 0.99])
            plt.savefig(
                "../reports/figures/nn_{cluster_model_name}_dim_reduction.jpg", dpi=150
            )


if __name__ == "__main__":
    # create maps
    CLUSTERING_EXPERIMENT_MAPPING = {
        "K-means": k_means_clustering_experiment,
        "Expectation Maximization": expectation_maximization_experiment,
    }
    CLUSTERING_MODEL_MAPPING = {
        "K-means": k_means_clustering,
        "Expectation Maximization": expectation_maximization_clustering,
    }
    DIM_REDUCE_MODEL_MAPPING = {
        "PCA": reduce_by_pca,
        "ICA": reduce_by_ica,
        "LDA": reduce_by_lda,
        "Random Projection": reduce_by_random_projection,
    }
    CLUSTERING_VISUAL_FUNC_MAPPING = {
        "K-means": kmeans_visuals,
        "Expectation Maximization": expectation_maximization_visuals,
    }
    DIM_REDUCE_VISUAL_FUNC_MAPPING = {
        "PCA": explained_variance_plot,
        "ICA": kurtosis_plot,
        "LDA": explained_variance_plot,
        "Random Projection": mean_squared_error_plot,
    }
    DATASET_MAP = {"NBA": NBADataset(), "Twitter": TwitterDataset()}
    PREPROCESS_FUNC_MAP = {
        "NBA": preprocess_nba_players_data,
        "Twitter": preprocess_tweets,
    }
    all_analysis("Twitter")
    # all_analysis("NBA")
