from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
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
    reduce_by_pca,
    reduce_by_random_projection,
    reduce_by_svd,
)
from src.visualization.visualize import (
    dim_reduced_metrics_plot,
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
    reduced_data, metrics = DIM_REDUCE_MODEL_MAPPING[model_name](data)

    # perform visualizations on the reduced data, get the number of components
    fitted_components, _ = DIM_REDUCE_VISUAL_FUNC_MAPPING[model_name](
        metrics, model_name, threshold=THRESHOLD_MAP[model_name]
    )
    print(
        f"The numnber of components for reducing the {dataset_name} dataset using {model_name} is {len(fitted_components)}"  # pylint: disable=line-too-long
    )
    # save figures
    plt.suptitle(
        f"Dimensionality Reduction using {model_name} Model For {dataset_name} Dataset",
        fontsize=20,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig(f"./reports/figures/{model_name}_{dataset_name}.jpg", dpi=150)
    if model_name == "Random Projection":
        return reduced_data[len(fitted_components) - 1]
    return reduced_data[fitted_components]


def all_analysis(dataset_name):
    """Perform all the analyses given the dataset name"""
    # preprocessing the data
    dataset = DATASET_MAP[dataset_name]
    data, label = dataset.build_training_test_set()
    data = PREPROCESS_FUNC_MAP[dataset_name](data)

    # perform clustering
    for cluster_model_name in CLUSTERING_MODEL_NAMES:
        print(f"Analyzing {cluster_model_name} Model for {dataset_name} Dataset")
        clustering_analysis(data, cluster_model_name, dataset_name)

    # perform dimensionality reduction and save reduced dataset
    reduced_data_map = defaultdict()
    for dim_reduction_model_name in DIM_REDUCE_MODEL_NAMES:
        print(f"Analyzing {dim_reduction_model_name} Model for {dataset_name} Dataset")
        reduced_data_map[dim_reduction_model_name] = dimensionality_reduction_analysis(
            data, dim_reduction_model_name, dataset_name
        )

    # perform clustering again on the reduced dataset
    cluster_data_map = defaultdict(lambda: defaultdict(lambda: dict))
    optimal_num_clusters = (
        {"K-means": 40, "Expectation Maximization": 45}
        if dataset_name == "NBA"
        else {"K-means": 35, "Expectation Maximization": 40}
    )

    for cluster_model_name in CLUSTERING_MODEL_NAMES:
        metrics_list = []
        for dim_reduction_model_name, reduced_data in reduced_data_map.items():
            try:
                print(
                    f"Clustering {dataset_name} Dataset with {dim_reduction_model_name} Using {cluster_model_name}"
                )
                clustered_data, metrics = CLUSTERING_MODEL_MAPPING[cluster_model_name](
                    reduced_data, optimal_num_clusters[cluster_model_name]
                )
                cluster_data_map[cluster_model_name][
                    dim_reduction_model_name
                ] = clustered_data
                print(
                    f"""
                    Metrics for Clustering {dataset_name} Dataset with {dim_reduction_model_name} Using {cluster_model_name}:
                    {metrics}
                    """
                )
                metrics_list.append({"model_name": dim_reduction_model_name, **metrics})
                dim_reduced_metrics_plot(
                    pd.DataFrame(metrics_list),
                    cluster_model_name,
                    dim_reduction_model_name,
                    dataset_name,
                )
                plt.tight_layout(rect=[0, 0.01, 1, 0.99])
                plt.savefig(
                    f"./reports/figures/{cluster_model_name}_{dataset_name}_metrics.jpg",
                    dpi=150,
                )
            except Exception:  # pylint: disable=broad-except
                pass

    # run neural network with dimension reduced features on NBA dataset only
    if dataset_name == "NBA":
        _, axes = plt.subplots(4, 4, figsize=(20, 20))
        for i, (dim_reduction_model_name, reduced_data) in enumerate(
            reduced_data_map.items()
        ):
            neural_network = MLPClassifier(hidden_layer_sizes=(32,), max_iter=5000)
            cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=RANDOM_SEED)
            reduced_data.columns = list(map(str, reduced_data.columns))
            # plot learning curves cluster on the corresponding axes
            plot_learning_curve(
                dim_reduction_model_name,
                neural_network,
                reduced_data,
                label,
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
        plt.savefig("./reports/figures/nn_dim_reduction.jpg", dpi=150)

        # run neural network with cluster label added as features
        for cluster_model_name in CLUSTERING_MODEL_NAMES:
            _, axes = plt.subplots(4, 4, figsize=(20, 20))
            for i, (dim_reduction_model_name, reduced_data) in enumerate(
                reduced_data_map.items()
            ):
                neural_network = MLPClassifier(hidden_layer_sizes=(32,), max_iter=5000)
                cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=RANDOM_SEED)
                # plot learning curves cluster on the corresponding axes
                plot_learning_curve(
                    dim_reduction_model_name,
                    neural_network,
                    reduced_data,
                    label,
                    ylim=(0.5, 1.01),
                    axes=axes[:, i],
                    cv=cv,
                    n_jobs=4,
                )
            # save figure
            plt.suptitle(
                f"Neural Network with {cluster_model_name} and Dimensionaility Reduction For {dataset_name} Dataset",  # pylint: disable=line-too-long
                fontsize=20,
            )
            plt.tight_layout(rect=[0, 0.01, 1, 0.99])
            plt.savefig(
                f"./reports/figures/nn_{cluster_model_name}_dim_reduction.jpg", dpi=150
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
        "SVD": reduce_by_svd,
        "Random Projection": reduce_by_random_projection,
    }
    CLUSTERING_VISUAL_FUNC_MAPPING = {
        "K-means": kmeans_visuals,
        "Expectation Maximization": expectation_maximization_visuals,
    }
    DIM_REDUCE_VISUAL_FUNC_MAPPING = {
        "PCA": explained_variance_plot,
        "ICA": kurtosis_plot,
        "SVD": explained_variance_plot,
        "Random Projection": mean_squared_error_plot,
    }
    DATASET_MAP = {"NBA": NBADataset(), "Twitter": TwitterDataset()}
    PREPROCESS_FUNC_MAP = {
        "NBA": preprocess_nba_players_data,
        "Twitter": preprocess_tweets,
    }
    # all_analysis("NBA")
    all_analysis("Twitter")
