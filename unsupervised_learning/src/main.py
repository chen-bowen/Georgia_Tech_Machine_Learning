import matplotlib.pyplot as plt
from src.config.config import THRESHOLD_MAP
from src.models.clustering import expectation_maximization, k_means_clustering
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
)

CLUSTERING_MODEL_MAPPING = {
    "K-means": k_means_clustering,
    "Expectation Maximization": expectation_maximization,
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


def clustering_analysis(data, model_name, dataset_name):
    """
    Performs clustering analysis given the dataset and model name and save the visualizations
    data - the dataframe that contains the data to be clustered, either NBA or Twitter, full
            or reduced
    model_name - the string name of the clustering model
    dataset_name - the string name of the dataset
    """
    # get the metrics tuple
    metrics = CLUSTERING_MODEL_MAPPING[model_name](data)
    # make plots
    CLUSTERING_VISUAL_FUNC_MAPPING[model_name](**metrics, dataset_name=dataset_name)
    plt.suptitle(
        f"Clustering using {model_name} Model For {dataset_name} Dataset", fontsize=20
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig(f"../reports/figures/{model_name}_{dataset_name}.jpg", dpi=150)


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


# if __name__ == "__main__":
