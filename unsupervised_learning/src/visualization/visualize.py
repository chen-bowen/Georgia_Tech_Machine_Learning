import matplotlib.pyplot as plt
import numpy as np
from src.config.config import NUM_CLUSTERS_LIST


def kmeans_visuals(siloutte_scores, variance_explained_cluster, dataset_name):
    """Plot the Siloutte scores and variance explained per cluster for k means"""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"Clustering {dataset_name} with K-Means Clustering")
    # plot Siloutte Scores vs number of clusters
    ax1.plot(NUM_CLUSTERS_LIST, siloutte_scores)
    ax1.set_xlabel("NUmber of Clusters")
    ax1.set_ylabel("Siloutte Scores")
    ax1.set_title("Siloutte Score vs. Number of Clusters")
    # plot the variance explained per cluster
    ax2.plot(NUM_CLUSTERS_LIST, variance_explained_cluster)
    ax2.set_xlabel("NUmber of Clusters")
    ax2.set_ylabel("Variance explained Per Cluster")
    ax2.set_title("Variance Explained vs. Number of Clusters")
    return plt


def expectation_maximization_visuals(
    aic_scores, bic_scores, siloutte_scores, variance_explained_cluster, dataset_name
):
    """
    Plot the aic, bic siloutte scores, and variance explained per cluster
    for expectation maximization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle(f"Clustering {dataset_name} with Expectation Maximization")
    # plot Siloutte Scores vs number of clusters
    ax1.plot(NUM_CLUSTERS_LIST, siloutte_scores)
    ax1.set_xlabel("NUmber of Clusters")
    ax1.set_ylabel("Siloutte Scores")
    ax1.set_title("Siloutte Score vs. Number of Clusters")
    # plot the variance explained per cluster
    ax2.plot(NUM_CLUSTERS_LIST, variance_explained_cluster)
    ax2.set_xlabel("NUmber of Clusters")
    ax2.set_ylabel("Variance explained Per Cluster")
    ax2.set_title("Variance Explained vs. Number of Clusters")
    # plot AIC curves
    ax3.plot(NUM_CLUSTERS_LIST, aic_scores)
    ax3.set_xlabel("NUmber of Clusters")
    ax3.set_ylabel("AIC Score")
    ax3.set_title("AIC Score vs. Number of Clusters")
    # plot BIC curves
    ax4.plot(NUM_CLUSTERS_LIST, bic_scores)
    ax4.set_xlabel("NUmber of Clusters")
    ax4.set_ylabel("BIC Score")
    ax4.set_title("BIC Score vs. Number of Clusters")
    return plt


def explained_variance_plot(explained_variance_ratio, model_name, threshold=0.85):
    """
    Creates a explained variance plot associated with the principal components,
    return the number of fitted components given the threshold
    """
    num_components = len(explained_variance_ratio)
    ind = np.arange(num_components)

    plt.figure(figsize=(15, 8))
    ax = plt.subplot(111)
    cumvals = np.cumsum(explained_variance_ratio)
    ax.bar(ind[:150], explained_variance_ratio[:150])
    ax.plot(ind[:150], cumvals[:150])
    try:
        for i in np.arange(0, 150, 10):
            ax.annotate(
                f"{((str(cumvals[i] * 100)[:4]))}",
                (ind[i], cumvals[i] + 0.02),
                va="bottom",
                ha="center",
                fontsize=12,
            )

        ax.xaxis.set_tick_params(width=0)
        ax.yaxis.set_tick_params(width=2, length=12)
        ax.set_xlim([-1, 150])
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Variance Explained (%)")
        plt.title(f"Explained Variance Per Component with {model_name}")
    except Exception:  # pylint: disable=broad-except
        pass

    fit_components = np.argmax(cumvals > threshold)
    return fit_components, plt


def kurtosis_plot(kurtosis_scores, threshold=None):
    """
    Creates a kurtosis plot for every component
    return the number of components that given the threshold
    """
    plt.figure(figsize=(15, 8))

    # Create bars
    bars = plt.bar(kurtosis_scores["component_num"], kurtosis_scores["score"])

    # Create names on the axis and set title
    plt.xlabel("Component Number")
    plt.ylabel("Kurtosis Score")
    plt.title("Kurtosis Score for Each ICA Component")
    plt.bar_label(bars, padding=0.5)

    # return the component numbers that has kurtosis scores larger than the threshold
    kurtosis_scores = kurtosis_scores.sort_values(by="score", ascending=False)
    fitted_components = kurtosis_scores[kurtosis_scores["score"] >= threshold]

    return fitted_components, plt


def mean_squared_error_plot(mse_reconstruct, threshold):
    """
    Create a mean squared error for reconstruction
    return the number of components with mse less than threshold
    """
    plt.figure(figsize=(15, 8))

    # Create bars
    bars = plt.bar(mse_reconstruct["component_num"], mse_reconstruct["mse"])

    # Create names on the axis and set title
    plt.xlabel("Component Number")
    plt.ylabel("MSE Reconstruction")
    plt.title("MSE Reconstruction for Each Random Projection Component")
    plt.bar_label(bars, padding=0.5)

    # return the component numbers that has MSE less than the threshold
    mse_reconstruct = mse_reconstruct.sort_values(by="mse", ascending=False)
    fitted_components = mse_reconstruct[mse_reconstruct["mse"] <= threshold]

    return fitted_components, plt
