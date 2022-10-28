import matplotlib.pyplot as plt
import numpy as np


def pca_variance_plot(pca):
    """
    Creates a variance plot associated with the principal components

    INPUT: pca - the result of instantian of PCA object in scikit learn

    OUTPUT: number of components suggested
    """
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_

    plt.figure(figsize=(15, 8))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind[:150], vals[:150])
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
        plt.title("Explained Variance Per Principal Component")
    except Exception:  # pylint: disable=broad-except
        pass

    fit_components = np.argmax(cumvals > 0.85)
    return fit_components
