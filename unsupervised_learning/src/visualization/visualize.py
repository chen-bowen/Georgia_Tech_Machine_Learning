import matplotlib.pyplot as plt
import numpy as np
from src.config.config import NUM_CLUSTERS_LIST


def kmeans_visuals(siloutte_scores, variance_explained_cluster, dataset_name):
    """Plot the Siloutte scores and variance explained per cluster for k means"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle(f"Clustering {dataset_name} with K-Means Clustering")
    # plot Siloutte Scores vs number of clusters
    ax1.plot(NUM_CLUSTERS_LIST, siloutte_scores)
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("Siloutte Scores")
    ax1.set_title("Siloutte Score vs. # Clusters")
    # plot the variance explained per cluster
    ax2.plot(NUM_CLUSTERS_LIST, variance_explained_cluster)
    ax2.set_xlabel("NUmber of Clusters")
    ax2.set_ylabel("Average Distance Between Points")
    ax2.set_title("Average Distance vs. # Clusters")
    return plt


def expectation_maximization_visuals(
    aic_scores, bic_scores, siloutte_scores, variance_explained_cluster, dataset_name
):
    """
    Plot the aic, bic siloutte scores, and variance explained per cluster
    for expectation maximization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
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
                (ind[i], cumvals[i] + 0.01),
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
    return np.arange(fit_components), plt


def kurtosis_plot(kurtosis_scores, model_name=None, threshold=None):
    """
    Creates a kurtosis plot for every component
    return the number of components that given the threshold
    """
    plt.figure(figsize=(15, 8))

    # Create bars
    bars = plt.bar(kurtosis_scores["component_num"], round(kurtosis_scores["score"], 1))

    # Create names on the axis and set title
    plt.xlabel("Component Number")
    plt.ylabel("Kurtosis Score")
    plt.title("Kurtosis Score for Each ICA Component")
    plt.bar_label(bars, padding=0.5)

    # return the component numbers that has kurtosis scores larger than the threshold
    kurtosis_scores = kurtosis_scores.sort_values(by="score", ascending=False)
    fitted_components = kurtosis_scores[kurtosis_scores["score"] >= threshold][
        "component_num"
    ].tolist()
    return fitted_components, plt


def mean_squared_error_plot(mse_reconstruct, model_name=None, threshold=None):
    """
    Create a mean squared error for reconstruction
    return the number of components with mse less than threshold
    """
    plt.figure(figsize=(15, 8))

    # Create bars
    plt.bar(mse_reconstruct["component_num"], mse_reconstruct["score"])

    # Create names on the axis and set title
    plt.xlabel("Number of Components")
    plt.ylabel("MSE Reconstruction")
    plt.title("MSE Reconstruction for Using x Random Projection Components")

    # return the component numbers that has MSE less than the threshold
    fitted_components = len(mse_reconstruct[mse_reconstruct["score"] >= threshold])
    return np.arange(fitted_components), plt


def plot_learning_curve(
    model_name,
    estimator,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 10),
):
    """
    Referenced from:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve

    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    model_name: the name of the model (used for saving figures)
    dataset_name: NBA or Twitter
    params_type: Default or Optimal
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 4, figsize=(24, 5))

    axes[0].set_title(
        f"""
        Learning Curve with {model_name} Transformation on the NBA DataSet
        """
    )
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training Instances")
    axes[0].set_ylabel("Accuracy")

    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(  # type: ignore
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    score_times_mean = np.mean(score_times, axis=1)
    score_times_std = np.std(score_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit Times")
    axes[1].set_title("Scalability - Training Wall Time")

    # Plot n_samples vs score_times
    axes[2].grid()
    axes[2].plot(train_sizes, score_times_mean, "o-")
    axes[2].fill_between(
        train_sizes,
        score_times_mean - score_times_std,
        score_times_mean + score_times_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("Scoring Instances")
    axes[2].set_ylabel("Fit Times")
    axes[2].set_title("Scalability - Scoring Wall Time")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[3].grid()
    axes[3].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[3].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[3].set_xlabel("fit_times")
    axes[3].set_ylabel("Accuracy")
    axes[3].set_title("Performance - Accuracy on Validation Data")

    return plt
