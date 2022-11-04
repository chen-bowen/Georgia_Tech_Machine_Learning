from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from src.config.config import NUM_CLUSTERS_LIST


def k_means_clustering(X):
    """Generate metric scores of k means clustering for NBA dataset given preprocessed X"""
    # loop through the number of clusters with kmeans and record the scores
    siloutte_scores = []
    variance_explained_cluster = []

    for num_clusters in NUM_CLUSTERS_LIST:
        # fit the k means clustering
        clf = KMeans(n_clusters=num_clusters, init="k-means++")
        clf.fit(X)
        predicted_label = clf.predict(X)

        # Silhoutette score
        siloutte_scores.append(
            metrics.silhouette_score(X, predicted_label, metric="euclidean")
        )
        # Variance explained by the cluster
        variance_explained_cluster.append(clf.score(X))

    return {
        "siloutte_scores": siloutte_scores,
        "variance_explained_cluster": variance_explained_cluster,
    }


def expectation_maximization(X):
    """Generate metric scores of k means clustering for NBA dataset given preprocessed X"""
    # loop through the number of clusters with EM and record the scores
    aic_scores = []
    bic_scores = []
    siloutte_scores = []
    variance_explained_cluster = []

    for num_clusters in NUM_CLUSTERS_LIST:
        # fit the gaussian mixture clustering
        clf = GaussianMixture(
            n_components=num_clusters, covariance_type="spherical", init_params="kmeans"
        )
        clf.fit(X)
        predicted_label = clf.predict(X)

        # AIC and BIC
        aic_scores.append(clf.aic(X))
        bic_scores.append(clf.aic(X))

        # Silhoutette score
        siloutte_scores.append(
            metrics.silhouette_score(X, predicted_label, metric="euclidean")
        )
        # Variance explained by the cluster
        variance_explained_cluster.append(clf.score(X))

    return {
        "aic_scores": aic_scores,
        "bic_scores": bic_scores,
        "siloutte_scores": siloutte_scores,
        "variance_explained_cluster": variance_explained_cluster,
    }
