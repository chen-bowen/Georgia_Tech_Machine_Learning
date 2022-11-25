from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from src.config.config import NUM_CLUSTERS_LIST, RANDOM_SEED


def k_means_clustering_experiment(data):
    """Generate metric scores of k means clustering for NBA dataset given preprocessed data"""
    # loop through the number of clusters with kmeans and record the scores
    siloutte_scores = []
    average_distance_cluster = []

    for num_clusters in NUM_CLUSTERS_LIST:
        # fit the k means clustering
        clf = KMeans(
            n_clusters=num_clusters,
            init="k-means++",
            random_state=RANDOM_SEED,
        )
        clf.fit(data)
        predicted_label = clf.predict(data)

        # Silhoutette score
        siloutte_scores.append(
            metrics.silhouette_score(data, predicted_label, metric="euclidean")
        )
        # Variance explained by the cluster
        average_distance_cluster.append(abs(clf.score(data)))

    return {
        "siloutte_scores": siloutte_scores,
        "average_distance_cluster": average_distance_cluster,
    }


def expectation_maximization_experiment(data):
    """Generate metric scores of k means clustering for NBA dataset given preprocessed data"""
    # loop through the number of clusters with EM and record the scores
    siloutte_scores = []
    average_distance_cluster = []

    for num_clusters in NUM_CLUSTERS_LIST:
        # fit the gaussian mixture clustering
        clf = GaussianMixture(
            n_components=num_clusters,
            covariance_type="spherical",
            init_params="kmeans",
            random_state=RANDOM_SEED,
        )
        clf.fit(data)
        predicted_label = clf.predict(data)

        # Silhoutette score
        siloutte_scores.append(
            metrics.silhouette_score(data, predicted_label, metric="euclidean")
        )
        # Variance explained by the cluster
        average_distance_cluster.append(abs(clf.score(data)))

    return {
        "siloutte_scores": siloutte_scores,
        "average_distance_cluster": average_distance_cluster,
    }


def k_means_clustering(data, num_clusters):
    """
    K means clustering and returned the original dataset
    with the clustering label given the number of clusters
    """
    clf = KMeans(n_clusters=num_clusters, init="k-means++", random_state=RANDOM_SEED)
    clf.fit(data)
    predicted_label = clf.predict(data)
    # get metrics
    sil_score = metrics.silhouette_score(data, predicted_label, metric="euclidean")
    average_distance_cluster = abs(clf.score(data))
    # add cluster labels
    data["cluster_label"] = predicted_label
    return data, {
        "silhouette_score": sil_score,
        "average_distance_cluster": average_distance_cluster,
    }


def expectation_maximization_clustering(data, num_clusters):
    """
    Expectation Maximization clustering and returned the original dataset
    with the clustering label given the number of clusters
    """
    clf = GaussianMixture(
        n_components=num_clusters,
        covariance_type="spherical",
        init_params="kmeans",
        random_state=RANDOM_SEED,
    )
    clf.fit(data)
    predicted_label = clf.predict(data)
    # get metrics
    sil_score = metrics.silhouette_score(data, predicted_label, metric="euclidean")
    average_distance_cluster = abs(clf.score(data))
    # add cluster label
    data["cluster_label"] = predicted_label
    return data, {
        "silhouette_score": sil_score,
        "average_distance_cluster": average_distance_cluster,
    }
