import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection


def reduce_by_pca(data, num_components):
    """
    Transform the data with PCA with number of components
    """
    pca = PCA(num_components)
    pca_model= pca.fit(data)
    reduced_data = pca_model.transform(data)
    return pd.DataFrame(reduced_data)

def reduce_by_ica(data, num_components):
    """
    Transform the data with ICA given the number of components
    """
    ica = FastICA(num_components)
    ica_model= ica.fit(data)
    reduced_data = ica_model.transform(data)
    return pd.DataFrame(reduced_data)

def reduce_by_random_projection(data, num_components):
    """
    Transform the data with gaussian random projection
    """
    rp = GaussianRandomProjection(n_components=num_components)
    rp_model = rp.fit(data)
    reduced_data = rp_model.transform(data)
    return pd.DataFrame(reduced_data)

def reduce_by_LDA(X, y, num_components):
    """
    Transform the data with LDA given the number of components
    """
    lda = LinearDiscriminantAnalysis(
    n_components=num_components,
    )
    lda_model = lda.fit(X, y)
    reduced_data = lda_model.transform(X)
    return reduced_data
    