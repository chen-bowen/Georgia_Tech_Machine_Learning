import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
from sklearn.random_projection import GaussianRandomProjection


def reduce_by_pca(data, num_components):
    """
    Transform the data with PCA with number of components, 
    return the reduced data and  pca object
    """
    pca = PCA(num_components)
    pca_model= pca.fit(data)
    reduced_data = pca_model.transform(data)
    return pd.DataFrame(reduced_data), pca_model

def reduce_by_ica(data, num_components):
    """
    Transform the data with ICA given the number of components, 
    return the reduced data and the kurtosis for each component
    """
    ica = FastICA(num_components)
    ica_model= ica.fit(data)
    reduced_data = ica_model.transform(data)
    kurt = kurtosis(reduced_data)
    return pd.DataFrame(reduced_data), kurt

def reduce_by_random_projection(data, num_components):
    """
    Transform the data with gaussian random projection,
    return the reduced data and mean squared error for reconstruction
    """
    rp = GaussianRandomProjection(n_components=num_components)
    rp_model = rp.fit(data)
    reduced_data = rp_model.transform(data)
    # find P_inverse and get the reconstructed_data
    p_inverse = np.linalg.pinv(rp_model.components_.T)
    reconstructed_data = reduced_data.dot(p_inverse)
    # calculate mean squared error for reconstruction
    mse_reconstructed = mean_squared_error(data, reconstructed_data)
    return pd.DataFrame(reduced_data), mse_reconstructed

def reduce_by_LDA(X, y, num_components):
    """
    Transform the data with LDA given the number of components
    """
    lda = LinearDiscriminantAnalysis(
    n_components=num_components,
    )
    lda_model = lda.fit(X, y)
    reduced_data = lda_model.transform(X)
    return reduced_data, lda_model
    