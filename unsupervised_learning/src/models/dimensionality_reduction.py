import warnings

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.random_projection import GaussianRandomProjection

warnings.filterwarnings("ignore")


def reduce_by_pca(data):
    """
    Transform the data with PCA with number of components,
    return the reduced data and  pca object
    """
    pca = PCA()
    pca_model = pca.fit(data)
    reduced_data = pca_model.transform(data)
    return pd.DataFrame(reduced_data), pca_model.explained_variance_ratio_


def reduce_by_ica(data):
    """
    Transform the data with ICA given the number of components,
    return the reduced data and the kurtosis for each component
    """
    ica = FastICA()
    ica_model = ica.fit(data)
    reduced_data = ica_model.transform(data)
    kurt = kurtosis(reduced_data)
    return pd.DataFrame(reduced_data), pd.DataFrame(
        {"component_num": np.arange(len(kurt)), "score": kurt}
    )


def reduce_by_random_projection(data):
    """
    Transform the data with gaussian random projection,
    return the reduced data and mean squared error for reconstruction
    """
    mse_reconstructed = []
    reduced_data_list = []
    for num_components in range(1, data.shape[1]):
        rp = GaussianRandomProjection(n_components=num_components)
        rp_model = rp.fit(data)
        reduced_data = rp_model.transform(data)
        reduced_data_list.append(pd.DataFrame(reduced_data))
        # find P_inverse and get the reconstructed_data
        p_inverse = np.linalg.pinv(rp_model.components_.T)
        reconstructed_data = reduced_data.dot(p_inverse)
        # calculate mean squared error for reconstruction
        mse_reconstructed.append(mean_squared_error(data, reconstructed_data))

    return reduced_data_list, pd.DataFrame(
        {
            "component_num": np.arange(1, len(mse_reconstructed) + 1),
            "score": mse_reconstructed,
        }
    )


def reduce_by_svd(data):
    """
    Transform the data with SVD given the number of components
    """
    svd = TruncatedSVD(n_components=data.shape[1])
    svd_model = svd.fit(data)
    reduced_data = svd_model.transform(data)
    return pd.DataFrame(reduced_data), svd_model.explained_variance_ratio_
