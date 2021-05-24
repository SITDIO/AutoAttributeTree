# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from .spectral import SpectralClustering


def dudahart2(X, clustering, alpha=0.001):
    """
    Duda-Hart test for whether a data set should be split into two clusters. The

    Based on the R implementation of the fpc package

    Parameters
    ----------
    x : Array-like
        Data matrix
    clustering : Array-like or list
        Vector of integers. Clustering into two clusters
    alpha : float, optional
        Numeric betwwen 0 and 1. Significance level (recommended to be small if this is
        used for estimating the number of clusters), by default 0.001
    """
    assert isinstance(X, np.ndarray), \
        "X must by a Numpy array of shape (n_samples, n_features)"
    assert len(np.unique(clustering)) == 2, \
        "clustering must have labels for 2 clusters"
    n, p = X.shape
    values, counts = np.unique(clustering, return_counts=True)
    W = np.zeros((p, p))

    for clus, cln in zip(values, counts):
        clx = X[clustering == clus, :]
        cclx = np.cov(clx.T)

        if cln < 2:
            cclx = 0
        W += (cln - 1) * cclx

    W1 = (n-1) * np.cov(X.T)
    dh = np.sum(np.diag(W))/np.sum(np.diag(W1))
    z = norm.ppf(1 - alpha)
    compare = 1 - 2/(np.pi * p) - z*np.sqrt(2 * (1 - 8/(np.pi**2 * p)) / (n*p))
    qz = (-dh + 1 - 2/(np.pi * p)) / \
        np.sqrt(2 * (1 - 8/(np.pi**2 * p)) / (n*p))
    p_value = 1 - norm.cdf(qz)
    cluster1 = dh >= compare
    out = {'p_value': p_value, 'dh': dh, 'compare': compare,
           'cluster1': cluster1, 'alpha': alpha, 'z': z}

    return out


def pamk(X, krange=np.arange(1, 11), method='pam', n_components=10,
         alpha=0.001, random_state=None):

    cluster1 = 1 in krange
    avg_sw = np. zeros(len(krange))
    pams = {1: None}
    for i, k in enumerate(krange):
        if k != 1:
            if method == 'pam':
                clust_method = KMedoids(n_clusters=k, init='k-medoids++', max_iter=300,
                                        random_state=None, method='pam')
                clust_method.maps_ = X
            elif method == 'spectral_pam':
                clust_method = SpectralClustering(n_clusters=k, n_components=n_components,
                                                  random_state=random_state,
                                                  assign_labels='kmedoids')
            else:
                raise ValueError('Method not implemented')

            pams[k] = clust_method.fit(X)
            avg_sw[i] = silhouette_score(
                clust_method.maps_, clust_method.labels_)

    k_best = krange[np.argmax(avg_sw)]
    if cluster1:
        cxx = dudahart2(pams[2].maps_, pams[2].labels_, alpha=alpha)
        avg_sw[0] = cxx['p_value']
        cluster1 = cxx['cluster1']

    if cluster1:
        k_best = 1

    return pams[k_best], k_best, avg_sw
