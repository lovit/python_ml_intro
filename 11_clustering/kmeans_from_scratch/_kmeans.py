import numpy as np
import scipy as sp
from time import time
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import paired_distances


def kmeans(X, n_clusters, metric, init='random', random_state=None,
    max_iter=10, tol=0.001, epsilon=0, min_size=0, verbose=False):

    """
    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    n_clusters : int
        Number of clusters
    metric : str
        Distance metric
    init : str, callable, or numpy.ndarray
        Initialization method
    random_state : int or None
        Random seed
    max_iter : int
        Maximum number of repetition
    tol : float
        Convergence threshold. if the distance between previous centroid
        and updated centroid is smaller than `tol`, it stops training step.
    epsilon : float
        Maximum distance from centroid to belonging data.
        The points distant more than epsilon are not assigned to any cluster.
    min_size : int
        Minimum number of assigned points.
        The clusters of which size is smaller than the value are disintegrated.
    verbose : Boolean
        If True, it shows training progress.

    Returns
    -------
    centers : numpy.ndarray
        Centroid vectors, shape = (n_clusters, X.shape[1])
    labels : numpy.ndarray
        Integer list, shape = (X.shape[0],)
    """

    # initialize
    centers = initialize(X, n_clusters, init, random_state)
    labels = -np.ones(X.shape[0])

    # train
    centers, labels = kmeans_core(X, centers, metric,
        labels, max_iter, tol, epsilon, min_size, verbose)

    return centers, labels

def kmeans_core(X, centers, metric, labels, max_iter, tol, epsilon, min_size, verbose):
    """
    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    centers : numpy.ndarray
        Initialized centroid vectors
    metric : str
        Distance metric
    labels : numpy.ndarray
        Cluster index list, shape=(n_data,)
    max_iter : int
        Maximum number of repetition
    tol : float
        Convergence threshold. if the distance between previous centroid
        and updated centroid is smaller than `tol`, it stops training step.
    epsilon : float
        Maximum distance from centroid to belonging data.
        The points distant more than epsilon are not assigned to any cluster.
    min_size : int
        Minimum number of assigned points.
        The clusters of which size is smaller than the value are disintegrated.
    verbose : Boolean
        If True, it shows training progress.

    Returns
    -------
    centers : numpy.ndarray
        Centroid vectors, shape = (n_clusters, X.shape[1])
    labels : numpy.ndarray
        Integer list, shape = (X.shape[0],)
    """
    begin_time = time()
    n_clusters = centers.shape[0]

    # repeat
    for i_iter in range(1, max_iter + 1):

        # training
        labels_, dist = reassign(X, centers, metric, epsilon, min_size)
        centers_, cluster_size = update_centroid(X, centers, labels_)

        # convergence check
        diff, n_changes, early_stop = check_convergence(
            centers, labels, centers_, labels_, tol, metric)
        if i_iter == max_iter:
            early_stop = False

        # reinitialize empty clusters
        n_empty_clusters = np.where(cluster_size == 0)[0].shape[0]
        if n_empty_clusters > 0:
            centers_ = reinitialize_empty_cluster_with_distant(
                X, centers_, cluster_size, dist)

        centers = centers_
        labels = labels_

        # verbose
        if verbose:
            strf = verbose_message(i_iter, max_iter, diff, n_changes,
                -1, -1, dist.mean(), early_stop, begin_time)
            print(strf)

        if early_stop:
            break

    return centers, labels

def reassign(X, centers, metric, epsilon=0, min_size=0, do_filter=True):
    """
    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    centers : numpy.ndarray
        Centroid vectors
    metric : str
        Distance metric
    epsilon : float
        Maximum distance from centroid to belonging data.
        The points distant more than epsilon are not assigned to any cluster.
    min_size : int
        Minimum number of assigned points.
        The clusters of which size is smaller than the value are disintegrated.
    do_filter : Boolean
        If True, it executes `epsilon` & `min_size` based filtering.
        Else, it works like k-means.

    Returns
    -------
    labels : numpy.ndarray
        Integer list, shape = (X.shape[0],)
        Not assigned points have -1
    dist : numpy.ndarray
        Distance from their corresponding cluster centroid
    """
    # find closest cluster
    labels, dist = pairwise_distances_argmin_min(X, centers, metric=metric)

    if (not do_filter) or (epsilon == 0 and min_size <= 1):
        return labels, dist

    # epsilon filtering
    labels[np.where(dist >= epsilon)[0]] = -1

    cluster_size = np.bincount(
        labels[np.where(labels >= 0)[0]],
        minlength = centers.shape[0]
    )

    # size filtering
    for label, size in enumerate(cluster_size):
        if size < min_size:
            labels[np.where(labels == label)[0]] = -1

    return labels, dist

def update_centroid(X, centers, labels):
    """
    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    centers : numpy.ndarray
        Centroid vectors of current step
    labels : numpy.ndarray
        Integer list, shape = (X.shape[0],)

    Returns
    -------
    centers_ : numpy.ndarray
        Updated centroid vectors
    cluster_size : numpy.ndarray
        Shape = (n_clusters,)
    """
    n_clusters = centers.shape[0]
    centers_ = np.zeros(centers.shape, dtype=np.float)
    cluster_size = np.bincount(
        labels[np.where(labels >= 0)[0]],
        minlength = n_clusters
    )

    for label, size in enumerate(cluster_size):
        if size == 0:
            centers_[label] == centers[label]
        else:
            idxs = np.where(labels == label)[0]
            centers_[label] = np.asarray(X[idxs,:].sum(axis=0)) / idxs.shape[0]
    return centers_, cluster_size

def reinitialize_empty_cluster_with_distant(X, centers, cluster_size, dist):
    """
    Reinitialize empty clusters with random sampling from distant points

    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    centers : numpy.ndarray
        Centroid vectors
    cluster_size : numpy.ndarray
        Shape = (n_clsuters,)
    dist : numpy.ndarray
        Distance from data and corresponding centroid

    Returns
    -------
    centers : numpy.ndarray
        Partially reinitialized centroid vectors
    """
    cluster_indices = np.where(cluster_size == 0)[0]
    n_empty = cluster_indices.shape[0]
    data_indices = dist.argsort()[-n_empty:]
    initials = X[data_indices,:]
    if sp.sparse.issparse(initials):
        initials = np.asarray(initials.todense())
    centers[cluster_indices,:] = initials
    return centers

def initialize(X, n_clusters, init, random_state):
    """
    Arguments
    ---------
    X : numpy.ndarray or scipy.sparse.csr_matrix
        Training data
    n_clusters : int
        Number of clusters
    init : str, callable, or numpy.ndarray
        Initialization method
    random_state : int or None
        Random seed

    Returns
    -------
    centers : numpy.ndarray
        Initialized centroid vectors, shape = (n_clusters, X.shape[1])
    """
    np.random.seed(random_state)
    if isinstance(init, str) and init == 'random':
        seeds = np.random.permutation(X.shape[0])[:n_clusters]
        if sp.sparse.issparse(X):
            centers = X[seeds,:].todense()
        else:
            centers = X[seeds,:]
    elif hasattr(init, '__array__'):
        centers = np.array(init, dtype=X.dtype)
        if centers.shape[0] != n_clusters:
            raise ValueError('the number of customized initial points '
                'should be same with n_clusters parameter')
    elif callable(init):
        centers = init(X, n_clusters, random_state=random_state)
        centers = np.asarray(centers, dtype=X.dtype)
    else:
        raise ValueError("init method should be "
            "['random', 'callable', 'numpy.ndarray']")
    return centers

def check_convergence(centers, labels, centers_, labels_, tol, metric):
    """
    Check whether k-means is converged or not. 
    If the proportion of reassigned points is smaller than `tol` or
    the difference between `centers` and `centers_` is `tol`,
    it decides to stop training.

    Arguments
    ---------
    centers : numpy.ndarray
        Centroid vectors of current step t    
    labels : numpy.ndarray
        Cluster index, shape = (n_data,)
    centers_ : numpy.ndarray
        Centroid vectors of next step t+1, same shape with `centers`
    labels : numpy.ndarray
        Updated cluster index, shape = (n_data,)
    tol : float
        tolerance parameter
    metric : str
        Distance metric

    Returns
    -------
    diff : float
        Difference between the two centroids
    n_cnahges : int
        Number of re-assigned points
    early_stop : Boolean
        Flag of early-stop
    """
    n_data = labels.shape[0]
    reassign_threshold = n_data * tol
    difference_threshold = tol
    diff = paired_distances(centers, centers_, metric=metric).mean()
    n_changes = np.where(labels != labels_)[0].shape[0]
    early_stop = (diff < difference_threshold) or (n_changes < reassign_threshold)
    return diff, n_changes, early_stop

def verbose_message(i_iter, max_iter, diff, n_changes, n_assigneds,
    n_clusters, inner_dist, early_stop, begin_time, prefix=''):
    """
    Arguments
    ---------
    i_iter : int
        Iteration index
    max_iter : int
        Last iteration index
    diff : float
        Centroid difference
    n_changes : int
        Number of re-assigned points
    n_assigneds : int
        Number of assigned points
    n_clusters : int
        Number of non-empty clusters
    inner_dist : float
        Average inner distance
    early_stop : Boolean
        Flag of early-stop
    begin_time : float
        UNIX time of training begin time

    Returns
    -------
    strf : str
        String formed verbose message
    """
    elapsed_t = time() - begin_time
    remain_t = (max_iter - i_iter) * elapsed_t / i_iter
    ct = as_minute(elapsed_t)
    rt = as_minute(remain_t)
    if rt:
        rt = f'(-{rt})'
    t = f'{ct} {rt}'.strip()
    strf = f'[{prefix}iter: {i_iter}/{max_iter}] #changes: {n_changes}, diff: {diff:.4}, inner: {inner_dist:.4}'
    if n_assigneds > 0:
        strf += f', #assigned: {n_assigneds}'
    if n_clusters > 0:
        strf += f', #clusters: {n_clusters}'
    if t:
        strf += ', time: '+t
    if early_stop:
        strf += f'\nEarly-stop'
    return strf

def as_minute(sec):
    """
    It transforms second to string formed min-sec

    Usage
    -----
        >>> as_minute(153.3)
        $ '2m 33s'

        >>> as_minute(3.21)
        $ '3s'
    """
    m, s = int(sec // 60), int(sec % 60)
    strf = ''
    if m > 0:
        strf += f'{m}m'
    if s > 1:
        strf += ((' ' if strf else '') + f'{s}s')
    return strf
