# -*- coding: utf-8 -*-

# from sklearn.neighbors import KDTree


class BinaryTree(object):
    # no doc
    def get_arrays(self):  # real signature unknown; restored from __doc__
        """
        get_arrays(self)

                Get data and node arrays.

                Returns
                -------
                arrays: tuple of array
                    Arrays for storing tree data, index, node data and node bounds.
        """
        pass

    def get_n_calls(self):  # real signature unknown; restored from __doc__
        """
        get_n_calls(self)

                Get number of calls.

                Returns
                -------
                n_calls: int
                    number of distance computation calls
        """
        pass

    def get_tree_stats(self):  # real signature unknown; restored from __doc__
        """
        get_tree_stats(self)

                Get tree status.

                Returns
                -------
                tree_stats: tuple of int
                    (number of trims, number of leaves, number of splits)
        """
        pass

    def kernel_density(self, X, h, kernel='gaussian', atol=0, rtol=1, *args,
                       **kwargs):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        kernel_density(self, X, h, kernel='gaussian', atol=0, rtol=1E-8,
                               breadth_first=True, return_log=False)

                Compute the kernel density estimate at points X with the given kernel,
                using the distance metric specified at tree creation.

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    An array of points to query.  Last dimension should match dimension
                    of training data.
                h : float
                    the bandwidth of the kernel
                kernel : str, default="gaussian"
                    specify the kernel to use.  Options are
                    - 'gaussian'
                    - 'tophat'
                    - 'epanechnikov'
                    - 'exponential'
                    - 'linear'
                    - 'cosine'
                    Default is kernel = 'gaussian'
                atol, rtol : float, default=0, 1e-8
                    Specify the desired relative and absolute tolerance of the result.
                    If the true result is K_true, then the returned result K_ret
                    satisfies ``abs(K_true - K_ret) < atol + rtol * K_ret``
                    The default is zero (i.e. machine precision) for both.
                breadth_first : bool, default=False
                    If True, use a breadth-first search.  If False (default) use a
                    depth-first search.  Breadth-first is generally faster for
                    compact kernels and/or high tolerances.
                return_log : bool, default=False
                    Return the logarithm of the result.  This can be more accurate
                    than returning the result itself for narrow kernels.

                Returns
                -------
                density : ndarray of shape X.shape[:-1]
                    The array of (log)-density evaluations
        """
        pass

    def query(self, X, k=1, return_distance=True, dualtree=False,
              breadth_first=False):  # real signature unknown; restored from __doc__
        """
        query(X, k=1, return_distance=True,
                      dualtree=False, breadth_first=False)

                query the tree for the k nearest neighbors

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    An array of points to query
                k : int, default=1
                    The number of nearest neighbors to return
                return_distance : bool, default=True
                    if True, return a tuple (d, i) of distances and indices
                    if False, return array i
                dualtree : bool, default=False
                    if True, use the dual tree formalism for the query: a tree is
                    built for the query points, and the pair of trees is used to
                    efficiently search this space.  This can lead to better
                    performance as the number of points grows large.
                breadth_first : bool, default=False
                    if True, then query the nodes in a breadth-first manner.
                    Otherwise, query the nodes in a depth-first manner.
                sort_results : bool, default=True
                    if True, then distances and indices of each point are sorted
                    on return, so that the first column contains the closest points.
                    Otherwise, neighbors are returned in an arbitrary order.

                Returns
                -------
                i    : if return_distance == False
                (d,i) : if return_distance == True

                d : ndarray of shape X.shape[:-1] + k, dtype=double
                    Each entry gives the list of distances to the neighbors of the
                    corresponding point.

                i : ndarray of shape X.shape[:-1] + k, dtype=int
                    Each entry gives the list of indices of neighbors of the
                    corresponding point.
        """
        pass

    def query_radius(self, X, r, return_distance=False, count_only=False,
                     sort_results=False):  # real signature unknown; restored from __doc__
        """
        query_radius(X, r, return_distance=False,
                count_only=False, sort_results=False)

                query the tree for neighbors within a radius r

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    An array of points to query
                r : distance within which neighbors are returned
                    r can be a single value, or an array of values of shape
                    x.shape[:-1] if different radii are desired for each point.
                return_distance : bool, default=False
                    if True,  return distances to neighbors of each point
                    if False, return only neighbors
                    Note that unlike the query() method, setting return_distance=True
                    here adds to the computation time.  Not all distances need to be
                    calculated explicitly for return_distance=False.  Results are
                    not sorted by default: see ``sort_results`` keyword.
                count_only : bool, default=False
                    if True,  return only the count of points within distance r
                    if False, return the indices of all points within distance r
                    If return_distance==True, setting count_only=True will
                    result in an error.
                sort_results : bool, default=False
                    if True, the distances and indices will be sorted before being
                    returned.  If False, the results will not be sorted.  If
                    return_distance == False, setting sort_results = True will
                    result in an error.

                Returns
                -------
                count       : if count_only == True
                ind         : if count_only == False and return_distance == False
                (ind, dist) : if count_only == False and return_distance == True

                count : ndarray of shape X.shape[:-1], dtype=int
                    Each entry gives the number of neighbors within a distance r of the
                    corresponding point.

                ind : ndarray of shape X.shape[:-1], dtype=object
                    Each element is a numpy integer array listing the indices of
                    neighbors of the corresponding point.  Note that unlike
                    the results of a k-neighbors query, the returned neighbors
                    are not sorted by distance by default.

                dist : ndarray of shape X.shape[:-1], dtype=object
                    Each element is a numpy double array listing the distances
                    corresponding to indices in i.
        """
        pass

    def reset_n_calls(self):  # real signature unknown; restored from __doc__
        """
        reset_n_calls(self)

                Reset number of calls to 0.
        """
        pass

    def two_point_correlation(self, X, r, dualtree=False):  # real signature unknown; restored from __doc__
        """
        two_point_correlation(X, r, dualtree=False)

                Compute the two-point correlation function

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    An array of points to query.  Last dimension should match dimension
                    of training data.
                r : array-like
                    A one-dimensional array of distances
                dualtree : bool, default=False
                    If True, use a dualtree algorithm.  Otherwise, use a single-tree
                    algorithm.  Dual tree algorithms can have better scaling for
                    large N.

                Returns
                -------
                counts : ndarray
                    counts[i] contains the number of pairs of points with distance
                    less than or equal to r[i]
        """
        pass

    def _update_memviews(self, *args, **kwargs):  # real signature unknown
        pass

    def _update_sample_weight(self, *args, **kwargs):  # real signature unknown
        pass

    def __getstate__(self, *args, **kwargs):  # real signature unknown
        """ get state for pickling """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        """ reduce method used for pickling """
        pass

    def __setstate__(self, *args, **kwargs):  # real signature unknown
        """ set state for pickling """
        pass

    data = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    idx_array = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    node_bounds = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    node_data = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    sample_weight = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    sum_weight = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    valid_metrics = [
        'euclidean',
        'l2',
        'minkowski',
        'p',
        'manhattan',
        'cityblock',
        'l1',
        'chebyshev',
        'infinity',
    ]
    __pyx_vtable__ = None  # (!) real value is '<capsule object NULL at 0x7f8e18460510>'


class KDTree(BinaryTree):
    """
    KDTree(X, leaf_size=40, metric='minkowski', **kwargs)

    KDTree for fast generalized N-point problems

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        n_samples is the number of points in the data set, and
        n_features is the dimension of the parameter space.
        Note: if X is a C-contiguous array of doubles then data will
        not be copied. Otherwise, an internal copy will be made.

    leaf_size : positive int, default=40
        Number of points at which to switch to brute-force. Changing
        leaf_size will not affect the results of a query, but can
        significantly impact the speed of a query and the memory required
        to store the constructed tree.  The amount of memory needed to
        store the tree scales as approximately n_samples / leaf_size.
        For a specified ``leaf_size``, a leaf node is guaranteed to
        satisfy ``leaf_size <= n_points <= 2 * leaf_size``, except in
        the case that ``n_samples < leaf_size``.

    metric : str or DistanceMetric object
        the distance metric to use for the tree.  Default='minkowski'
        with p=2 (that is, a euclidean metric). See the documentation
        of the DistanceMetric class for a list of available metrics.
        kd_tree.valid_metrics gives a list of the metrics which
        are valid for KDTree.

    Additional keywords are passed to the distance metric class.
    Note: Callable functions in the metric parameter are NOT supported for KDTree
    and Ball Tree. Function call overhead will result in very poor performance.

    Attributes
    ----------
    data : memory view
        The training data

    Examples
    --------
    Query for k-nearest neighbors

        >>> import numpy as np
        >>> rng = np.random.RandomState(0)
        >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
        >>> tree = KDTree(X, leaf_size=2)              # doctest: +SKIP
        >>> dist, ind = tree.query(X[:1], k=3)                # doctest: +SKIP
        >>> print(ind)  # indices of 3 closest neighbors
        [0 3 1]
        >>> print(dist)  # distances to 3 closest neighbors
        [ 0.          0.19662693  0.29473397]

    Pickle and Unpickle a tree.  Note that the state of the tree is saved in the
    pickle operation: the tree needs not be rebuilt upon unpickling.

        >>> import numpy as np
        >>> import pickle
        >>> rng = np.random.RandomState(0)
        >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
        >>> tree = KDTree(X, leaf_size=2)        # doctest: +SKIP
        >>> s = pickle.dumps(tree)                     # doctest: +SKIP
        >>> tree_copy = pickle.loads(s)                # doctest: +SKIP
        >>> dist, ind = tree_copy.query(X[:1], k=3)     # doctest: +SKIP
        >>> print(ind)  # indices of 3 closest neighbors
        [0 3 1]
        >>> print(dist)  # distances to 3 closest neighbors
        [ 0.          0.19662693  0.29473397]

    Query for neighbors within a given radius

        >>> import numpy as np
        >>> rng = np.random.RandomState(0)
        >>> X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
        >>> tree = KDTree(X, leaf_size=2)     # doctest: +SKIP
        >>> print(tree.query_radius(X[:1], r=0.3, count_only=True))
        3
        >>> ind = tree.query_radius(X[:1], r=0.3)  # doctest: +SKIP
        >>> print(ind)  # indices of neighbors within distance 0.3
        [3 0 1]


    Compute a gaussian kernel density estimate:

        >>> import numpy as np
        >>> rng = np.random.RandomState(42)
        >>> X = rng.random_sample((100, 3))
        >>> tree = KDTree(X)                # doctest: +SKIP
        >>> tree.kernel_density(X[:3], h=0.1, kernel='gaussian')
        array([ 6.94114649,  7.83281226,  7.2071716 ])

    Compute a two-point auto-correlation function

        >>> import numpy as np
        >>> rng = np.random.RandomState(0)
        >>> X = rng.random_sample((30, 3))
        >>> r = np.linspace(0, 1, 5)
        >>> tree = KDTree(X)                # doctest: +SKIP
        >>> tree.two_point_correlation(X, r)
        array([ 30,  62, 278, 580, 820])
    """

    def __init__(self, X, leaf_size=40, metric='minkowski', **kwargs):  # real signature unknown; restored from __doc__
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    __pyx_vtable__ = None  # (!) real value is '<capsule object NULL at 0x7f8e18460570>'
