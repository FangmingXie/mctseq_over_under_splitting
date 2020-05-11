"""
"""
from __init__ import *
from scipy import sparse
from annoy import AnnoyIndex

def build_knn_map(X, metric='euclidean', n_trees=10, verbose=True):
    """X is expected to have low feature dimensions (n_obs, n_features) with (n_features <= 50)

    return:
         t: annoy knn object, can be used in the following ways 
                t.get_nns_by_vector
                t.get_nns_by_item
    """
    ti = time.time()

    n_obs, n_f = X.shape
    t = AnnoyIndex(n_f, metric=metric)  # Length of item vector that will be indexed
    for i, X_row in enumerate(X):
        t.add_item(i, X_row)
    t.build(n_trees) # 10 trees
    if verbose:
        print("Time used to build kNN map {}".format(time.time()-ti))
    return t 

def get_knn_by_items(t, k, 
    form='list', 
    search_k=-1, 
    include_distances=False,
    verbose=True, 
    ):
    """Get kNN for each item in the knn map t
    """
    ti = time.time()
    # set up
    n_obs = t.get_n_items()
    n_f = t.f
    if k > n_obs:
        print("Actual k: {}->{} due to low n_obs".format(k, n_obs))
        k = n_obs

    knn = [0]*(n_obs)
    knn_dist = [0]*(n_obs)
    # this block of code can be optimized
    if include_distances:
        for i in range(n_obs):
            res = t.get_nns_by_item(i, k, search_k=search_k, include_distances=include_distances)
            knn[i] = res[0]
            knn_dist[i] = res[1]
    else:
        for i in range(n_obs):
            res = t.get_nns_by_item(i, k, search_k=search_k, include_distances=include_distances) 
            knn[i] = res

    knn = np.array(knn)
    knn_dist = np.array(knn_dist)

    if verbose:
        print("Time used to get kNN {}".format(time.time()-ti))

    if form == 'adj':
        # row col 1/dist 
        row_inds = np.repeat(np.arange(n_obs), k)
        col_inds = np.ravel(knn)
        if include_distances:
            data = np.ravel(knn_dist) 
        else:
            data = [1]*len(row_inds)
        knn_dist_mat = sparse.coo_matrix((data, (row_inds, col_inds)), shape=(n_obs, n_obs))
        return knn_dist_mat
    elif form == 'list':  #
        if include_distances:
            return knn, knn_dist
        else:
            return knn
    else:
        raise ValueError("Choose from 'adj' and 'list'")

def get_knn_by_vectors(t, X, k, 
    form='list', 
    search_k=-1, 
    include_distances=False,
    verbose=True, 
    ):
    """Get kNN for each row vector of X 
    """
    ti = time.time()
    # set up
    n_obs = t.get_n_items()
    n_f = t.f
    n_obs_test, n_f_test = X.shape
    assert n_f_test == n_f

    if k > n_obs:
        print("Actual k: {}->{} due to low n_obs".format(k, n_obs))
        k = n_obs

    knn = [0]*(n_obs_test)
    knn_dist = [0]*(n_obs_test)
    if include_distances:
        for i, vector in enumerate(X):
            res = t.get_nns_by_vector(vector, k, search_k=search_k, include_distances=include_distances) 
            knn[i] = res[0]
            knn_dist[i] = res[1]
    else:
        for i, vector in enumerate(X):
            res = t.get_nns_by_vector(vector, k, search_k=search_k, include_distances=include_distances) 
            knn[i] = res

    knn = np.array(knn)
    knn_dist = np.array(knn_dist)

    if verbose:
        print("Time used to get kNN {}".format(time.time()-ti))

    if form == 'adj':
        # row col 1/dist 
        row_inds = np.repeat(np.arange(n_obs_test), k)
        col_inds = np.ravel(knn)
        if include_distances:
            data = np.ravel(knn_dist) 
        else:
            data = [1]*len(row_inds)
        knn_dist_mat = sparse.coo_matrix((data, (row_inds, col_inds)), shape=(n_obs_test, n_obs))
        return knn_dist_mat
    elif form == 'list':  #
        if include_distances:
            return knn, knn_dist
        else:
            return knn
    else:
        raise ValueError("Choose from 'adj' and 'list'")

def gen_knn_annoy(X, k, form='list', 
    metric='euclidean', n_trees=10, search_k=-1, verbose=True, 
    include_distances=False,
    ):
    """X is expected to have low feature dimensions (n_obs, n_features) with (n_features <= 50)
    """
    ti = time.time()

    n_obs, n_f = X.shape
    t = build_knn_map(X, metric=metric, n_trees=n_trees, verbose=verbose)

    return get_knn_by_items(t, k,                             
                            form=form, 
                            search_k=search_k, 
                            include_distances=include_distances,
                            verbose=verbose, 
                            )

def gen_knn_annoy_train_test(X_train, X_test, k, 
    form='list', 
    metric='euclidean', n_trees=10, search_k=-1, verbose=True, 
    include_distances=False,
    ):
    """X is expected to have low feature dimensions (n_obs, n_features) with (n_features <= 50)
    For each row in X_test, find k nearest neighbors in X_train
    """
    ti = time.time()
    
    n_obs, n_f = X_train.shape
    n_obs_test, n_f_test = X_test.shape
    assert n_f == n_f_test 
    
    t = build_knn_map(X_train, metric=metric, n_trees=n_trees, verbose=verbose)
    return get_knn_by_vectors(t, X_test, k, 
                                form=form, 
                                search_k=search_k, 
                                include_distances=include_distances,
                                verbose=verbose, 
                                )

