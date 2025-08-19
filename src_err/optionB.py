import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import inv

def _prepare_scaler_and_metric(X, standardize=True, metric='euclidean', cov=None):
    """
    Returns (transform_fn, metric, metric_params)
    - transform_fn: function to standardize/whiten new X
    - metric_params: params for sklearn NearestNeighbors (e.g., VI for mahalanobis)
    """
    if not standardize and metric != 'mahalanobis':
        def transform_fn(Z): return Z
        return transform_fn, metric, {}

    if metric == 'mahalanobis':
        # If cov not provided, estimate from X
        if cov is None:
            cov = np.cov(X, rowvar=False)
        VI = inv(cov + 1e-12*np.eye(cov.shape[0]))
        def transform_fn(Z): return Z  # Mahalanobis handles scaling via VI
        return transform_fn, 'mahalanobis', {'VI': VI}

    # Default: standardize then Euclidean
    scaler = StandardScaler().fit(X)
    def transform_fn(Z): return scaler.transform(Z)
    return transform_fn, 'euclidean', {}
    

def knn_conditional_variance(
    X_train, y_train, X_query=None, k=25,
    metric='euclidean', standardize=True, cov=None,
    groups=None, return_cov=False, min_k=5
):
    """
    Option B: kNN conditional variance estimator.
    
    Parameters
    ----------
    X_train : (n, d) array
    y_train : (n,) or (n, p) array
    X_query : (m, d) array or None -> if None, estimates at X_train
    k       : int, target neighbors
    metric  : 'euclidean' | 'minkowski' | 'mahalanobis'
    standardize : bool, standardize inputs when not using mahalanobis
    cov     : (d, d) covariance for mahalanobis; estimated if None
    groups  : (n,) array of group labels; if not None, neighbors drawn only from same group
    return_cov : bool, for vector targets return full local covariance per query.
    min_k   : fallback minimum neighbors if dataset smaller than k
    
    Returns
    -------
    If scalar y: (m,) std estimate
    If vector y and return_cov=False: (m, p) per-component std
    If vector y and return_cov=True : list of (p, p) covariance matrices length m
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    n, d = X_train.shape
    vector_target = (y_train.ndim == 2)
    p = y_train.shape[1] if vector_target else 1

    if X_query is None:
        X_query = X_train
    X_query = np.asarray(X_query)

    # Set up transform + metric
    transform_fn, eff_metric, metric_params = _prepare_scaler_and_metric(
        X_train, standardize=standardize, metric=metric, cov=cov
    )
    Xt = transform_fn(X_train)
    Xq = transform_fn(X_query)

    if groups is None:
        # Single neighbor index structure for all
        n_neighbors = max(min(k, n), min_k)
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=eff_metric,
            metric_params=metric_params if metric_params else None,
            algorithm='brute' if eff_metric == 'mahalanobis' else 'auto'
        ).fit(Xt)

        idx = nn.kneighbors(Xq, return_distance=False)
        # compute local stats
        if not vector_target:
            y_nb = y_train[idx]                  # (m, k)
            m_loc = np.mean(y_nb, axis=1)        # (m,)
            v_loc = np.sum((y_nb - m_loc[:,None])**2, axis=1) / np.maximum(y_nb.shape[1]-1, 1)
            return np.sqrt(np.maximum(v_loc, 0.0))
        else:
            y_nb = y_train[idx]                  # (m, k, p)
            m_loc = np.mean(y_nb, axis=1, keepdims=True)   # (m,1,p)
            if return_cov:
                covs = []
                for i in range(y_nb.shape[0]):
                    Yc = y_nb[i] - m_loc[i]                    # (k,p)
                    # (p,p) sample covariance
                    C = (Yc.T @ Yc) / max(Yc.shape[0]-1, 1)
                    covs.append(C)
                return covs
            else:
                var = np.var(y_nb, axis=1, ddof=1)             # (m, p)
                var = np.maximum(var, 0.0)
                return np.sqrt(var)
    else:
        # Group-aware: build neighbor structures per group
        Xq = np.asarray(X_query)
        groups = np.asarray(groups)
        if Xq.shape[0] == X_train.shape[0] and np.allclose(Xq, X_train):
            # If querying training points, we can use their groups directly
            q_groups = groups
        else:
            # If querying new points and groups are needed, user must pass group labels for query
            # For safety, require q_groups passed via groups_query in that case.
            raise ValueError("When 'groups' is provided, X_query must be training points or you must "
                             "split queries and call per-group. For new X, call this function per group.")

        out = []
        for g in np.unique(groups):
            mask = (groups == g)
            if not np.any(mask):
                continue
            Xt_g = Xt[mask]
            y_g  = y_train[mask]
            Xq_g_idx = np.where(q_groups == g)[0]
            Xq_g = Xq[Xq_g_idx]

            n_neighbors = max(min(k, Xt_g.shape[0]), min_k)
            
            nn = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric=eff_metric,
                metric_params=metric_params if metric_params else None,
                algorithm='brute' if eff_metric == 'mahalanobis' else 'auto'
            ).fit(Xt_g)

            idx = nn.kneighbors(Xq_g, return_distance=False)
            if not vector_target:
                y_nb = y_g[idx]
                m_loc = np.mean(y_nb, axis=1)
                v_loc = np.sum((y_nb - m_loc[:,None])**2, axis=1) / np.maximum(y_nb.shape[1]-1, 1)
                out.append((Xq_g_idx, np.sqrt(np.maximum(v_loc, 0.0))))
            else:
                y_nb = y_g[idx]
                if return_cov:
                    covs = []
                    for i in range(y_nb.shape[0]):
                        Yc = y_nb[i] - np.mean(y_nb[i], axis=0, keepdims=True)
                        C = (Yc.T @ Yc) / max(Yc.shape[0]-1, 1)
                        covs.append(C)
                    out.append((Xq_g_idx, covs))
                else:
                    var = np.var(y_nb, axis=1, ddof=1)
                    out.append((Xq_g_idx, np.sqrt(np.maximum(var, 0.0))))

        # stitch back in original order
        if not vector_target or not return_cov:
            out_arr = np.empty((Xq.shape[0], p if vector_target else 1), dtype=float).squeeze()
            for idxs, vals in out:
                out_arr[idxs] = vals
            return out_arr
        else:
            # list of covs in query order
            covs_full = [None]*Xq.shape[0]
            for idxs, covs in out:
                for j, qi in enumerate(idxs):
                    covs_full[qi] = covs[j]
            return covs_full


def calibrate_k_via_nll(
    X_train, y_train, mu_pred_train, ks=(5,10,15,20,25,40,60),
    **knn_kwargs
):
    """
    Pick k that minimizes Gaussian NLL using a provided mean predictor mu_pred_train.
    Returns best_k, nll_table (dict).
    """
    def gaussian_nll(y, mu, sigma):
        sigma = np.clip(sigma, 1e-6, None)
        return 0.5*np.log(2*np.pi*sigma**2) + 0.5*((y-mu)/sigma)**2

    nlls = {}
    for k in ks:
        sigma = knn_conditional_variance(X_train, y_train, k=k, **knn_kwargs)
        nll = np.mean(gaussian_nll(y_train, mu_pred_train, sigma))
        nlls[k] = float(nll)
    best_k = min(nlls, key=nlls.get)
    return best_k, nlls
