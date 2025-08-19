import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree
from numpy.linalg import inv

def _kernel_weights(r2, kernel='gaussian'):
    if kernel == 'gaussian':
        return np.exp(-0.5*r2)
    elif kernel == 'epanechnikov':
        w = 1.0 - r2
        w[w < 0] = 0.0
        return w
    elif kernel == 'box':
        # r2 <= 1 inside the box radius; weight 1 there
        return (r2 <= 1.0).astype(float)
    else:
        raise ValueError("Unsupported kernel")

def kernel_moment_variance(
    X_train, y_train, X_query=None,
    bandwidth=0.6, kernel='gaussian',
    metric='euclidean', standardize=True, cov=None,
    groups=None, min_points=10, return_cov=False,
    mean_model=None
):
    """
    Option C: Two-stage local moment estimator of Var(Y|X).
      1) m(x)    ~ E[Y|X=x] via local kernel smoother (or provided mean_model)
      2) m2(x)   ~ E[Y^2|X=x] via local kernel smoother
      Var ~ m2 - m^2
    
    Parameters
    ----------
    X_train : (n,d)
    y_train : (n,) or (n,p)
    X_query : (m,d) or None -> if None, estimates at X_train
    bandwidth : float, kernel bandwidth on standardized space (acts like radius)
    kernel    : 'gaussian' | 'epanechnikov' | 'box'
    metric    : 'euclidean' | 'mahalanobis'
    standardize : bool (ignored if metric='mahalanobis')
    cov       : (d,d) covariance for mahalanobis; estimated if None
    groups    : (n,), if provided makes estimates within-group only (for queries==train)
    min_points: ensure at least this many neighbors via kNN fallback
    return_cov: if vector targets and True, return list of (p,p) cov matrices
    mean_model: callable f(X_query)->mu(x) to supply the mean; if None, use local kernel mean
    
    Returns
    -------
    If scalar y: (m,) std estimate
    If vector y and return_cov=False: (m, p) per-component std
    If vector y and return_cov=True : list of (p,p) covariance matrices length m
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    n, d = X_train.shape
    vector_target = (y_train.ndim == 2)
    p = y_train.shape[1] if vector_target else 1

    if X_query is None:
        X_query = X_train
    X_query = np.asarray(X_query)

    # Scaling / metric
    if metric == 'mahalanobis':
        if cov is None:
            cov = np.cov(X_train, rowvar=False)
        VI = inv(cov + 1e-12*np.eye(cov.shape[0]))
        def transform_fn(Z): return Z
        # We'll use Euclidean in whitened space via Cholesky of VI for radius; simpler approach:
        # define distance function via BallTree(metric='mahalanobis') not available -> we use Euclidean with transform
        # For simplicity: build whitening transform from VI
        # Decompose VI = L^T L => whitening: Z @ L
        w, V = np.linalg.eigh(VI)
        L = V @ np.diag(np.sqrt(np.maximum(w, 0))) @ V.T
        def transform_fn(Z): return Z @ L
    else:
        scaler = StandardScaler().fit(X_train) if standardize else None
        def transform_fn(Z): return scaler.transform(Z) if scaler is not None else Z

    Xt = transform_fn(X_train)
    Xq = transform_fn(X_query)
    d_eff = Xt.shape[1]

    tree = BallTree(Xt)  # Euclidean on transformed space

    # radius based on bandwidth
    radius = float(bandwidth) * np.sqrt(d_eff)

    def _estimate_for_indices(idx, xq_vec):
        """
        Compute local moments for one query using neighbors idx (array of indices).
        Fallback: if too few neighbors, use kNN to reach min_points.
        """
        if len(idx) < min_points:
            # kNN fallback
            dist, nn = tree.query(xq_vec[None,:], k=min(min_points, len(Xt)))
            idx = nn[0]
            r2 = dist[0]**2
        else:
            # compute r2 to weight by kernel
            r2 = np.sum((Xt[idx] - xq_vec)**2, axis=1) / (radius**2 if radius > 0 else 1.0)

        w = _kernel_weights(r2, kernel=kernel)
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / (w.sum() + 1e-12)

        Y = y_train[idx]
        if vector_target:
            m = np.sum(w[:,None]*Y, axis=0) if mean_model is None else mean_model(xq_vec[None,:]).reshape(-1)
            m2 = np.sum(w[:,None]*(Y**2), axis=0)
            var = np.maximum(m2 - m**2, 0.0)
            if return_cov:
                Yc = Y - m  # (k,p)
                C = (Yc.T * w) @ Yc / (1.0 - np.sum(w**2) + 1e-12)  # weighted cov (bias-corrected approx)
                return var, C
            else:
                return var, None
        else:
            m = np.sum(w*Y) if mean_model is None else float(mean_model(xq_vec[None,:]).reshape(()))
            m2 = np.sum(w*(Y**2))
            var = max(m2 - m**2, 0.0)
            return var, None

    # Query loop
    if groups is None:
        idx_lists = tree.query_radius(Xq, r=radius)
        if vector_target and return_cov:
            vars_, covs_ = [], []
            for xq, idx in zip(Xq, idx_lists):
                v, C = _estimate_for_indices(idx, xq)
                vars_.append(v)
                covs_.append(C)
            return np.sqrt(np.array(vars_)), covs_
        else:
            vars_ = [ _estimate_for_indices(idx, xq)[0] for xq, idx in zip(Xq, idx_lists) ]
            vars_ = np.array(vars_)
            return np.sqrt(vars_)
    else:
        # Within-group only (requires X_query == X_train)
        groups = np.asarray(groups)
        if Xq.shape[0] != Xt.shape[0] or not np.allclose(Xq, Xt):
            raise ValueError("When 'groups' is provided, X_query must equal X_train for within-group estimation.")
        out_var = np.zeros((Xq.shape[0], p), dtype=float).squeeze()
        out_cov = [None]*Xq.shape[0] if (vector_target and return_cov) else None

        for g in np.unique(groups):
            mask = (groups == g)
            tree_g = BallTree(Xt[mask])
            idx_lists = tree_g.query_radius(Xq[mask], r=radius)
            indices_g = np.where(mask)[0]
            for local_i, (xq, idx_loc) in enumerate(zip(Xq[mask], idx_lists)):
                # map local idx to global
                idx_global = indices_g[idx_loc]
                v, C = _estimate_for_indices(idx_global, xq)
                out_var[indices_g[local_i]] = v
                if out_cov is not None:
                    out_cov[indices_g[local_i]] = C
        if out_cov is not None:
            return np.sqrt(out_var), out_cov
        return np.sqrt(out_var)


def calibrate_bandwidth_via_nll(
    X_train, y_train, mu_pred_train,
    bandwidths=(0.3, 0.45, 0.6, 0.8, 1.0),
    **km_kwargs
):
    """
    Pick bandwidth minimizing Gaussian NLL using baseline mu_pred_train.
    Returns best_h, nll_table (dict).
    """
    def gaussian_nll(y, mu, sigma):
        sigma = np.clip(sigma, 1e-6, None)
        return 0.5*np.log(2*np.pi*sigma**2) + 0.5*((y-mu)/sigma)**2

    nlls = {}
    for h in bandwidths:
        sigma = kernel_moment_variance(
            X_train, y_train, X_query=None, bandwidth=h, **km_kwargs
        )
        nll = np.mean(gaussian_nll(y_train, mu_pred_train, sigma))
        nlls[h] = float(nll)
    best_h = min(nlls, key=nlls.get)
    return best_h, nlls
