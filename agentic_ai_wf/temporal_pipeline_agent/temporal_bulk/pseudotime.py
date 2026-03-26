from .core import *

def pseudotime_pca(expr_log: pd.DataFrame, seed: int = 42) -> pd.Series:
    if PCA is None:
        warnings.warn('sklearn not available; returning zero pseudotime')
        return pd.Series(0.0, index=expr_log.columns, name='pseudotime')
    X = expr_log.T.values
    if StandardScaler is not None:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    pca = PCA(n_components=3, random_state=seed)
    Z = pca.fit_transform(X)
    z = pd.Series(Z[:,0], index=expr_log.columns, name='pseudotime')
    z = (z - z.min())/(z.max()-z.min()+1e-12)
    return z




def try_phenopath(expr_log: pd.DataFrame, meta: pd.DataFrame, covariate: str, thin: int, maxiter: int) -> Tuple[pd.Series, Optional[np.ndarray]]:
    """Return (pseudotime_series, elbo_vector_or_None)."""
    if not R_AVAILABLE:
        warnings.warn('rpy2/R not available; using PCA pseudotime')
        return pseudotime_pca(expr_log), None
    try:
        phenopath = importr('phenopath'); base = importr('base')
    except Exception as e:
        warnings.warn(f'phenopath import failed: {e}; using PCA pseudotime')
        return pseudotime_pca(expr_log), None
    if covariate and covariate in meta.columns:
        x_vec = meta[covariate].loc[expr_log.columns]
    else:
        x_vec = pd.Series(1.0, index=expr_log.columns)
    r_expr = pandas2ri.py2rpy(expr_log)
    r_x = pandas2ri.py2rpy(pd.Series(x_vec))
    try:
        fit = phenopath.phenopath(Y=r_expr, x=r_x, covariates=ro.NULL, thin=thin, maxiter=maxiter)
        z = np.array(base.getattr(fit, 'z')).flatten()
        # ELBO (best effort)
        elbo = None
        for slot in ['elbo','ELBO','elbo_hist','ELBO_hist']:
            try:
                elbo = np.array(base.getattr(fit, slot)).flatten(); break
            except Exception:
                continue
        z = pd.Series(z, index=expr_log.columns, name='pseudotime')
        z = (z - z.min())/(z.max()-z.min()+1e-12)
        return z, elbo
    except Exception as e:
        warnings.warn(f'phenopath run failed: {e}; using PCA pseudotime')
        return pseudotime_pca(expr_log), None

# -------------------------------
# impulse & helpers
# -------------------------------

