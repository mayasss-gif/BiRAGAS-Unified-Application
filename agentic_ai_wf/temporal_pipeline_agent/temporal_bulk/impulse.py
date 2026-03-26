from .core import *
def impulse_fn(t: np.ndarray, b0, b1, k1, t1, b2, k2, t2):
    return b0 + b1/(1.0 + np.exp(-k1*(t - t1))) - b2/(1.0 + np.exp(-k2*(t - t2)))




def initial_guess(t: np.ndarray, y: np.ndarray) -> Tuple[float,...]:
    b0 = float(np.nanmedian(y))
    ymin, ymax = np.nanmin(y), np.nanmax(y); amp = float(max(1e-6, ymax - ymin))
    b1 = amp/2.0; b2 = amp/2.0
    tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
    t1 = tmin + 0.33*(tmax - tmin); t2 = tmin + 0.66*(tmax - tmin)
    k1 = 5.0/max(1e-6, (tmax - tmin)); k2 = 5.0/max(1e-6, (tmax - tmin))
    return (b0, b1, k1, t1, b2, k2, t2)




def fit_impulse_single(gene: str, t: np.ndarray, y: np.ndarray, max_iter: int) -> Tuple[str, dict, dict, dict]:
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask].astype(float); y = y[mask].astype(float)
    n = t.size
    if n < 4 or curve_fit is None:
        return gene, {'ok': False}, {'ok': False}, {'autocorr_lag1': np.nan, 'r2_impulse': np.nan, 'r2_linear': np.nan, 'cooks_max': np.nan}
    idx = np.argsort(t); t = t[idx]; y = y[idx]
    # constant
    ybar = float(np.nanmean(y)); rss0 = float(np.nansum((y - ybar)**2)); k0 = 1
    # linear
    X = np.column_stack([np.ones_like(t), t])
    beta_lin, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat_lin = X @ beta_lin; rss1 = float(np.nansum((y - yhat_lin)**2)); k1 = 2
    # impulse
    p0 = initial_guess(t, y)
    bounds = ([-np.inf, 0.0, 1e-6, t.min()-1.0, 0.0, 1e-6, t.min()-1.0], [np.inf, np.inf, 100.0, t.max()+1.0, np.inf, 100.0, t.max()+1.0])
    try:
        popt, _ = curve_fit(impulse_fn, t, y, p0=p0, bounds=bounds, maxfev=max_iter)
        yhat_imp = impulse_fn(t, *popt); rss2 = float(np.nansum((y - yhat_imp)**2)); k2 = 7; ok = True
    except Exception:
        popt, yhat_imp, rss2, k2, ok = (None, None, np.inf, 0, False)
    def aic(rss, k, n): return n*np.log(rss/n + 1e-12) + 2*k
    a1 = aic(rss1, k1, n); a2 = aic(rss2, k2, n)
    if np.isfinite(rss2) and rss2 < rss1 and chi2 is not None:
        lr = n*np.log((rss1+1e-12)/(rss2+1e-12))
        p_imp_vs_lin = float(1.0 - chi2.cdf(max(0.0, 2*lr), df=max(1, k2-k1)))
    else:
        p_imp_vs_lin = 1.0
    # QC
    try: ac1 = float(pd.Series(y).autocorr(lag=1))
    except Exception: ac1 = np.nan
    def r2(y, yhat):
        if yhat is None: return np.nan
        ssr = np.nansum((y - yhat)**2); sst = np.nansum((y - np.nanmean(y))**2) + 1e-12
        return 1.0 - ssr/sst
    r2_imp = r2(y, yhat_imp); r2_lin = r2(y, yhat_lin)
    cooks = np.nan
    try:
        if sm is not None:
            Xc = sm.add_constant(t); fit = sm.OLS(y, Xc).fit(); infl = OLSInfluence(fit); cooks = float(np.nanmax(infl.cooks_distance[0]))
    except Exception: pass
    imp = {'ok': ok, 'params': popt.tolist() if ok else [], 'rss': rss2, 'aic': a2, 'p_vs_linear': p_imp_vs_lin}
    lin = {'ok': True, 'rss': rss1, 'aic': a1}
    qc = {'autocorr_lag1': ac1, 'r2_impulse': r2_imp, 'r2_linear': r2_lin, 'cooks_max': cooks}
    return gene, imp, lin, qc

# -------------------------------
# ssGSEA & pathway summary
# -------------------------------

