from .core import *
def cpm_log1p(expr_counts: pd.DataFrame) -> pd.DataFrame:
    lib = expr_counts.sum(axis=0).replace(0, np.nan)
    cpm = expr_counts.divide(lib, axis=1) * 1e6
    cpm = cpm.fillna(0.0)
    return np.log1p(cpm)


