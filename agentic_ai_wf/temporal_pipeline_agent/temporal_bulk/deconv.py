from .core import *
def _infer_sample_id_column(df: pd.DataFrame, z_index: pd.Index) -> str:
    """Heuristically determine the sample id column in deconv table."""
    # perfect match
    for c in df.columns:
        if df[c].astype(str).isin(z_index).all():
            return c
    # best overlap among string-like columns
    best_c, best_hit = None, -1
    for c in df.columns:
        # skip numeric-only columns
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        hits = int(df[c].astype(str).isin(z_index).sum())
        if hits > best_hit:
            best_hit, best_c = hits, c
    if best_c is not None and best_hit > 0:
        return best_c
    # fallback: first column
    return df.columns[0]


