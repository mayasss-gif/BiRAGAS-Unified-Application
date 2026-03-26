from .core import *
def _img_tag_if_exists(p: Path, alt: str, w: str = "100%") -> str:
    return f'<img src="{p.name}" alt="{alt}" style="max-width:{w};height:auto;border:1px solid #eee;border-radius:8px;margin:8px 0;">' if p.exists() else f'<div style="color:#999">(missing: {p.name})</div>'


def _normalize_gene_id_column(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Ensure a 'gene_id' column exists; recover it from index or common dump columns."""
    if df is None or df.empty:
        return df
    d = df.copy()
    # If 'gene_id' already a column, done.
    if 'gene_id' in d.columns:
        return d
    # If the index is named like gene_id/gene/symbol, bring it out.
    if d.index.name and str(d.index.name).lower() in {'gene_id', 'gene', 'symbol'}:
        d = d.reset_index().rename(columns={d.index.name: 'gene_id'})
        return d
    # If pandas wrote the index as a plain 'index' or 'Unnamed: 0' column upstream, fix it.
    if 'index' in d.columns:
        d = d.rename(columns={'index': 'gene_id'})
        return d
    if 'Unnamed: 0' in d.columns:
        d = d.rename(columns={'Unnamed: 0': 'gene_id'})
        return d
    # Fallback: treat the first column as gene_id (last resort, but prevents Nones in HTML)
    first = d.columns[0]
    d = d.rename(columns={first: 'gene_id'})
    return d




