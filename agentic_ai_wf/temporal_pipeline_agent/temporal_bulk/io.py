from .core import *
def log(msg: str):
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")
    sys.stdout.flush()




def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Data IO
# -------------------------------



def read_table_auto(path: Path) -> pd.DataFrame:
    sep = '	' if path.suffix.lower() in {'.tsv', '.tab'} else ','
    try:
        return pd.read_csv(path, sep=sep)
    except Exception:
        return pd.read_csv(path)




def read_counts(counts_path: Path, gene_col: str) -> pd.DataFrame:
    df = read_table_auto(counts_path)
    # Try to locate gene id column
    candidates = [gene_col] + [c for c in df.columns if c.lower() in {'gene','genes','gene_id','symbol','ensembl','ensembl_id'}]
    hit = None
    for c in candidates:
        if c in df.columns: hit = c; break
        # case-insensitive match
        for col in df.columns:
            if col.lower() == c.lower(): hit = col; break
        if hit: break
    if hit is None:
        raise ValueError(f"Gene column not found. Tried: {candidates} in {counts_path}")
    df = df.copy(); df.index = df[hit].astype(str); df.drop(columns=[hit], inplace=True)
    # coerce to numeric, missing to NaN then 0 later
    df = df.apply(pd.to_numeric, errors='coerce')
    return df




def read_metadata(meta_path: Path) -> pd.DataFrame:
    meta = read_table_auto(meta_path)
    if 'sample_id' not in meta.columns:
        cands = [c for c in meta.columns if c.lower() in {'sample','sampleid','id','sample_id','run','srx','srr'}]
        if not cands:
            # if first column looks like IDs, use it
            cands = [meta.columns[0]]
        meta = meta.rename(columns={cands[0]:'sample_id'})
    meta['sample_id'] = meta['sample_id'].astype(str)
    meta = meta.set_index('sample_id', drop=False)
    return meta




def harmonize(expr: pd.DataFrame, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = expr.columns.intersection(meta.index)
    if len(common)==0 and expr.index.intersection(meta.index).size>0:
        expr = expr.T; common = expr.columns.intersection(meta.index)
    if len(common)==0:
        raise ValueError('No overlap between counts COLNAMES and metadata rownames after attempts.')
    expr = expr.loc[:, common]; meta = meta.loc[common, :]
    return expr, meta

# -------------------------------
# transforms & pseudotime
# -------------------------------

