from .core import *
from .io import *
def _normalize_gene_sets_arg(gene_sets: str) -> str:
    if not gene_sets: return 'Hallmark'
    alias = {'H':'Hallmark','h':'Hallmark'}
    return alias.get(gene_sets, gene_sets)




def _resolve_gene_sets_for_gseapy(gs: str, organism: str = 'Human') -> str:
    """Resolve user-provided gene_sets to a valid gseapy library name or existing GMT path.
    Heuristics:
      - If path to .gmt exists, use it.
      - Try exact library name (case-insensitive) via gp.get_library_name(organism).
      - If 'Hallmark' or alias was given, pick the first available library containing 'hallmark'.
      - Otherwise return input and let gseapy raise a clean error.
    """
    gs = _normalize_gene_sets_arg(gs)
    p = Path(gs)
    if p.suffix.lower() == '.gmt' and p.exists():
        return str(p)
    if gp is None:
        return gs
    try:
        libs = gp.get_library_name(organism=organism)
        # exact match
        for nm in libs:
            if str(nm).lower() == gs.lower():
                return str(nm)
        # best Hallmark match
        if gs.lower() in {'hallmark','h'}:
            cand = [str(nm) for nm in libs if 'hallmark' in str(nm).lower()]
            if cand:
                return cand[0]
    except Exception:
        pass
    return gs




def run_ssgsea(expr_log: pd.DataFrame, gene_sets: str, outdir: Path, organism: str = 'Human') -> Optional[pd.DataFrame]:
    """Run ssGSEA and return a (pathways x samples) matrix aligned to expr_log columns."""
    if gp is None:
        warnings.warn('gseapy not installed; skipping ssGSEA')
        return None

    outdir.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_gene_sets_for_gseapy(gene_sets, organism=organism)
    
    import traceback
    from .io import log

    try:
        log(f"Attempting to run ssGSEA with gene sets: {resolved}")
        # Newer gseapy uses 'threads'
        res = gp.ssgsea(
            data=expr_log,
            gene_sets=resolved,
            outdir=str(outdir),
            sample_norm_method=None,
            min_size=5,
            max_size=10000,
            scale=True,
            threads=1,
            seed=42,
            verbose=False,
        )
        # Expect res.res2d to be a DataFrame (samples x pathways) or (pathways x samples)
        if not hasattr(res, 'res2d') or not isinstance(res.res2d, pd.DataFrame):
            log("WARNING: ssGSEA returned no res2d table; skipping results")
            warnings.warn('ssGSEA returned no res2d table; skipping results')
            return None

        df = res.res2d.copy()
        df.columns = df.columns.astype(str)
        df.index = df.index.astype(str)
        z_index = expr_log.columns.astype(str)

        # Align orientation: choose the one that overlaps your sample IDs
        col_overlap = len(set(z_index) & set(df.columns))
        idx_overlap = len(set(z_index) & set(df.index))
        log(f"Sample alignment check: {col_overlap} columns overlap, {idx_overlap} index overlap")
        
        if col_overlap >= max(2, len(z_index)//4):
            ss = df  # samples are columns already
            log(f"Using columns as samples (shape: {ss.shape})")
        elif idx_overlap >= max(2, len(z_index)//4):
            ss = df.T  # samples are rows; transpose
            log(f"Transposed to use index as samples (shape: {ss.shape})")
        else:
            log(f"WARNING: Could not align ssGSEA output. Column overlap: {col_overlap}/{len(z_index)}, Index overlap: {idx_overlap}/{len(z_index)}")
            warnings.warn('Could not align ssGSEA output to samples; skipping ssGSEA results')
            return None

        # Keep only samples present in pseudotime/expression, preserve order
        common = [s for s in z_index if s in ss.columns]
        if len(common) < 2:
            log(f"WARNING: Only {len(common)} common samples found after alignment")
            warnings.warn(f'Only {len(common)} common samples found after alignment')
            return None
        ss = ss.loc[:, common]
        log(f"Final ssGSEA matrix: {ss.shape[0]} pathways × {ss.shape[1]} samples")
        return ss

    except Exception as e:
        error_msg = f'ssGSEA failed: {e}'
        log(f"ERROR: {error_msg}")
        log(f"Traceback: {traceback.format_exc()}")
        warnings.warn(error_msg)
        return None






def pathway_temporal_summary(ss: pd.DataFrame, z: pd.Series) -> pd.DataFrame:
    # align by sample IDs; auto-transpose if needed
    if ss is None or ss.empty:
        return pd.DataFrame(columns=['pathway','rho','amp','score']).set_index('pathway')
    ss = ss.copy()
    ss.columns = ss.columns.astype(str); ss.index = ss.index.astype(str)
    z = z.copy(); z.index = z.index.astype(str)
    cols_common = ss.columns.intersection(z.index)
    idx_common = ss.index.intersection(z.index)
    if len(cols_common) == 0 and len(idx_common) > 0:
        ss = ss.T
        cols_common = ss.columns.intersection(z.index)
    if len(cols_common) < 2:
        warnings.warn('ssGSEA produced no overlapping samples with pseudotime; skipping pathway summary.')
        return pd.DataFrame(columns=['pathway','rho','amp','score']).set_index('pathway')
    ss = ss.loc[:, cols_common]
    z = z.loc[cols_common]
    rows = []
    for pw, vec in ss.iterrows():
        v = vec.values
        if v.size == 0 or np.all(~np.isfinite(v)):
            continue
        rho = pd.Series(v, index=cols_common).corr(z, method='spearman')
        vmin = np.nanmin(v) if np.isfinite(v).any() else np.nan
        vmax = np.nanmax(v) if np.isfinite(v).any() else np.nan
        amp = float(vmax - vmin) if np.all(np.isfinite([vmin, vmax])) else np.nan
        score = abs((rho if pd.notna(rho) else 0.0) * (amp if pd.notna(amp) else 0.0))
        rows.append((pw, rho, amp, score))
    out = pd.DataFrame(rows, columns=['pathway','rho','amp','score']).set_index('pathway')
    if not out.empty:
        out = out.sort_values('score', ascending=False)
    return out




def ssgsea_report_to_matrix(report_csv: Path) -> Optional[pd.DataFrame]:
    """
    Convert a gseapy ssGSEA long-form report (Name, Term, ES, NES)
    into a pathways x samples matrix using NES values.
    """
    try:
        df = read_table_auto(report_csv)
    except Exception:
        try:
            df = pd.read_csv(report_csv)
        except Exception as e:
            warnings.warn(f"Could not read ssGSEA report: {e}"); return None

    cols = {c.lower(): c for c in df.columns}
    if not {'name','term'}.issubset(cols.keys()) or ('nes' not in cols and 'es' not in cols):
        warnings.warn("ssGSEA report missing required columns (need Name, Term and NES/ES).")
        return None

    sample_col = cols['name']; path_col = cols['term']
    value_col  = cols.get('nes', cols.get('es'))

    df = df[[sample_col, path_col, value_col]].copy()
    df[sample_col] = df[sample_col].astype(str)
    df[path_col]   = df[path_col].astype(str)

    try:
        M = df.pivot_table(index=path_col, columns=sample_col, values=value_col, aggfunc='mean')
        M.index = M.index.astype(str); M.columns = M.columns.astype(str)
        return M
    except Exception as e:
        warnings.warn(f"Failed to pivot ssGSEA report: {e}")
        return None




def find_ssgsea_report_in(out_dir: Path) -> Optional[Path]:
    """Search typical locations for gseapy ssGSEA report within out_dir."""
    ssg_dir = out_dir / 'ssgsea'
    candidates = []
    # canonical name
    cand = ssg_dir / 'gseapy.gene_set.ssgsea.report.csv'
    if cand.exists(): return cand
    # any gseapy.*ssgsea.report.csv in ssgsea/
    if ssg_dir.exists():
        for p in ssg_dir.glob('gseapy.*ssgsea.report.csv'):
            candidates.append(p)
    return candidates[0] if candidates else None


# -------------------------------
# plotting helpers (matplotlib only)
# -------------------------------

