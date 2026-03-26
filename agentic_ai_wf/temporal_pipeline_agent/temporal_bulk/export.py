from __future__ import annotations
from .core import *
from .io import log
from .deconv import _infer_sample_id_column
from .plotting import per_entity_pngs
def export_trajectory_tables(z: pd.Series, ss: Optional[pd.DataFrame], deconv: Optional[pd.DataFrame], out_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    # Build pathway long table
    pw_long = None
    if ss is not None and not ss.empty:
        common = ss.columns.intersection(z.index)
        if len(common) > 0:
            zz = z.loc[common]
            pw_long = ss.loc[:, common].copy()
            pw_long['entity'] = pw_long.index
            pw_long = pw_long.melt(id_vars=['entity'], var_name='sample_id', value_name='score')
            pw_long['t'] = pw_long['sample_id'].map(zz)
            tmin, tmax = float(zz.min()), float(zz.max())
            pw_long['t_norm'] = (pw_long['t'] - tmin) * 100.0 / (tmax - tmin + 1e-12)
            pw_long[['entity','t','score','t_norm']].to_csv(out_dir / 'pathway_profiles.csv', index=False)

    # Build cellmix long table from deconv proportions if available
    cm_long = None
    if deconv is not None and not deconv.empty:
        df = deconv.copy()
        df.columns = df.columns.astype(str)
        z_index = z.index.astype(str)

    # --- NEW: auto-detect orientation ---
    # Case 1: samples are COLUMNS (your file); first column is usually 'cell_type'
        col_overlap = len(set(df.columns) & set(z_index))
        if col_overlap >= max(3, len(z_index) // 6):
        # If there's a left-most label column (e.g., 'cell_type'), use it as index before transpose
            if df.columns[0].lower() in {"cell_type", "celltype", "type"}:
                df = df.set_index(df.columns[0])
            df = df.T  # transpose → rows=samples, cols=cell types
            df.index.name = 'sample_id'
            df.reset_index(inplace=True)

    # Case 2: samples are already ROWS, but sample ids might be in the index or a nonstandard column
        if 'sample_id' not in df.columns:
        # If the row index already looks like sample IDs, promote it
            if len(set(df.index.astype(str)) & set(z_index)) >= max(3, len(z_index) // 6):
                df = df.reset_index().rename(columns={'index': 'sample_id'})
            else:
                sid = _infer_sample_id_column(df, z.index)
                if sid not in df.columns:
                    sid = df.columns[0]
                df = df.rename(columns={sid: 'sample_id'})

    # Align and melt to tidy long
        df = df.set_index('sample_id', drop=False)
        df.index = df.index.astype(str)
        common = df.index.intersection(z_index)

        if len(common) < 3:
            warnings.warn(f'Cellmix: only {len(common)} overlapping samples with pseudotime; skipping cellmix export.')
        else:
            X = df.loc[common].drop(columns=['sample_id'], errors='ignore')
            long = X.reset_index().rename(columns={'index': 'sample_id'})
            cm_long = long.melt(id_vars=['sample_id'], var_name='entity', value_name='score')
            cm_long['t'] = cm_long['sample_id'].map(z.astype(float))
            tmin, tmax = float(z.loc[common].min()), float(z.loc[common].max())
            cm_long['t_norm'] = (cm_long['t'] - tmin) * 100.0 / (tmax - tmin + 1e-12)
            out_csv = out_dir / 'cellmix_profiles.csv'
            cm_long[['entity', 't', 'score', 't_norm']].to_csv(out_csv, index=False)
            log(f'Wrote {out_csv.name}: {cm_long.shape[0]} rows, '
                f'{cm_long["entity"].nunique()} entities, {len(common)} samples aligned.')

    return pw_long, cm_long





def build_causal_pack(out_dir: Path, pw_long: Optional[pd.DataFrame], cm_long: Optional[pd.DataFrame], top_n: int) -> Path:
    pack_dir = out_dir / 'temporal_pack'; plots_dir = pack_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    created = []
    # per-entity PNGs
    if pw_long is not None and not pw_long.empty:
        created += per_entity_pngs(pw_long, plots_dir, 'pathway', top_n)
    if cm_long is not None and not cm_long.empty:
        created += per_entity_pngs(cm_long, plots_dir, 'cellmix', top_n)
    # manifest
    manifest = {
        'axis': 'pseudotime',
        'files': {
            'pathway_profiles_csv': str((out_dir / 'pathway_profiles.csv')) if (out_dir / 'pathway_profiles.csv').exists() else None,
            'cellmix_profiles_csv': str((out_dir / 'cellmix_profiles.csv')) if (out_dir / 'cellmix_profiles.csv').exists() else None,
            'plots': created,
        }
    }
    (pack_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return pack_dir





def _safe_read_csv(path: Path, sep: str | None = None):
    try:
        if not path.exists(): return None
        return pd.read_csv(path, sep=sep) if sep is not None else pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep='\t')
        except Exception:
            return None
