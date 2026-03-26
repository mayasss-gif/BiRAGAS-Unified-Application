from .core import *
def save_boxplot_pseudotime(z: pd.Series, meta: pd.DataFrame, covariate: str, out: Path):
    if not covariate or covariate not in meta.columns: return
    levels = [lv for lv in pd.Categorical(meta[covariate]).categories if pd.notna(lv)]
    if len(levels) < 1: return
    groups = [z.loc[meta.index[meta[covariate]==level]] for level in levels]
    labels = list(map(str, levels))
    fig, ax = plt.subplots(figsize=(8,5))
    # Matplotlib ≥3.9 uses tick_labels
    try:
        ax.boxplot(groups, tick_labels=labels)
    except TypeError:
        ax.boxplot(groups, labels=labels)
    ax.set_xlabel(covariate); ax.set_ylabel('Pseudotime')
    ax.set_title('Pseudotime by ' + covariate)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)




def save_bar_top_pathways(pw_summary: pd.DataFrame, out: Path, top: int = 15):
    if pw_summary is None or pw_summary.empty: return
    topdf = pw_summary.reindex(pw_summary['rho'].abs().sort_values(ascending=False).head(top).index)
    fig, ax = plt.subplots(figsize=(10,6))
    y = np.arange(len(topdf))
    ax.barh(y, topdf['rho'].values)
    ax.set_yticks(y); ax.set_yticklabels(topdf.index)
    ax.set_xlabel('Spearman rho'); ax.set_title('Pathway activity vs pseudotime')
    fig.tight_layout(); fig.savefig(out); plt.close(fig)




def grid_plot_trajectories(df_long: pd.DataFrame, title: str, out: Path, top_n: int = 12):
    if df_long is None or df_long.empty: return
    amp = df_long.groupby('entity')['score'].agg(lambda x: float(np.nanmax(x)-np.nanmin(x)))
    keep = amp.sort_values(ascending=False).head(top_n).index
    sub = df_long[df_long['entity'].isin(keep)]
    entities = list(keep)
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16,10))
    axes = axes.flatten()
    for i, ent in enumerate(entities):
        ax = axes[i]
        d = sub[sub['entity']==ent].sort_values('t_norm')
        ax.plot(d['t_norm'].values, d['score'].values)
        ax.set_title(str(ent), fontsize=10)
        ax.set_xlabel('Temporal axis (0–100)'); ax.set_ylabel('Score')
    for j in range(len(entities), rows*cols):
        fig.delaxes(axes[j])
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(out); plt.close(fig)




def per_entity_pngs(df_long: pd.DataFrame, out_dir: Path, prefix: str, top_n: int = 12) -> List[str]:
    if df_long is None or df_long.empty: return []
    amp = df_long.groupby('entity')['score'].agg(lambda x: float(np.nanmax(x)-np.nanmin(x)))
    keep = amp.sort_values(ascending=False).head(top_n).index
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ent in keep:
        d = df_long[df_long['entity']==ent].sort_values('t_norm')
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(d['t_norm'].values, d['score'].values)
        ax.set_title(str(ent)); ax.set_xlabel('Temporal axis (0–100)'); ax.set_ylabel('Score')
        fig.tight_layout()
        fn = out_dir / f"{prefix}_{re.sub(r'[^A-Za-z0-9._-]+','_',str(ent))[:80]}.png"
        fig.savefig(fn); plt.close(fig)
        paths.append(str(fn))
    return paths

# -------------------------------
# DE vs interaction scatter (approximate limma analog)
# -------------------------------



def de_and_interaction_scatter(expr_log: pd.DataFrame, meta: pd.DataFrame, z: pd.Series, covariate: str, treatment_level: str, extras: List[str], out_png: Path, out_csv: Path):
    if smf is None or not covariate or covariate not in meta.columns: return
    # levels
    cat = pd.Categorical(meta[covariate])
    levels = list(cat.categories)
    if len(levels) < 2: return
    trt = treatment_level if (treatment_level in levels) else levels[-1]
    Xbase = meta.copy(); Xbase['z'] = z
    rows = []
    for gene, row in expr_log.iterrows():
        df = pd.concat([Xbase, row.rename('y')], axis=1)
        # DE main effect
        try:
            rhs = "C(%s)" % covariate
            if extras: rhs += " + " + " + ".join(extras)
            fit_de = smf.ols(f"y ~ {rhs}", data=df).fit()
            coef_name = f"C({covariate})[T.{trt}]"
            p_de = float(fit_de.pvalues.get(coef_name, np.nan))
        except Exception:
            p_de = np.nan
        # Interaction with pseudotime
        try:
            rhs = f"z * C({covariate})"
            if extras: rhs += " + " + " + ".join(extras)
            fit_int = smf.ols(f"y ~ {rhs}", data=df).fit()
            coef_int = f"z:C({covariate})[T.{trt}]"
            beta = float(fit_int.params.get(coef_int, np.nan))
        except Exception:
            beta = np.nan
        rows.append((gene, p_de, beta))
    out = pd.DataFrame(rows, columns=['gene','p_de','beta_interaction']).dropna()
    out['neglog10_p'] = -np.log10(out['p_de'].clip(lower=1e-300))
    out.sort_values('neglog10_p', ascending=False).to_csv(out_csv, index=False)
    # plot
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(out['neglog10_p'].values, out['beta_interaction'].values, s=18)
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_xlabel('-log10 p (DE)'); ax.set_ylabel('Interaction effect (β)')
    ax.set_title(f'DE vs Interaction: {covariate} (coef: {trt})')
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

# -------------------------------
# Trajectory export (causal interface)
# -------------------------------



def _rank_genes_for_gallery(fits: pd.DataFrame, by: str, top_n: int) -> List[str]:
    df = fits.copy()
    if 'gene_id' in df.columns:
        df = df.set_index('gene_id')
    if 'p_adj' not in df.columns and 'padj' in df.columns:
        df = df.rename(columns={'padj':'p_adj'})
    df = df[['p_adj','r2_impulse']].dropna()
    if df.empty:
        return []
    if by == 'fdr':
        df = df.sort_values('p_adj', ascending=True)
    else:
        df = df.sort_values('r2_impulse', descending=False)  # ← if you used ascending=False originally, keep that
    return df.head(max(1, top_n)).index.tolist()



def _smooth_trend(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 6:
        xs = np.sort(x[mask])
        return xs, y[mask][np.argsort(x[mask])]
    xx = x[mask]; yy = y[mask]
    coefs = np.polyfit(xx, yy, 3)
    xs = np.linspace(xx.min(), xx.max(), 200)
    ys = np.polyval(coefs, xs)
    return xs, ys



def plot_gene_trajectory(gene: str, expr_log: pd.DataFrame, z: pd.Series,
                         out_png: Path, color_by: Optional[pd.Series] = None):
    if gene not in expr_log.index:
        return
    y = expr_log.loc[gene, z.index].values.astype(float)
    x = z.values.astype(float)
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=160)
    if color_by is not None and len(color_by.unique()) <= 10:
        cats = color_by.loc[z.index].astype(str)
        for lvl in sorted(cats.unique()):
            m = (cats == lvl).values
            ax.scatter(x[m], y[m], s=18, alpha=0.7, label=str(lvl))
        ax.legend(frameon=False, fontsize=8, loc='best')
    else:
        ax.scatter(x, y, s=18, alpha=0.7)
    xs, ys = _smooth_trend(x, y)
    ax.plot(xs, ys, linewidth=2)
    ax.set_title(gene); ax.set_xlabel('pseudotime (0–1)'); ax.set_ylabel('expression (log-CPM)')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


# -------------------------------
#               main
# -------------------------------

