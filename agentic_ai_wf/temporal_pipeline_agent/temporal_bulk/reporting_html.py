from .core import *
from .reporting import *
from .reporting import _img_tag_if_exists, _normalize_gene_id_column

def write_detailed_html_report(
    out_dir: Path,
    counts_path: Path,
    meta_path: Path,
    expr_counts: pd.DataFrame,
    z: pd.Series,
    fits_df: pd.DataFrame,
    pw_summary: Optional[pd.DataFrame],
    pw_long: Optional[pd.DataFrame],
    cm_long: Optional[pd.DataFrame],
    used_gene_sets: Optional[str],
    made_elbo: bool,
    made_de_interaction: bool,
):
    """Create a rich, self-explanatory HTML: report_detailed.html."""
    # Resolve key paths
    pseudotime_csv = out_dir / "pseudotime.csv"
    fits_tsv = out_dir / "temporal_gene_fits.tsv"
    qc_tsv = out_dir / "temporal_qc.tsv"
    pw_sum_csv = out_dir / "pathway_temporal_summary.csv"
    pw_prof_csv = out_dir / "pathway_profiles.csv"
    cm_prof_csv = out_dir / "cellmix_profiles.csv"
    ssgsea_dir = out_dir / "ssgsea"
    gseapy_report = ssgsea_dir / "gseapy.gene_set.ssgsea.report.csv"
    gene_sets_gmt = ssgsea_dir / "gene_sets.gmt"
    manifest = out_dir / "temporal_pack" / "manifest.json"

    fig_cov = out_dir / "pseudotime_by_covariate.png"
    fig_pwbar = out_dir / "pseudotime_vs_pathways_top15.png"
    fig_pwgrid = out_dir / "temporal_pathways_top12.png"
    fig_cmgrid = out_dir / "temporal_cellmix_top12.png"
    fig_elbo = out_dir / "elbo.png"
    fig_debeta = out_dir / "de_vs_beta.png"
    fig_genegrid = out_dir / "temporal_genes_topN.png"



    # Normalize fits so a 'gene_id' column always exists for the "Top genes" sections
    if fits_df is not None:
        fits_df = _normalize_gene_id_column(fits_df)


    # Basic metrics
    n_samples = int(z.shape[0]) if z is not None else None
    n_genes = int(expr_counts.shape[0]) if expr_counts is not None else None
    n_fit = int(fits_df.shape[0]) if fits_df is not None else None
    n_sig = None
    patt_counts = {}
    if fits_df is not None:
        if 'p_adj' in fits_df.columns:
            n_sig = int((fits_df['p_adj'].astype(float) < 0.05).sum())
        if 'pattern' in fits_df.columns:
            patt_counts = fits_df['pattern'].value_counts().to_dict()

    # Top items
    def _top_pathways(df: Optional[pd.DataFrame], k=5) -> list[tuple[str, float, float, float]]:
        if df is None or df.empty or 'rho' not in df.columns:
            return []
        d = df.copy()
        if 'pathway' not in d.columns:
            d = d.reset_index().rename(columns={'index':'pathway'})
        order = d['rho'].abs().sort_values(ascending=False).head(k).index
        d = d.loc[order, ['pathway','rho','amp','score']]
        out = []
        for _, r in d.iterrows():
            out.append((str(r['pathway']), float(r['rho']), float(r['amp']), float(r['score'])))
        return out

    def _top_genes_q(df: Optional[pd.DataFrame], k=5) -> list[dict]:
        if df is None or df.empty or 'p_adj' not in df.columns:
            return []
        d = df.sort_values('p_adj', ascending=True).head(k)
        cols = ['gene_id','p_adj','pattern','r2_impulse','time_of_peak','time_of_valley']
        cols = [c for c in cols if c in d.columns]
        return d[cols].to_dict('records')

    def _top_genes_r2(df: Optional[pd.DataFrame], k=5) -> list[dict]:
        if df is None or df.empty:
            return []
        r2c = next((c for c in df.columns if c.lower()=='r2_impulse'), None)
        if r2c is None: return []
        pcol = 'p_adj' if 'p_adj' in df.columns else ('p_value' if 'p_value' in df.columns else None)
        d = df.sort_values(r2c, ascending=False).head(k)
        want = ['gene_id', r2c, 'pattern', 'time_of_peak','time_of_valley']
        if pcol: want.append(pcol)
        want = [c for c in want if c in d.columns]
        return d[want].to_dict('records')

    def _top_cells(cm: Optional[pd.DataFrame], k=5) -> list[tuple[str, float]]:
        if cm is None or cm.empty or not {'entity','score'}.issubset(cm.columns):
            return []
        amp = cm.groupby('entity')['score'].agg(lambda x: float(np.nanmax(x)-np.nanmin(x))).sort_values(ascending=False)
        ents = list(amp.head(k).index)
        return [(e, float(amp[e])) for e in ents]

    tops_pw = _top_pathways(pw_summary, 5)
    tops_gq = _top_genes_q(fits_df, 5)
    tops_gr2 = _top_genes_r2(fits_df, 5)
    tops_cells = _top_cells(cm_long, 5)

    patt_str = ", ".join([f"{k}: {v}" for k, v in sorted(patt_counts.items(), key=lambda kv: -kv[1])]) if patt_counts else "(n/a)"
    gs_meta = used_gene_sets if used_gene_sets else ("from_report" if (pw_summary is not None and not pw_summary.empty) else None)

    # Simple presence helper
    def _exists(p: Path) -> str:
        return "present" if p.exists() else "missing"

    # HTML (styling kept modest for portability)
    css = """
    <style>
    body{font-family:ui-sans-serif,system-ui,Arial; margin:24px; line-height:1.45}
    h1,h2,h3{margin-top:18px}
    code,kbd{background:#f7f7f9;border:1px solid #e1e1e8;border-radius:4px;padding:1px 4px}
    .card{border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin:12px 0;background:#fff}
    .muted{color:#6b7280}
    ul{margin:0 0 0 20px}
    table{border-collapse:collapse;width:100%;margin:8px 0}
    th,td{border:1px solid #e5e7eb;padding:6px;text-align:left}
    .pill{display:inline-block;background:#eef2ff;color:#3730a3;border-radius:999px;padding:2px 8px;margin-left:6px;font-size:12px}
    .good{color:#065f46}.warn{color:#92400e}.bad{color:#7f1d1d}
    .k{color:#111827}
    </style>
    """

    def _li_link(p: Path, label: str) -> str:
        return f'<li><a href="{p.name}">{label}</a> <span class="pill { "good" if p.exists() else "bad"}">{_exists(p)}</span></li>'

    dir_map = f"""
    <div class="card">
      <h2>Directory map (what’s what)</h2>
      <ul>
        {_li_link(pseudotime_csv, "pseudotime.csv — sample-level temporal axis")}
        {_li_link(fits_tsv, "temporal_gene_fits.tsv — per-gene impulse fits + stats")}
        {_li_link(qc_tsv, "temporal_qc.tsv — per-gene quality metrics")}
        <li>ssgsea/:
          <ul>
            {_li_link(gene_sets_gmt, "gene_sets.gmt — gene set library used")}
            {_li_link(gseapy_report, "gseapy.gene_set.ssgsea.report.csv — long-form pathway × sample scores")}
          </ul>
        </li>
        {_li_link(pw_sum_csv, "pathway_temporal_summary.csv — per-pathway correlation vs time + amplitude")}
        {_li_link(pw_prof_csv, "pathway_profiles.csv — tidy pathway trajectories")}
        {_li_link(cm_prof_csv, "cellmix_profiles.csv — tidy cell-type trajectories")}
        {_li_link(fig_cov, "pseudotime_by_covariate.png")}
        {_li_link(fig_pwbar, "pseudotime_vs_pathways_top15.png")}
        {_li_link(fig_pwgrid, "temporal_pathways_top12.png")}
        {_li_link(fig_cmgrid, "temporal_cellmix_top12.png")}
        {_li_link(fig_genegrid, "temporal_genes_topN.png — grid of top gene trajectories")}
        <li>genes/ — individual per-gene PNGs (present if you used <code>--save_gene_plots</code>)</li>
        <li>temporal_pack/manifest.json <span class="pill { "good" if manifest.exists() else "bad"}">{_exists(manifest)}</span></li>
        <li>report.html <span class="pill good">present</span></li>
        <li>report_detailed.html <span class="pill good">present</span></li>
      </ul>
    </div>
    """

    # Quick metrics
    quick = f"""
    <div class="card">
      <h2>Run summary</h2>
      <table>
        <tr><th>Samples</th><td>{n_samples if n_samples is not None else "-"}</td></tr>
        <tr><th>Genes (input)</th><td>{n_genes if n_genes is not None else "-"}</td></tr>
        <tr><th>Genes modeled (impulse)</th><td>{n_fit if n_fit is not None else "-"}</td></tr>
        <tr><th>Significant temporal genes (FDR&lt;0.05)</th><td>{n_sig if n_sig is not None else "-"}</td></tr>
        <tr><th>Pattern distribution</th><td>{patt_str}</td></tr>
        <tr><th>Gene sets</th><td>{gs_meta if gs_meta else "-"}</td></tr>
      </table>
    </div>
    """

    # Tops blocks
    def _render_tops():
        html = '<div class="card"><h2>Highlights</h2>'
        # Pathways
        html += '<h3>Top pathways by |rho|</h3><ul>'
        if tops_pw:
            for name, rho, amp, score in tops_pw:
                html += f'<li><span class="k">{name}</span> — rho={rho:.2f}, amp={amp:.2f}, score={score:.2f}</li>'
        else:
            html += '<li class="muted">(not available)</li>'
        html += '</ul>'
        # Genes by q
        html += '<h3>Top genes by FDR</h3><ul>'
        if tops_gq:
            for r in tops_gq:
                pk = "-" if pd.isna(r.get('time_of_peak')) else f"{float(r.get('time_of_peak')):.2f}"
                html += f"<li><span class='k'>{r.get('gene_id')}</span> — q={float(r.get('p_adj')):.2e}, R²={float(r.get('r2_impulse')):.2f}, pattern={r.get('pattern')}, peak={pk}</li>"
        else:
            html += '<li class="muted">(not available)</li>'
        html += '</ul>'
        # Genes by R2
        html += '<h3>Top genes by impulse R²</h3><ul>'
        if tops_gr2:
            for r in tops_gr2:
                r2name = next((c for c in r.keys() if c.lower()=="r2_impulse"), "r2_impulse")
                pv = r.get('p_adj', r.get('p_value', None))
                pv_str = f"{float(pv):.2e}" if isinstance(pv, (int,float,np.floating)) else "-"
                html += f"<li><span class='k'>{r.get('gene_id')}</span> — R²={float(r.get(r2name)):.2f}, q/p={pv_str}, pattern={r.get('pattern')}</li>"
        else:
            html += '<li class="muted">(not available)</li>'
        html += '</ul>'
        # Cells
        html += '<h3>Top cell types by amplitude (Δ proportion)</h3><ul>'
        if tops_cells:
            for name, amp in tops_cells:
                html += f'<li><span class="k">{name}</span> — Δ={amp:.2f}</li>'
        else:
            html += '<li class="muted">(not available)</li>'
        html += '</ul></div>'
        return html

    # Figures block
    figs = f"""
    <div class="card">
      <h2>Figures</h2>
      <h3>Pseudotime by covariate</h3>{_img_tag_if_exists(fig_cov, "pseudotime_by_covariate")}
      <h3>Top pathways vs pseudotime (|rho|)</h3>{_img_tag_if_exists(fig_pwbar, "pseudotime_vs_pathways_top15")}
      <h3>Pathway trajectories (top 12)</h3>{_img_tag_if_exists(fig_pwgrid, "temporal_pathways_top12")}
      <h3>Cell-type trajectories (top 12)</h3>{_img_tag_if_exists(fig_cmgrid, "temporal_cellmix_top12")}
      <h3>Gene trajectories (top N)</h3>{_img_tag_if_exists(fig_genegrid, "temporal_genes_topN")}
      {"<h3>DE vs Interaction</h3>"+_img_tag_if_exists(fig_debeta, "de_vs_beta") if made_de_interaction else ""}
      {"<h3>PhenoPath ELBO</h3>"+_img_tag_if_exists(fig_elbo, "elbo") if made_elbo else ""}
    </div>
    """

    # Long explanations (static text you requested), trimmed for brevity but complete.
    explanations = """
    <div class="card">
      <h2>File-by-file explanation</h2>
      <h3>1) pseudotime.csv</h3>
      <p><b>What it is:</b> A temporal coordinate per sample scaled 0–1. 0=early, 1=late.</p>
      <p><b>Columns:</b> sample_id (index), <code>pseudotime</code>.</p>
      <p><b>Biology:</b> Early vs late states; compare groups with <code>pseudotime_by_covariate.png</code>.</p>
      <p><b>Checks:</b> Range ~[0,1]; use Spearman downstream.</p>

      <h3>2) temporal_gene_fits.tsv</h3>
      <p><b>What it is:</b> Per-gene impulse fit vs linear; FDR across genes.</p>
      <p><b>Columns:</b> gene_id, time_of_peak, time_of_valley, p_value, p_adj, pattern, aic, r2_impulse.</p>
      <p><b>Biology:</b> monotonic_up/down (progressive changes); up_then_down/down_then_up (transient programs); biphasic (multi-stage transitions).</p>
      <p><b>Thresholds:</b> FDR q<0.05 (strict 0.01); r2_impulse ≥0.30 (strong ≥0.50); watch edge peaks (≈0 or 1).</p>

      <h3>3) temporal_qc.tsv</h3>
      <p><b>What it is:</b> Diagnostics per gene: smoothness, linear vs impulse, outliers.</p>
      <p><b>Columns:</b> gene, autocorr_lag1, r2_impulse, r2_linear, cooks_max.</p>
      <p><b>Thresholds:</b> autocorr_lag1 &gt;0.2; cooks_max &gt;1 warrants inspection.</p>

      <h3>Gene trajectory plots</h3>
      <p><b>What it is:</b> Scatter of gene expression (log-CPM) versus pseudotime with a smooth trend line.</p>
      <p><b>Why it helps:</b> Visual inspection of dynamic behavior for the most significant / highest-R² genes.</p>
      <p><b>Where:</b> <code>temporal_genes_topN.png</code> (gallery) and <code>genes/</code> (per-gene PNGs).</p>

      <h3>4) ssgsea/gseapy.gene_set.ssgsea.report.csv</h3>
      <p><b>What it is:</b> Long-form pathway scores per (sample, pathway). Use NES for comparisons.</p>

      <h3>5) pathway_temporal_summary.csv</h3>
      <p><b>What it is:</b> Per-pathway Spearman ρ vs pseudotime + amplitude + composite score.</p>
      <p><b>Thresholds:</b> |ρ| ≥0.30 moderate; ≥0.50 strong. Rank by <code>score = |ρ| × amp</code>.</p>

      <h3>6) pathway_profiles.csv</h3>
      <p><b>What it is:</b> Tidy long table for plotting trajectories (entity, t, score, t_norm).</p>

      <h3>7) cellmix_profiles.csv</h3>
      <p><b>What it is:</b> Tidy long table of cell-type proportions vs time.</p>
      <p><b>Thresholds:</b> Δ proportion ≥0.10 notable; ≥0.05 mild but credible.</p>

      <h3>Figures</h3>
      <ul>
        <li><b>pseudotime_by_covariate.png</b> — Are groups at different temporal stages?</li>
        <li><b>pseudotime_vs_pathways_top15.png</b> — Direction (ρ) of pathway changes.</li>
        <li><b>temporal_pathways_top12.png</b> — Timing of pathway waves.</li>
        <li><b>temporal_cellmix_top12.png</b> — Cell-type expansion/contraction.</li>
        <li><b>de_vs_beta.png</b> — Genes with condition-specific temporal slopes.</li>
        <li><b>elbo.png</b> — Variational convergence (when PhenoPath used).</li>
      </ul>
    </div>
    """

    thresholds = """
    <div class="card">
      <h2>Quick thresholds (cheat-sheet)</h2>
      <table>
        <tr><th>Genes</th><td>FDR q &lt; 0.05 (strict 0.01); R²_impulse ≥ 0.30 (strong ≥ 0.50); autocorr_lag1 &gt; 0.20; Cook’s max &gt; 1 ⇒ inspect</td></tr>
        <tr><th>Pathways</th><td>|ρ| ≥ 0.30 (strong ≥ 0.50); use amp (max−min NES) & score = |ρ|×amp</td></tr>
        <tr><th>Cell mix</th><td>Δ proportion ≥ 0.10 notable (≥ 0.05 mild); check direction vs time</td></tr>
      </table>
    </div>
    """

    story = f"""
    <div class="card">
      <h2>How to tell the biological story</h2>
      <ol>
        <li><b>Axis:</b> “We inferred a single dominant axis of progression (pseudotime) across {n_samples if n_samples is not None else "N"} samples (0→1).”</li>
        <li><b>Gene programs:</b> “We modeled {n_fit if n_fit is not None else "N"} genes, finding {n_sig if n_sig is not None else "M"} with significant non-linear dynamics (FDR&lt;0.05). Patterns indicate multi-stage transitions.”</li>
        <li><b>Pathways:</b> “Among { (0 if pw_summary is None else len(pw_summary)) } pathways, top up: [list], top down: [list].”</li>
        <li><b>Cellular composition:</b> “Cell types [A,B] expand; [C,D] recede — consistent with pathway dynamics.”</li>
        <li><b>Condition effect:</b> “Treatment shifts temporal position and/or modifies slopes for selected genes.”</li>
        <li><b>Takeaways:</b> “Early wave: [programs]; Late wave: [programs]; consistent with [mechanistic hypothesis].”</li>
      </ol>
    </div>
    """

    # Put it all together
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TDP Temporal Report — Detailed</title>{css}</head>
<body>
<h1>TDP Temporal Report — Detailed</h1>
<p class="muted">Auto-generated from the temporal run in <code>{out_dir}</code>. All links are relative; ship this whole folder to share.</p>

{quick}
{_render_tops()}
{figs}
{dir_map}
{explanations}
{thresholds}
{story}

</body></html>
"""
    (out_dir / "report_detailed.html").write_text(html, encoding="utf-8")

# -------------------------------
#           Gene plots
# -------------------------------

# --- put these ABOVE main() ---

