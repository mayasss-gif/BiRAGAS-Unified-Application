from .core import *
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TDP Temporal Module (Step 46) — raw counts -> pseudotime -> impulse fits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i','--input_dir', required=True)
    p.add_argument('-o','--output_dir', required=True)
    p.add_argument('--counts', required=True, help='Counts CSV/TSV (first col gene IDs, others samples)')
    p.add_argument('--metadata', required=True, help='Metadata CSV/TSV (must include sample_id)')
    p.add_argument('--gene_col', default='gene', help='Name of gene ID column in counts file')
    p.add_argument('--time_col', default='', help='Numeric column in metadata for true time; otherwise pseudotime is estimated')
    p.add_argument('--covariate', default='', help='Primary covariate (e.g., condition)')
    p.add_argument('--treatment_level', default='', help='Name of treatment level inside covariate (for DE/interaction plots)')
    p.add_argument('--extra_covariates', default='', help='Comma-separated extra covariates (optional)')

    # transforms
    p.add_argument('--assume_log', action='store_true', help='Treat counts as already log-scale')

    # pseudotime
    p.add_argument('--use_phenopath', action='store_true', help='Use R phenopath via rpy2 if available')
    p.add_argument('--phenopath_thin', type=int, default=5)
    p.add_argument('--phenopath_maxiter', type=int, default=500)

    # impulse fitting
    p.add_argument('--max_iter_nonlin', type=int, default=800, help='Max fev for curve_fit')

    # prefilter & workload control
    p.add_argument('--prefilter_rho', type=float, default=0.2, help='Min |Spearman rho| with pseudotime to run impulse')
    p.add_argument('--prefilter_min_var', type=float, default=0.0, help='Min variance to run impulse')
    p.add_argument('--max_genes', type=int, default=20000, help='Cap number of genes for impulse (picked by |rho|)')
    p.add_argument('--n_jobs', type=int, default=1, help='Parallel jobs for per-gene fits (joblib)')
    p.add_argument('--checkpoint_every', type=int, default=50, help='Write partial results every N genes')
    p.add_argument('--progress_every', type=int, default=100, help='Print progress every N genes')
    p.add_argument('--precheck_only', action='store_true', help='Stop after pseudotime.csv')
    p.add_argument('--verbose', action='store_true')
        # targeted gene list
    p.add_argument('--genes_list', default='', help='Path to TXT/CSV with selected gene IDs (one per line or provide --genes_list_col)')
    p.add_argument('--genes_list_col', default='', help='Column name in the genes_list file to use (optional)')
    p.add_argument('--signature_from_gene_list', action='store_true', help='Create a single signature trajectory from the selected genes (mean logCPM)')
    p.add_argument('--restrict_pathways_by_gene_list', action='store_true', help='Filter pathway outputs to those overlapping the selected genes (requires local .gmt in --gene_sets)')


    # ssgsea
    p.add_argument('--run_ssgsea', action='store_true')
    p.add_argument('--gene_sets', default='Hallmark',    help='gseapy library name (e.g., MSigDB_Hallmark_2020) or path to local .gmt; aliases: H/h -> best Hallmark match')
    p.add_argument('--organism', default='Human', help='Organism label for gseapy library resolution (e.g., Human, Mouse)')
    p.add_argument('--ssgsea_report', default='', help='Optional: path to a gseapy ssGSEA long-form report (columns: Name,Term,ES,NES) to build pathway outputs without running ssGSEA')

    # deconvolution
    p.add_argument('--deconv_csv', default='', help='CSV/TSV of deconvolution proportions (rows=samples, cols=cell types)')

    # report & graphics
    p.add_argument('--make_figures', action='store_true', help='Produce PNG figures & causal pack')
    p.add_argument('--top_n', type=int, default=12, help='Top N entities for grids & per-entity PNGs')
    p.add_argument('--render_report', action='store_true')
    # gene plotting
    p.add_argument('--plot_genes_top_n', type=int, default=24, help='How many genes to showcase as a grid in the report')
    p.add_argument('--plot_genes_rank_by', default='fdr', choices=['fdr','r2'], help='Ranking for gene gallery: fdr (ascending) or r2 (descending)')
    p.add_argument('--save_gene_plots', action='store_true', help='Save per-gene PNGs under genes/')

    # misc
    p.add_argument('--seed', type=int, default=42)

    return p

# -------------------------------
# IO & logging
# -------------------------------

