# run_check_cell_essentiality.py
from pathlib import Path
from typing import Optional
from logging import Logger


import pandas as pd


# ✅ Use the new non-interactive core pipeline from src/check_cell_essentiality.py
from .check_cell_essentiality import (
    run_check_cell_essentiality as run_core_check_cell_essentiality,
    CheckCellEssentialityResult,
)

# Dependencies / essentiality / plots
from .depmap_dependencies_uif import (
    build_tidy_dependencies,
    boxplot_by_lineage,
    plot_top_dependents,
    summarize_gene_essentiality,
    build_essentiality_tables,
)


# -------------------------------------------------------------------
# Main orchestrator: Steps 4–8 = Check_CellEssentiality + dependencies
# -------------------------------------------------------------------
def run_check_cell_essentiality(output_dir: Path, deg_simple_path: Path, logger: Optional[Logger] = None) -> None:
    """
    High-level entry point for the DepMap pipeline:

      4–8a  -> core Check_CellEssentiality pipeline
               (cell lines, model selection, gene selection, gene layers)
      8b+   -> tidy dependencies, summary, plots, essentiality tables

    All model/gene selections are NON-INTERACTIVE and driven by:
      - BASE_INPUT_DIR / ModelSelection.txt
      - BASE_INPUT_DIR / GeneSelection.txt

    Default behavior:
      - If GeneSelection.txt uses mode=all → ALL genes go downstream.
      - If mode=top → uses the top_up / top_down values in that file.
      - Per-gene plots:
          * top 20 genes by median essentiality
          * top 20 cell lines per gene
    """


    logger.info("=== Step 04–08: Check_CellEssentiality pipeline started ===")
    logger.info("Output directory: %s", output_dir)

    print("=== Check_CellEssentiality pipeline ===")


    # ---------------------------------------------------------------
    # CORE PIPELINE (NON-INTERACTIVE)
    #   - DepMap cell-line setup
    #   - Model selection (from ModelSelection.txt)
    #   - Gene selection  (from GeneSelection.txt)
    #   - Gene identity check
    #   - Gene-layer matrices
    # ---------------------------------------------------------------
    core_result: CheckCellEssentialityResult = run_core_check_cell_essentiality(
        output_dir=output_dir,
        deg_simple_path=deg_simple_path,
        logger=logger
    )

    ctx = core_result.ctx
    model_sel = core_result.model_selection

    # Path where select_models() wrote the selected models
    models_path = Path(model_sel.out_path)

    # Path where build_gene_list_from_prepared_simple() wrote the gene list
    genes_path = output_dir / "DepMap_Genes" / "InputGenes_selected.csv"

    print("\n=== Core Check_CellEssentiality complete ===")
    print("Selected models file:", models_path)
    print("Selected genes file :", genes_path)

    # ---------------------------------------------------------------
    # STEP 8b: Dependencies tidy
    # ---------------------------------------------------------------
    dep_result = build_tidy_dependencies(
        output_dir=output_dir,
        selected_genes_path=genes_path,
        selected_models_path=models_path,
        logger=logger,
        TH_CORE=-1.0,
        TH_STRONG=-0.7,
        TH_MOD=-0.3,
    )

    print("\n=== DepMap Dependencies Tidy Completed ===")
    print(f"Selected genes:  {dep_result.n_selected_genes}")
    print(f"Selected models: {dep_result.n_selected_models}")
    print(f"Tidy rows:       {dep_result.n_rows}")
    print("Mapping file:    ", dep_result.mapping_path)
    print("Tidy CSV:        ", dep_result.tidy_path)
    print("Thresholds:      ", dep_result.thresholds)

    tidy = pd.read_csv(dep_result.tidy_path)

    # ---------------------------------------------------------------
    # Gene-level summary (always computed; needed for tables)
    # ---------------------------------------------------------------
    summary_result = summarize_gene_essentiality(
        tidy,
        output_dir=output_dir,
        TH_CORE=dep_result.thresholds["TH_CORE"],
        TH_STRONG=dep_result.thresholds["TH_STRONG"],
        TH_MOD=dep_result.thresholds["TH_MOD"],
        top_per_gene=2,
        min_prob=None,
    )
    summary_df = summary_result["summary_df"]

    # ---------------------------------------------------------------
    # Per-gene plots (NON-INTERACTIVE)
    #   - top 20 genes by median_effect (most negative = most essential)
    #   - top 20 cell lines per gene
    # ---------------------------------------------------------------
    if summary_df.empty:
        print("\nNo genes in summary_df – skipping per-gene plots.")
    else:
        print("\n=== Per-gene plotting (non-interactive) ===")
        # Top N genes by median essentiality (most negative)
        N_GENES_FOR_PLOTS = 20
        N_CELLS_PER_GENE = 20

        genes_for_plots = (
            summary_df.sort_values("median_effect")
            .head(N_GENES_FOR_PLOTS)["Gene"]
            .tolist()
        )

        print(f"Will generate per-gene plots for up to {len(genes_for_plots)} genes.")
        print(f"- Top {N_GENES_FOR_PLOTS} genes by median essentiality")
        print(f"- Top {N_CELLS_PER_GENE} dependent cell lines per gene")

        # Boxplots by lineage (limit to first 4 genes to avoid huge spam)
        for g in genes_for_plots[: min(4, len(genes_for_plots))]:
            boxplot_by_lineage(
                tidy,
                g,
                TH_STRONG=dep_result.thresholds["TH_STRONG"],
                min_n=5,
                show=False,
            )

        # Top dependent barplots for all selected genes (top N cells per gene)
        tidy_subset = tidy[tidy["Gene"].isin(genes_for_plots)].copy()
        if not tidy_subset.empty:
            plot_top_dependents(
                output_dir=output_dir,
                tidy=tidy_subset,
                top_n=N_CELLS_PER_GENE,
                TH_STRONG=dep_result.thresholds["TH_STRONG"],
                show=False,
            )
        else:
            print("tidy_subset is empty – skipping top-dependent barplots.")

    # ---------------------------------------------------------------
    # Essentiality tables (per-model, by-median, by-any-model)
    # ---------------------------------------------------------------
    build_essentiality_tables(
        output_dir=output_dir,
        tidy=tidy,
        TH_CORE=dep_result.thresholds["TH_CORE"],
        TH_STRONG=dep_result.thresholds["TH_STRONG"],
        TH_MOD=dep_result.thresholds["TH_MOD"],
        min_prob=None,
        enforce_min_prob_per_model=False,
        essential_rule="strong_or_better",
        summary_df=summary_df,
    )

    logger.info("Check_CellEssentiality pipeline (full) completed successfully.")
    print("\n✅ Full Check_CellEssentiality + dependencies pipeline completed.")

