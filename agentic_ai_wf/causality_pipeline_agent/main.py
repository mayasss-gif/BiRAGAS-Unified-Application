from causality_main import PipelineConfig, run_causality_pipeline

# ── SLE (Systemic Lupus Erythematosus) ──────────────────────────────────────
sle_config = PipelineConfig(
    disease_name="SLE",

    RAW_COUNTS=r"lupusInput_data\Systemic Lupus Erythematosus_GSE112087_raw_count.tsv",
    DEGS_FULL=r"lupusInput_data\DEGs_genes_pathway\c1df9595-9e18-4db1-9779-11b212c7d646_DEGs.csv",
    METADATA=r"lupusInput_data\DEGs_genes_pathway\prep_meta.csv",
    DEGS_PRIORITY=r"lupusInput_data\DEGs_genes_pathway\systemic_lupus_erythematosus_DEGs_prioritized.csv",

    PATHWAYS=r"lupusInput_data\DEGs_genes_pathway\Lupus_Pathways_Enrichment.csv",
    DECONVOLUTION=r"lupusInput_data\deconvolution_data\signature_matrix.tsv",

    TEMPORAL_FITS=r"lupusInput_data\temporal_data\temporal_gene_fits.tsv",
    GRANGER_EDGES=r"lupusInput_data\temporal_data\granger_edges_raw.csv",

    PERTURBATION=r"lupusInput_data\perturbation_data\CRISPR_GuideLevel_Avana_SelectedModels_long.csv",
    ESSENTIALITY=r"lupusInput_data\perturbation_data\GeneEssentiality_ByMedian.csv",
    DRUG_LINKS=r"lupusInput_data\perturbation_data\causal_link_table_with_relevance.csv",
    CAUSAL_DRIVERS=r"lupusInput_data\perturbation_data\CausalDrivers_Ranked.csv",

    GWAS_ASSOC=r"lupusInput_data\GWAS_data\Systemic lupus erythematosus__genetic-evidence.xlsx",
    SIGNOR_EDGES=r"lupusInput_data\SIGNOR_data\SIGNOR_Subnetwork_Edges.tsv",
    GENETIC_PRIORS=r"lupusInput_data\SIGNOR_data\Combined_Genetic_SIGNOR_Prioritization.tsv",
    GENE_EVIDENCE=r"lupusInput_data\GWAS_data\Systemic Lupus Erythematosus_Whole_Blood_GeneLevel_GeneticEvidence.tsv",
    VARIANT_EVIDENCE=r"lupusInput_data\GWAS_data\Systemic Lupus Erythematosus_Whole_Blood_VariantLevel_GeneticEvidence.tsv",
    MR_RESULTS=r"lupusInput_data\GWAS_data\MR_MAIN_RESULTS_ALL_GENES.csv",
)

result = run_causality_pipeline(sle_config, output_dir="my_output")

print("\n── DAG Stats (Phase 1) ──")
for k, v in result['dag_stats'].items():
    print(f"  {k}: {v}")

print("\n── Centrality (Phase 2) ──")
report = result['metrics_report']
print(f"  Tier 1 Master Regulators:    {len(report['hub_classifications']['Tier_1_Master_Regulators'])}")
print(f"  Tier 2 Secondary Drivers:    {len(report['hub_classifications']['Tier_2_Secondary_Drivers'])}")
print(f"  Tier 3 Downstream Effectors: {len(report['hub_classifications']['Tier_3_Downstream_Effectors'])}")


# ── Another disease (example skeleton) ──────────────────────────────────────
# ra_config = PipelineConfig(
#     disease_name="Rheumatoid Arthritis",
#     RAW_COUNTS=r"raInput_data\RA_GSE00000_raw_count.tsv",
#     DEGS_PRIORITY=r"raInput_data\DEGs_genes_pathway\ra_DEGs_prioritized.csv",
#     METADATA=r"raInput_data\DEGs_genes_pathway\prep_meta.csv",
#     PATHWAYS=r"raInput_data\DEGs_genes_pathway\RA_Pathways_Enrichment.csv",
#     # ... remaining paths ...
# )
# ra_result = run_causality_pipeline(ra_config, output_dir="output/ra")
