from .gwas_mr import run_full_pipeline


results = run_full_pipeline(
    disease_name="Lupus", #Breast Cancer
    # disease_name="Diabetes",
    biosample_type="Whole Blood", #Lupus
    # biosample_type="Pancreas", #Type 2 Diabetes
    output_dir="agentic_ai_wf/shared/gwas_mr_data",
    gwas_data_dir="agentic_ai_wf/shared/gwas_data",
)