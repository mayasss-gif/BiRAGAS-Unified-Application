from .ipaa_causality import run_ipaa_pipeline,ItemSpec

rc = run_ipaa_pipeline(
    outdir="results",
    item=[
        ItemSpec(
            name="Breast_Cancer", 
            input=r"agentic_ai_wf/ipaa_causality/input_data/colorectal_rectal_cancer_GSE218999_raw_counts.csv", 
            meta=r"agentic_ai_wf/ipaa_causality/input_data/colorectal_rectal_cancer_GSE218999_metadata.csv",
        ),
        # ItemSpec(
        #     name="Colorectal Rectal Cancer", 
        #     input="input_data/colorectal_rectal_cancer_GSE218999_raw_counts.csv", 
        #     meta="input_data/colorectal_rectal_cancer_GSE218999_metadata.csv",
        # ),
        # ItemSpec(
        #     name="Lung Cancer", 
        #     input="input_data/DS-20250517_counts.csv", 
        #     meta="input_data/DS-20250517_metadata.csv",
        #     tissue="lung"
        # ),
        # ItemSpec(
        #     name="Colorectal Rectal Cancer", 
        #     input=r"C:\Users\Raafeh\Downloads\colorectal_rectal_cancer_GSE218999_raw_counts.csv", 
        #     meta=r"C:\Users\Raafeh\Downloads\colorectal_rectal_cancer_GSE218999_metadata.csv",
        # )
    ]
)