from .pipeline import run_pipeline



if __name__ == "__main__":
    run_pipeline(
        output_dir=r"results",
        layers={
            "genomics": r"Data/epigenomics.csv",
            "transcriptomics": r"Data/transcriptomics.csv",
            "epigenomics": r"Data/epigenomics.csv",
            "proteomics": r"Data/proteomics.csv",
            "metabolomics": r"Data/metabolomics.csv",
        },
        # metadata_path=r"",
        query_term="multi-omics integration",
        disease_term="breast cancer",

    )