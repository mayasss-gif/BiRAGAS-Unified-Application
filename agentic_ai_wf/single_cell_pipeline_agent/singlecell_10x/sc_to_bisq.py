import scanpy as sc
from pathlib import Path
import os
import argparse
import scipy.sparse as sp


# ----------------------------------------------------------
# 1) Argument parser
# ----------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a Scanpy h5ad file for Bisque deconvolution."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input h5ad file."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the Bisque preparation pipeline."
    )
    return parser.parse_args()


# ----------------------------------------------------------
# 2) Function: Prepare AnnData for Bisque
# ----------------------------------------------------------
def prepare_for_bisque(adata_in):
    """
    Create a *copy* of the AnnData object with only what Bisque needs:
      - adata.X: raw counts (preferred) or normalized matrix
      - adata.obs['celltype']
      - adata.obs['sample']
      - adata.var_names
    """
    adata = adata_in.copy()

    # ---- Use raw counts if available ----
    if getattr(adata, "raw", None) is not None:
        print("[INFO] adata.raw detected; using adata.raw.X as main matrix.")
        adata.X = adata.raw.X.copy()
    else:
        print("[WARN] adata.raw is None — using adata.X (likely normalized).")

    # ---- Required columns ----
    for col in ["celltype", "sample"]:
        if col not in adata.obs.columns:
            print(f"[WARN] obs['{col}'] missing!")
        else:
            print(f"[OK] obs['{col}'] present, {adata.obs[col].nunique()} unique values")

    # ---- Remove raw layer ----
    if getattr(adata, "raw", None) is not None:
        print("Removing adata.raw...")
        adata.raw = None

    # ---- Remove embeddings ----
    if adata.obsm:
        print(f"Clearing obsm keys: {list(adata.obsm.keys())}")
        adata.obsm.clear()

    if adata.varm:
        print(f"Clearing varm keys: {list(adata.varm.keys())}")
        adata.varm.clear()

    # ---- Remove graphs ----
    if hasattr(adata, "obsp"):
        for key in ["connectivities", "distances"]:
            if key in adata.obsp:
                print(f"Removing obsp['{key}']")
                del adata.obsp[key]

    # ---- Remove unnecessary var columns ----
    if "highly_variable" in adata.var.columns:
        print("Dropping var['highly_variable']")
        del adata.var["highly_variable"]

    # ---- Clean obs ----
    for col in ["leiden", "pct_counts_mt", "total_counts"]:
        if col in adata.obs.columns:
            print(f"Dropping obs['{col}']")
            del adata.obs[col]

    # ---- Clear uns ----
    if adata.uns:
        print(f"Clearing uns keys: {list(adata.uns.keys())}")
        adata.uns.clear()

    # ---- Ensure CSR format ----
    if sp.issparse(adata.X):
        print("Converting X to CSR format...")
        adata.X = adata.X.tocsr()

    print(f"[FINAL] Bisque-ready shape: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


# ----------------------------------------------------------
# 3) Main processing function (can be called directly)
# ----------------------------------------------------------
def process_h5ad_file(input_file, output_file=None, return_adata=False):
    """
    Process an h5ad file to prepare it for Bisque deconvolution.
    
    Parameters
    ----------
    input_file : str or Path
        Path to input h5ad file.
    output_file : str or Path, optional
        Path to output h5ad file. If None, will be generated automatically
        as "bisque_ready_<input_filename>".
    return_adata : bool, default False
        If True, return the processed AnnData object instead of saving to file.
    
    Returns
    -------
    AnnData or None
        If return_adata=True, returns the processed AnnData object.
        Otherwise, returns None and saves to file.
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"\nLoading input AnnData from:\n  {input_path}")
    adata_full = sc.read_h5ad(input_path)
    print(f"Loaded {adata_full.n_obs} cells × {adata_full.n_vars} genes")

    # Generate output filename if not provided
    if output_file is None:
        output_path = input_path.with_name("bisque_ready_" + input_path.name)
    else:
        output_path = Path(output_file)

    if not return_adata:
        print(f"\nOutput file will be saved as:\n  {output_path}")

    # Prepare copy
    adata_bisque = prepare_for_bisque(adata_full)

    if return_adata:
        print("\nDone. Returning AnnData object.")
        return adata_bisque

    # Save output
    print(f"\nSaving Bisque-ready AnnData to:\n  {output_path}")
    adata_bisque.write_h5ad(output_path, compression="gzip")

    # Show file sizes
    full_size = os.path.getsize(input_path) / (1024**3)
    bisque_size = os.path.getsize(output_path) / (1024**3)

    print("\nDone.")
    print(f"Original file size:   {full_size:.2f} GB")
    print(f"Bisque-ready size:    {bisque_size:.2f} GB\n")
    
    return None


# ----------------------------------------------------------
# 4) CLI main function
# ----------------------------------------------------------
# def main():

#     process_h5ad_file(input_file=r"C:\Ayass Bio Work\Agentic_AI_ABS\single_cell_pipeline\sc_test_run\GSM6360681_N_HPV_NEG_2\SC_RESULTS\single_dataset_processed_scanpy_output.h5ad")


# ----------------------------------------------------------
# 5) Entry point
# ----------------------------------------------------------
# if __name__ == "__main__":
#     main()
