"""
Direct function-based interface for the harmonizer pipeline.
Call harmonize_local() or harmonize_single() directly with parameters.
"""
from pathlib import Path
import json
import sys
import os

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from .pipeline_agent import harmonize_from_local, harmonize_single_paths
except Exception as e:
    print("Could not import pipeline code. Make sure pipeline_agent.py and harmonizer_core.py are in the same folder.")
    print(f"Import error: {e}")
    raise


def pretty(obj):
    """Format object as pretty JSON string."""
    try:
        return json.dumps(obj, indent=2)
    except Exception:
        return str(obj)


def harmonize_local(
    data_root: str,
    output_dir: str,
    combine: bool = True,
) -> dict:
    """
    Harmonize datasets by auto-discovering 'prep' folders in a root directory.
    
    Args:
        data_root: Root directory to crawl. Any subfolder whose path contains 'prep' will be considered.
        output_dir: Directory where all results should be stored.
        combine: If True, attempt to combine multiple datasets (default: True).
        
    Returns:
        Dictionary containing:
            - result: Full result dictionary from harmonize_from_local()
            - outputs: List of all output paths (outdir, figdir, zip files)
            - summary_path: Path to saved summary file (if save_summary=True)
    
    Example:
        >>> result = harmonize_local(
        ...     data_root="data/my_datasets",
        ...     output_dir="/output",
        ...     combine=True
        ... )
        >>> print(result["outputs"])
    """
    save_summary: bool = True,
    print_output: bool = True
    result = harmonize_from_local(
        data_root=data_root,
        combine=combine,
        out_root=output_dir
    )
    
    outputs = []
    if result.get("mode") == "single":
        outputs.append(result["result"]["outdir"])
        outputs.append(result["result"]["figdir"])
        outputs.append(result["result"]["zip"])
    elif result.get("mode") == "multi":
        combined = result.get("combined") or {}
        for name, run in (result.get("runs") or {}).items():
            outputs.extend([run.get("outdir"), run.get("figdir"), run.get("zip")])
        if combined:
            outputs.extend([combined.get("outdir"), combined.get("figdir"), combined.get("zip")])
    
    outputs = [o for o in outputs if o]
    
    summary_path = None
    if save_summary:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "last_run_summary.local.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"result": result, "outputs": outputs}, f, indent=2)
    
    if print_output:
        print("\n=== Run summary (local) ===")
        print(pretty(result))
        if summary_path:
            print(f"\nWrote summary: {summary_path}")
        print("\nOutput paths:")
        for o in outputs:
            print(f" - {o}")
    
    return {
        "result": result,
        "outputs": outputs,
        "summary_path": summary_path
    }


def harmonize_single(
    counts_path: str,
    meta_path: str,
    output_dir: str,
    
) -> dict:
    """
    Harmonize a single dataset with explicit file paths.
    
    Args:
        counts_path: Path to the counts/expression table (csv/tsv/xlsx).
        meta_path: Path to the metadata table (csv/tsv/xlsx).
        output_dir: Directory where all results should be stored.
    
    Returns:
        Dictionary containing:
            - result: Full result dictionary from harmonize_single_paths()
            - outputs: List of all output paths (outdir, figdir, zip)
            - summary_path: Path to saved summary file (if save_summary=True)
    
    Example:
        >>> result = harmonize_single(
        ...     counts_path="C:/data/counts.csv",
        ...     meta_path="C:/data/metadata.csv",
        ...     output_dir="C:/output"
        ... )
        >>> print(result["outputs"])
    """

    save_summary: bool = True,
    print_output: bool = True
    res = harmonize_single_paths(
        counts_path=counts_path,
        meta_path=meta_path,
        out_mode="default",
        out_root=output_dir,
        create_zip=False
    )
    
    outputs = [res.get("outdir"), res.get("figdir"), res.get("zip")]
    outputs = [o for o in outputs if o]
    
    summary_path = None
    if save_summary:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "last_run_summary.single.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"result": res, "outputs": outputs}, f, indent=2)
    
    if print_output:
        print("\n=== Run summary (single) ===")
        print(pretty(res))
        if summary_path:
            print(f"\nWrote summary: {summary_path}")
        print("\nOutput paths:")
        for o in outputs:
            print(f" - {o}")
    
    return {
        "result": res,
        "outputs": outputs,
        "summary_path": summary_path
    }


# Example usage when run directly
if __name__ == "__main__":
    # Example 1: Single dataset
    # result = harmonize_single(
    #     counts_path="path/to/counts.csv",
    #     meta_path="path/to/metadata.csv",
    #     output_dir="path/to/output"
    # )
    
    # Example 2: Local discovery mode
    # result = harmonize_local(
    #     data_root="path/to/data/root",
    #     output_dir="path/to/output",
    #     combine=True
    # )
    
    print("Import this module and call harmonize_local() or harmonize_single() directly.")
    print("\nExample:")
    print("  from harmonizer import harmonize_single")
    print("  result = harmonize_single(counts_path='...', meta_path='...', output_dir='...')")

