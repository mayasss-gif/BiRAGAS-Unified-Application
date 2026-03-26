from pathlib import Path
import gzip, shutil, re
from typing import List, Tuple
from agents import function_tool  # <-- use your actual import if different

def extract_gse_id(filename: str):
    match = re.search(r"(GSE\d+)", filename)
    return match.group(1) if match else None

def is_series_matrix(fname: str) -> bool:
    return "series_matrix" in fname and fname.endswith((".txt", ".txt.gz"))

def is_counts_matrix(fname: str) -> bool:
    return fname.endswith((".csv", ".tsv", ".txt", ".txt.gz")) and "series_matrix" not in fname

def _detect_de_dataset_pairs(folder: str) -> List[Tuple[Path, Path]]:
    """
    Internal logic to detect count–metadata file pairs. Also unzips .gz files.
    """
    folder = Path(folder).resolve()
    for path in folder.rglob("*.gz"):
        out = path.with_suffix("")
        try:
            with gzip.open(path, "rb") as fin, open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        except Exception as e:
            print(f"❌ Failed to unzip {path}: {e}")
            continue

    all_files = list(folder.rglob("*"))
    pairs = []
    for c in all_files:
        for m in all_files:
            if c == m: continue
            if is_counts_matrix(c.name) and is_series_matrix(m.name):
                if extract_gse_id(c.name) == extract_gse_id(m.name):
                    pairs.append((c, m))
    return pairs

@function_tool
def detect_de_dataset_pairs(data_folder: str) -> List[Tuple[str, str]]:
    """
    Detects count/metadata file pairs from nested GSE folders.
    Auto-unzips .gz files before matching.
    Returns list of string paths for agent use.
    """
    return [(str(c), str(m)) for c, m in _detect_de_dataset_pairs(data_folder)]
from pathlib import Path
import time
from typing import Union

def _run_deg_pipeline(
    counts_file: Union[str, Path],
    metadata_file: Union[str, Path],
    disease_name: str,
    outdir: Union[str, Path] = "results"
) -> str:
    """
    Core DEG analysis logic (simulated for now).
    Creates dummy output file.
    """
    counts_file = Path(counts_file)
    outdir = Path(outdir).resolve()
    output = outdir / counts_file.stem
    output.mkdir(parents=True, exist_ok=True)

    # Simulate DEG output
    time.sleep(1.0)
    result = output / "DEG_results.txt"
    result.write_text("Gene\tlog2FoldChange\tpvalue\nBRCA1\t2.3\t0.01\nTP53\t-1.7\t0.04\n")

    return f"✅ DEG complete for {counts_file.name} → {result}"

@function_tool
def run_deg_pipeline(
    counts_file: str,
    metadata_file: str,
    disease_name: str,
    outdir: str = "results"
) -> str:
    """
    Runs DEG analysis for one count/metadata pair.
    Simulated; replace with real pipeline (e.g., DESeq2 or limma).
    """
    return _run_deg_pipeline(counts_file, metadata_file, disease_name, outdir)



import time, resource
from typing import List


def _run_deg_folder_pipeline(data_folder: str, disease_name: str) -> str:
    """
    Runs DEG analysis on all matched dataset pairs in a directory.
    Tracks time and memory usage.
    """
    from .code_runner import _detect_de_dataset_pairs, _run_deg_pipeline  # adjust if needed

    pairs = _detect_de_dataset_pairs(data_folder)
    if not pairs:
        return "⚠️ No valid dataset pairs found."

    log = []
    t0 = time.time()

    for i, (counts, meta) in enumerate(pairs, 1):
        try:
            start = time.time()
            msg = _run_deg_pipeline(counts, meta, disease_name)
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            log.append(f"{msg} | Time: {time.time()-start:.2f}s | Mem: {mem:.1f}MB")
        except Exception as e:
            log.append(f"❌ Sample {i} failed: {e}")
            continue

    t1 = time.time()
    log.append(f"⏱️ Total time: {t1-t0:.2f}s across {len(pairs)} samples.")
    return "\n".join(log)

@function_tool
def run_deg_folder_pipeline(data_folder: str, disease_name: str) -> str:
    """
    Agent-compatible wrapper for full folder batch DEG execution.
    """
    return _run_deg_folder_pipeline(data_folder, disease_name)
