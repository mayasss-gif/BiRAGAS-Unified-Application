#fastq_agents/fastq_tools.py
import os
import re
import json
import pathlib
import zipfile
import subprocess
import pandas as pd
from pathlib import Path
from agents import function_tool
from .constants import INDEX_PATH


N_THREADS = os.environ.get("FASTQ_THREADS", "16")


@function_tool
def run_fastqc(fastq_path: str, output_dir: str) -> str:
    """
    Tool: Run FastQC for a single FASTQ file.
    Args:
        fastq_path (str): Path to the input FASTQ file.
        output_dir (str): Directory to save FastQC outputs.
    Returns:
        str: FastQC run log.
    """
    os.makedirs(output_dir, exist_ok=True)
    fq_base = os.path.basename(fastq_path).replace('.gz','').replace('.fastq','')
    qc_folder = os.path.join(output_dir, fq_base + "_fastqc")
    expected_summary = os.path.join(qc_folder, "summary.txt")
    if os.path.exists(expected_summary):
        msg = f"[FastQC] Report already exists for {fastq_path} → Skipping."
        print(msg)
        return msg
    cmd = ["fastqc", "--threads", N_THREADS, "--outdir", output_dir, fastq_path]
    print(f"[FastQC] Running: {' '.join(cmd)}")
    # Stream the output live:
    proc = subprocess.Popen(cmd, stdout=None, stderr=None)
    proc.communicate()
    return f"[FastQC] Finished for {fastq_path}"


@function_tool
def load_fastqc_summary(fastqc_dir: str, fq_base: str) -> str:
    """
    Loads FastQC summary.txt from zipped or unzipped FastQC output for a given sample.
    - Looks for fq_base_fastqc.zip and reads summary.txt inside (recommended FastQC output)
    - If not found, looks for *_fastqc/summary.txt (unpacked)
    Returns summary.txt as a string or '' if not found.
    """
    from pathlib import Path
    import zipfile

    fastqc_dir = Path(fastqc_dir)
    fq_base_clean = Path(fq_base).stem.replace('.fastq', '').replace('.fq', '').replace('.gz','')

    # 1. Look for <fq_base>_fastqc.zip and extract summary.txt
    for zip_path in fastqc_dir.glob(f"{fq_base_clean}_fastqc.zip"):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    if name.endswith("summary.txt"):
                        print(f"[load_fastqc_summary] Found summary.txt in {zip_path} ({name})")
                        return zf.read(name).decode()
        except Exception as e:
            print(f"[load_fastqc_summary][WARN] Could not open {zip_path}: {e}")

    # 2. Look for *_fastqc/summary.txt (unpacked)
    for summary_path in fastqc_dir.glob(f"{fq_base_clean}_fastqc/summary.txt"):
        print(f"[load_fastqc_summary] Found summary.txt at {summary_path}")
        return summary_path.read_text()

    print(f"[load_fastqc_summary][ERROR] Could not find summary.txt for {fq_base} in {fastqc_dir}")
    return ''


@function_tool
def trimming_decision(
    fastqc_summary_r1: str,
    fastqc_summary_r2: str = None
) -> dict:
    """
    Agent tool: Decide trimming tool and parameters based on FastQC summary content.
    Always prints the full summary.txt content for R1 and R2 (if provided),
    or states clearly if summary.txt is not found.
    Also prints which important flags (if any) were found.

    Args:
        fastqc_summary_r1 (str): Contents of FastQC summary.txt for R1.
        fastqc_summary_r2 (str, optional): Contents of FastQC summary.txt for R2 (optional).

    Returns:
        dict: {tool, params, reason}
    """
    important_flags = ("Per base sequence quality", "Adapter content", "Overrepresented sequences")
    results = {"R1": {}, "R2": {}}
    trimming_reasons = []

    def parse_summary(txt, label):
        lines = txt.strip().splitlines()
        flagged = []
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 2:
                status, mod = parts[0].strip(), parts[1].strip()
                if mod in important_flags and status in ("FAIL", "WARN"):
                    flagged.append((mod, status))
                    if mod == "Adapter content":
                        trimming_reasons.append("adapter")
                    else:
                        trimming_reasons.append("quality")
                results[label][mod] = status
        return flagged

    # ----------- Always print actual summary.txt or missing message -----------
    print("\n========= FastQC summary.txt (R1) =========")
    if fastqc_summary_r1:
        print(fastqc_summary_r1)
    else:
        print("summary.txt not found for R1")
    print("===========================================\n")

    if fastqc_summary_r2 is not None:  # can be '' or None
        print("\n========= FastQC summary.txt (R2) =========")
        if fastqc_summary_r2:
            print(fastqc_summary_r2)
        else:
            print("summary.txt not found for R2")
        print("===========================================\n")

    flagged_r1 = parse_summary(fastqc_summary_r1, "R1") if fastqc_summary_r1 else []
    flagged_r2 = parse_summary(fastqc_summary_r2, "R2") if fastqc_summary_r2 else []

    # Print flagged results
    for label, flagged in (("R1", flagged_r1), ("R2", flagged_r2)):
        if flagged:
            print(f"[TrimmingDecision] Important FastQC warnings/failures for {label}:")
            for mod, status in flagged:
                print(f"  {mod}: {status}")

    # Decision logic (unchanged)
    if not (flagged_r1 or flagged_r2):
        print("[TrimmingDecision] No important FastQC flags detected. Skipping trimming for this sample.")
        return {"tool": "none", "params": "", "reason": "PASS"}
    elif "adapter" in trimming_reasons:
        print(f"[TrimmingDecision] Adapter-related flags detected. Will run Cutadapt.")
        return {"tool": "cutadapt", "params": "-a AGAT... -A AGAT...", "reason": "adapter"}
    else:
        print(f"[TrimmingDecision] Quality-related flags detected. Will run Trimmomatic.")
        return {"tool": "trimmomatic", "params": "LEADING:3 ...", "reason": "quality"}



def ensure_parent_dir(path):
    """Ensures the parent directory for a file exists."""
    if path is not None:
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


@function_tool
def run_cutadapt(
    r1_path: str,
    r2_path: str = None,
    out_r1: str = None,
    out_r2: str = None
) -> str:
    """
    Runs Cutadapt for adapter and quality trimming of FASTQ files.

    - Handles both single-end (only r1_path and out_r1) and paired-end (both R1 and R2) files.
    - Detects mode automatically: if r2_path and out_r2 are given, runs paired-end mode.
    - Uses standard Illumina adapter sequences.
    - Ensures output directories exist before running.
    - Returns the complete log output from Cutadapt (stdout and stderr).

    Notes:
        - The function detects single-end or paired-end mode by checking if both `r2_path` and `out_r2` are provided.
        - The adapters used are standard Illumina universal adapters.
        - Parent directories for output files are created if needed.
        - Errors in execution will be returned in the output string for debugging/logging.
        - This tool is designed for autonomous agent pipelines: always pass only real input files.
    """
    ensure_parent_dir(out_r1)
    if r2_path and out_r2:
        ensure_parent_dir(out_r2)
        # Paired-end mode
        cmd = [
            "cutadapt",
            "-a", "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA",
            "-A", "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT",
            "-q", "20,20", "--minimum-length", "36",
            "-j", "8", "-o", out_r1, "-p", out_r2, r1_path, r2_path
        ]
    else:
        # Single-end mode
        cmd = [
            "cutadapt",
            "-a", "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA",
            "-q", "20", "--minimum-length", "36",
            "-j", N_THREADS, "-o", out_r1, r1_path
        ]
    print("✂️  Step 2: Running Cutadapt...")
    print(f"⚙️ Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    return (result.stdout or "") + "\n" + (result.stderr or "")


@function_tool
def run_trimmomatic_pe(
    r1_in: str,
    r2_in: str = None,
    r1_out: str = None,
    r1_unpaired: str = None,
    r2_out: str = None,
    r2_unpaired: str = None
) -> str:
    """
    Runs Trimmomatic for trimming FASTQ files.

    - Supports both single-end (just r1_in and r1_out) and paired-end mode (all arguments).
    - If all paired-end arguments are given, runs in paired mode; otherwise, uses single-end.
    - Trims adapters and low-quality bases with recommended parameters.
    - Creates output directories if they do not exist.
    - Returns all Trimmomatic log output.

    Notes:
        - Detects mode (paired/single) by the presence of all six paired-end args.
        - Automatically creates output directories as needed.
        - For single-end, unpaired outputs are ignored.
        - For paired-end, all outputs must be specified.
        - Designed for use by agentic RNA-seq pipelines.
    """
    ensure_parent_dir(r1_out)
    if r2_in and r2_out and r1_unpaired and r2_unpaired:
        for out_path in [r1_out, r1_unpaired, r2_out, r2_unpaired]:
            ensure_parent_dir(out_path)
        # Paired-end mode
        cmd = [
            "trimmomatic", "PE", "-phred33", "-threads", N_THREADS,
            r1_in, r2_in,
            r1_out, r1_unpaired,
            r2_out, r2_unpaired,
            "LEADING:3", "TRAILING:3", "SLIDINGWINDOW:4:15", "MINLEN:36"
        ]
    else:
        # Single-end mode
        cmd = [
            "trimmomatic", "SE", "-phred33", "-threads", N_THREADS,
            r1_in, r1_out,
            "LEADING:3", "TRAILING:3", "SLIDINGWINDOW:4:15", "MINLEN:36"
        ]
    print("🧹 Step 3: Running Trimmomatic...")
    print(f"⚙️ Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    return (result.stdout or "") + "\n" + (result.stderr or "")



@function_tool
def run_salmon(
    fastq_1: str,
    fastq_2: str = None,
    output_dir: str = "results/salmon_quant",
    index_path: str = INDEX_PATH
) -> str:
    """
    Run Salmon quantification on FASTQ files.

    Args:
        fastq_1 (str): Path to read 1 FASTQ file (required).
        fastq_2 (str, optional): Path to read 2 FASTQ file (if paired-end).
        output_dir (str): Output directory for Salmon quant results.
        index_path (str): Path to pre-built Salmon index directory.
            Defaults to INDEX_PATH.
    Returns:
        str: JSON with keys: status, output_dir, quant_files.
              On error, returns JSON with error details.
    Notes:
        - Salmon must be installed and callable from PATH.
        - Assumes transcriptome index is pre-built at the given `index_path`.
        - Automatically creates output directory if needed.
        - Designed for use inside autonomous agentic pipelines.
    """
    print("🐟 Step 4: Running Salmon quantification...")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "salmon", "quant",
        "-i", index_path,
        "-l", "A",
        "-o", output_dir,
        "--threads", N_THREADS
    ]
    if fastq_2:
        cmd += ["-1", fastq_1, "-2", fastq_2]
    else:
        cmd += ["-r", fastq_1]
    print(f"[Salmon] Running: {' '.join(cmd)}")
    # Stream Salmon output live
    proc = subprocess.Popen(cmd, stdout=None, stderr=None)
    proc.communicate()
    return f"[Salmon] Finished quantification for {fastq_1}"




@function_tool
def run_multiqc(input_dir: str, output_dir: str) -> str:
    """
    Runs MultiQC to summarize all QC and quantification reports in a directory.
    Args:
        input_dir (str): Directory to scan for reports (e.g., results/<disease_name>_fastq).
        output_dir (str): Where to save the MultiQC HTML report.
    Returns:
        str: Path to the MultiQC HTML report.
    Notes:
        - MultiQC must be installed and callable from the environment.
        - Scans recursively for FastQC, Trimmomatic, Salmon, etc reports.
        - Recommended to run at the end, after all processing is complete.
    """
    import subprocess, os
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["multiqc", input_dir, "-o", output_dir]
    print(f"[MultiQC] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    html_report = os.path.join(output_dir, "multiqc_report.html")
    if os.path.exists(html_report):
        print(f"[MultiQC] MultiQC report created: {html_report}")
        return html_report
    else:
        print(f"[MultiQC][WARN] MultiQC did not create report as expected.")
        return result.stdout + "\n" + result.stderr



from agents import function_tool
from pathlib import Path
import pandas as pd
import os
import re


@function_tool
def convert_sample_transcripts_to_genes(
    transcript_counts_path: str,
    output_dir: str,
    sample_name: str,
    tx2gene_path: str = INDEX_PATH + "/tx2gene.tsv"
) -> str:
    """
    Converts a *single-sample* transcript-level counts CSV to a gene-level (HGNC) counts CSV.

    Default tx2gene_path is hard-coded for EC2:
      INDEX_PATH + "/tx2gene.tsv"

    You can still override tx2gene_path manually if needed.

    Safety:
      - If tx2gene missing OR no matches OR any error:
        logs warning and returns warning string (does not raise),
        so the agent continues processing all samples.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        transcript_counts_path = Path(transcript_counts_path)
        tx2gene_path = Path(tx2gene_path)

        # ---- Validate files ----
        if not transcript_counts_path.exists():
            msg = f"[convert_sample_transcripts_to_genes][WARN] counts file not found: {transcript_counts_path}. Skipping gene conversion for {sample_name}."
            print(msg)
            return msg

        if not tx2gene_path.exists():
            msg = f"[convert_sample_transcripts_to_genes][WARN] tx2gene.tsv not found: {tx2gene_path}. Skipping gene conversion for {sample_name}."
            print(msg)
            return msg

        # ---- Load transcript counts ----
        df = pd.read_csv(transcript_counts_path)

        # Case A: transcript IDs in "Name" column
        if "Name" in df.columns:
            count_cols = [c for c in df.columns if c != "Name"]
            if len(count_cols) != 1:
                msg = f"[convert_sample_transcripts_to_genes][WARN] Unexpected columns in {transcript_counts_path}: {df.columns.tolist()}. Skipping {sample_name}."
                print(msg)
                return msg

            counts_col = count_cols[0]
            df = df.set_index("Name")
            df = df.rename(columns={counts_col: sample_name})

        # Case B: transcript IDs already index
        else:
            if df.columns[0].lower().startswith("unnamed"):
                df = df.set_index(df.columns[0])

            if df.shape[1] != 1:
                msg = f"[convert_sample_transcripts_to_genes][WARN] Expected 1 counts column in {transcript_counts_path}, found {df.shape[1]}. Skipping {sample_name}."
                print(msg)
                return msg

            df.columns = [sample_name]

        # ---- Strip ENST version suffix ----
        df.index = df.index.astype(str).str.replace(r"\.\d+$", "", regex=True)

        # ---- Load tx2gene ----
        tx2gene = pd.read_csv(tx2gene_path, sep="\t")

        possible_tx_cols = ["transcript_id", "transcript", "tx_id", "Name"]
        possible_gene_cols = ["gene_name", "gene", "symbol", "hgnc_symbol"]

        tx_col = next((c for c in possible_tx_cols if c in tx2gene.columns), None)
        gene_col = next((c for c in possible_gene_cols if c in tx2gene.columns), None)

        if tx_col is None or gene_col is None:
            msg = f"[convert_sample_transcripts_to_genes][WARN] tx2gene missing required cols. Found: {tx2gene.columns.tolist()}. Skipping {sample_name}."
            print(msg)
            return msg

        tx2gene = tx2gene[[tx_col, gene_col]].copy()
        tx2gene[tx_col] = tx2gene[tx_col].astype(str).str.replace(r"\.\d+$", "", regex=True)

        # ---- Merge counts with mapping ----
        merged = df.merge(
            tx2gene,
            left_index=True,
            right_on=tx_col,
            how="inner"
        )

        if merged.empty:
            msg = f"[convert_sample_transcripts_to_genes][WARN] No matching transcripts found in tx2gene for {sample_name}. Skipping gene conversion."
            print(msg)
            return msg

        # ---- Group by gene symbol & sum ----
        gene_counts = merged.groupby(gene_col)[sample_name].sum()

        out_path = Path(output_dir) / f"counts_gene_{sample_name}.csv"
        gene_counts.to_frame(name=sample_name).to_csv(out_path)

        print(f"[convert_sample_transcripts_to_genes] ✅ Saved gene counts: {out_path}")
        return str(out_path)

    except Exception as e:
        msg = f"[convert_sample_transcripts_to_genes][WARN] Conversion failed for {sample_name}: {e}. Skipping."
        print(msg)
        return msg





@function_tool
def detect_and_generate_metadata_from_sra(
    counts_matrix_path: str,
    fastq_path: str
) -> str:
    """
    Detects an SRA/metadata table in the FASTQ input folder and generates
    metadata_from_sra.csv aligned to the counts matrix.

    Detection rules:
    - Looks in the *folder of the FASTQ file* for any file whose name contains
      'meta' or 'metadata' (case-insensitive), OR common SRA run table patterns.
    - Supports .csv, .tsv, .txt.
    - Tries multiple possible sample / run columns (Run, run_accession, sample, etc.).

    Args:
        counts_matrix_path (str): Path to combined HGNC gene matrix
            e.g., results/<disease>_fastq/combined/<disease>_gene_counts.csv
        fastq_path (str): Path to any FASTQ file from jobs (fastq_1 is fine).

    Returns:
        str: Path to metadata_from_sra.csv if generated, else "" (empty string).
    """
    import os
    import pandas as pd
    from pathlib import Path

    print(f"[detect_and_generate_metadata_from_sra] Starting...")
    print(f"[detect_and_generate_metadata_from_sra] counts: {counts_matrix_path}")
    print(f"[detect_and_generate_metadata_from_sra] fastq:  {fastq_path}")

    # ---------------------------
    # 0) Locate FASTQ input folder
    # ---------------------------
    fastq_path = Path(fastq_path)
    input_dir = fastq_path.parent

    # ---------------------------
    # 1) Detect metadata file
    # ---------------------------
    # More aggressive patterns: any file with "meta" or "metadata" in its name
    # + common SRA/RunTable patterns.
    patterns = [
        "*meta*.csv", "*meta*.tsv", "*meta*.txt",
        "*Meta*.csv", "*Meta*.tsv", "*Meta*.txt",
        "*META*.csv", "*META*.tsv", "*META*.txt",
        "*metadata*.csv", "*metadata*.tsv", "*metadata*.txt",
        "*Metadata*.csv", "*Metadata*.tsv", "*Metadata*.txt",
        "*SraRunTable*.csv", "*SraRunTable*.tsv",
        "*run*table*.csv", "*RunTable*.csv", "*runinfo*.csv",
        "*sra*table*.csv", "*.runs.csv"
    ]

    candidates = []
    for pat in patterns:
        candidates.extend(input_dir.glob(pat))

    # De-duplicate and sort by modification time (newest first)
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)

    if not candidates:
        print(f"[detect_and_generate_metadata_from_sra] No metadata-like file found in: {input_dir}")
        return ""

    sra_table_path = candidates[0]
    print(f"[detect_and_generate_metadata_from_sra] Using metadata file: {sra_table_path}")

    # ---------------------------
    # 2) Load counts matrix
    # ---------------------------
    if not os.path.exists(counts_matrix_path):
        raise FileNotFoundError(f"Counts matrix not found at: {counts_matrix_path}")

    counts_df = pd.read_csv(counts_matrix_path, index_col=0)
    processed_samples = counts_df.columns.tolist()
    print(f"[detect_and_generate_metadata_from_sra] Found {len(processed_samples)} samples in counts.")

    # ---------------------------
    # 3) Load metadata/SRA table robustly
    # ---------------------------
    if not sra_table_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {sra_table_path}")

    try:
        sra_df = pd.read_csv(sra_table_path)
        # If it looks malformed (only 1 column or missing any plausible id col), try alt separators
        if len(sra_df.columns) == 1:
            print("[detect_and_generate_metadata_from_sra] Retrying with TAB separator...")
            sra_df = pd.read_csv(sra_table_path, sep="\t")
        if len(sra_df.columns) == 1:
            print("[detect_and_generate_metadata_from_sra] Retrying with whitespace separator...")
            sra_df = pd.read_csv(sra_table_path, sep=r"\s+")
    except Exception as e:
        raise RuntimeError(f"Failed to read metadata/SRA file: {e}")

    # ---------------------------
    # 4) Find the 'run/sample' column that matches counts
    # ---------------------------
    possible_run_cols = [
        "Run", "run", "run_accession", "Run accession", "Run_accession",
        "sample", "Sample", "sample_name", "Sample Name", "Sample_Name",
        "Run ID", "Run_ID", "SRR", "srr"
    ]

    run_col = None
    for col in sra_df.columns:
        if col in possible_run_cols:
            run_col = col
            break

    if run_col is None:
        # Try fuzzy match: any column whose values intersect counts sample IDs
        for col in sra_df.columns:
            if sra_df[col].astype(str).isin(processed_samples).any():
                run_col = col
                print(f"[detect_and_generate_metadata_from_sra] Using column '{col}' as run/sample id (detected by intersection).")
                break

    if run_col is None:
        raise ValueError(
            "[detect_and_generate_metadata_from_sra] Could not find a suitable run/sample column.\n"
            f"Columns found: {list(sra_df.columns)}"
        )

    print(f"[detect_and_generate_metadata_from_sra] Using '{run_col}' as run/sample column.")

    # ---------------------------
    # 5) Match rows to counts samples
    # ---------------------------
    sra_df[run_col] = sra_df[run_col].astype(str)
    matched_sra = sra_df[sra_df[run_col].isin(processed_samples)].copy()

    if matched_sra.empty:
        raise RuntimeError(
            "[detect_and_generate_metadata_from_sra] No matching IDs between metadata file and counts matrix.\n"
            f"Counts samples (first 5): {processed_samples[:5]}\n"
            f"{run_col} values (first 5): {sra_df[run_col].head().tolist()}"
        )

    # ---------------------------
    # 6) Auto-detect condition column
    # ---------------------------
    potential_condition_cols = [
        "patient", "Patient", "subject", "Subject",
        "source_name", "source name",
        "cell_type", "Cell Type",
        "tissue", "Tissue",
        "treatment", "Treatment",
        "disease_state", "Disease State",
        "Group", "group", "condition", "Condition"
    ]

    found_col = None
    for col in potential_condition_cols:
        if col in matched_sra.columns and matched_sra[col].nunique() > 1:
            found_col = col
            print(f"[detect_and_generate_metadata_from_sra] Detected condition column: {col}")
            break

    # ---------------------------
    # 7) Build metadata dataframe
    # ---------------------------
    metadata = pd.DataFrame()
    metadata["sample"] = matched_sra[run_col].astype(str)

    if found_col:
        metadata["condition"] = matched_sra[found_col].astype(str)
    else:
        metadata["condition"] = "fill_this_in"
        print("[detect_and_generate_metadata_from_sra] WARN: condition not detected, using placeholder.")

    metadata["replicate"] = metadata.groupby("condition").cumcount() + 1

    # Reorder rows to match counts matrix column order
    metadata = metadata.set_index("sample")
    valid_samples = [s for s in processed_samples if s in metadata.index]
    metadata = metadata.reindex(valid_samples).reset_index()

    # ---------------------------
    # 8) Save next to counts
    # ---------------------------
    out_path = os.path.join(os.path.dirname(counts_matrix_path), "metadata_from_sra.csv")
    metadata.to_csv(out_path, index=False)

    print(f"[detect_and_generate_metadata_from_sra] ✅ Saved metadata: {out_path}")
    return out_path
