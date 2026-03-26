from __future__ import annotations
#!/usr/bin/env python3
"""
prepare_samplesheet3.py (LATEST)

KEY FEATURES:
1) TEMPLATE IS OPTIONAL:
   - Script does NOT force template prompts.
   - Output always includes a 'template' column, but it can be blank.

2) START BANNER:
   - Prints big green ASCII banner: "CRISPR TARGETED PIPELINE"

3) OUTPUT BOTH TSV + CSV:
   - Writes samplesheet_final.tsv
   - Writes samplesheet_final.csv

4) USER SAMPLESHEET VALIDATION:
   - Accepts TSV with required columns:
       sample, fastq_1, fastq_2, reference, protospacer
     template column is OPTIONAL (auto-added blank if missing).
"""

import os
import sys
import re
import csv
import json
import gzip
import shutil
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

# ----------------------------
# PACKAGE-BUNDLED REFERENCE FILES
# ----------------------------
_REFERENCE_DIR = Path("targeted_crispr_reference_data")
_DEFAULT_HG38 = str(_REFERENCE_DIR / "hg38.fa")
_DEFAULT_GTF = str(_REFERENCE_DIR / "gencode.v44.annotation.gtf")

# ----------------------------
# OPTIONAL requests FOR LLM
# ----------------------------
try:
    import requests  # type: ignore
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

# ----------------------------
# ANSI STYLES (BOLD + COLORS)
# ----------------------------
BOLD = "\033[1m"
RESET = "\033[0m"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"

BOLD_GREEN = "\033[1;92m"
BOLD_YELLOW = "\033[1;93m"
BOLD_RED = "\033[1;91m"

def g(msg: str) -> str: return f"{BOLD_GREEN}{msg}{RESET}"
def y(msg: str) -> str: return f"{BOLD_YELLOW}{msg}{RESET}"
def r(msg: str) -> str: return f"{BOLD_RED}{msg}{RESET}"
def b(msg: str) -> str: return f"{BOLD}{msg}{RESET}"

def tick(msg: str) -> None:
    print(f"{BOLD_GREEN}[✓]{RESET} {msg}")

def info(msg: str) -> None:
    print(f"{BOLD_YELLOW}[INFO]{RESET} {msg}")

def warn(msg: str) -> None:
    print(f"{BOLD_YELLOW}[WARN]{RESET} {msg}")

def err(msg: str, code: int = 1) -> None:
    print(f"{BOLD_RED}[ERROR]{RESET} {msg}")
    sys.exit(code)

def banner_started() -> None:
    banner = r"""
CRISPR TARGETED PIPELINE

"""
    print(f"{BOLD_GREEN}{banner}{RESET}")
    print(g("[STARTED PIPELINE]"))
    print("")

# ----------------------------
# FASTQ SCANNING
# ----------------------------
FASTQ_RE = re.compile(r"^(?P<sample>.+?)_(?P<mate>[12])\.fastq\.gz$")

def scan_fastqs(fastq_dir: Path):
    if not fastq_dir.exists():
        err(f"FASTQ DIRECTORY NOT FOUND: {fastq_dir}")

    samples = defaultdict(lambda: {"fastq_1": "", "fastq_2": ""})
    for fn in sorted(os.listdir(fastq_dir)):
        if not fn.endswith(".fastq.gz"):
            continue
        m = FASTQ_RE.match(fn)
        if not m:
            continue
        sample = m.group("sample")
        mate = m.group("mate")
        if mate == "1":
            samples[sample]["fastq_1"] = str((fastq_dir / fn).as_posix())
        elif mate == "2":
            samples[sample]["fastq_2"] = str((fastq_dir / fn).as_posix())

    rows = []
    for sample, mates in samples.items():
        rows.append({"sample": sample, "fastq_1": mates["fastq_1"], "fastq_2": mates["fastq_2"]})

    pe = sum(1 for r in rows if r["fastq_1"] and r["fastq_2"])
    se = sum(1 for r in rows if r["fastq_1"] and not r["fastq_2"])
    missing_r1 = sum(1 for r in rows if not r["fastq_1"])
    return sorted(rows, key=lambda x: x["sample"]), pe, se, missing_r1

# ----------------------------
# SAFE INPUT HELPERS
# ----------------------------
def ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    suffix = " (Y/n): " if default_yes else " (y/N): "
    v = input(prompt + suffix).strip().lower()
    if not v:
        return default_yes
    return v in ("y", "yes")

def ask_choice(prompt: str, choices):
    choices_str = "/".join(choices)
    while True:
        v = input(f"{prompt} ({choices_str}): ").strip().upper()
        if v in choices:
            return v
        print(f"PLEASE CHOOSE ONE OF: {choices_str}")

def ask_block_triple_quotes(prompt: str):
    print(b(prompt))
    print(b('PASTE NOW. FINISH BY TYPING A LINE CONTAINING ONLY: """'))
    lines = []
    started = False
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not started:
            if line.strip() == '"""':
                started = True
                continue
            started = True
            lines.append(line)
        else:
            if line.strip() == '"""':
                break
            lines.append(line)
    return "\n".join(lines)

def parse_sequence_from_text(text: str) -> str:
    seq_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            continue
        seq_lines.append(line)
    seq = "".join(seq_lines)
    seq = seq.replace(",", "").replace(" ", "").replace("\t", "")
    seq = seq.upper()
    seq = re.sub(r"[^ACGTN]", "", seq)
    return seq

# ----------------------------
# LLM SYSTEM PROMPTS
# ----------------------------
LLM_PARSE_SYSTEM = """You are a bioinformatics assistant.
You will receive a pasted block that contains:
- gRNA table: target/gene name and gRNA sequence (ACGT)
- primer table: target/gene name with ON TARGET forward and reverse primers (ACGT)
Return strict JSON only in this schema:

{
  "grnas": [{"target":"RAB11A","sequence":"GGT..."}],
  "primers": [{
    "target":"RAB11A",
    "on_target": {"forward":"ACCC...","reverse":"AACC..."}
  }]
}

Rules:
- Uppercase sequences (A/C/G/T only).
- Do NOT include commentary or markdown.
"""

LLM_CONTROL_SYSTEM = """You are a bioinformatics assistant.
You will receive runs metadata (CSV-derived fields) for sequencing runs.
Classify which runs are CONTROLS/NEGATIVE/WT/NC/no-guide samples.
Return strict JSON only:

{
  "controls": ["SRR...","SRR..."],
  "non_controls": ["SRR...","SRR..."],
  "reason": {"SRR...":"short reason", "...":"..."}
}

Rules:
- A run is control if sample name suggests NC, WT, untreated, no-guide, negative control, etc.
- If uncertain, put it in non_controls.
- No extra text.
"""

def call_openai_json(messages, api_key, model="gpt-4.1-mini", timeout=60):
    if not HAS_REQUESTS:
        raise RuntimeError("requests NOT INSTALLED. INSTALL: pip install requests")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)

    if resp.status_code != 200:
        snippet = resp.text[:300].replace("\n", " ")
        raise RuntimeError(f"OPENAI API FAILED: HTTP {resp.status_code}. RESPONSE: {snippet}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)

def read_openai_key(env_key: str, key_file: str | None, project_dir: Path) -> str:
    k = os.environ.get(env_key, "").strip()
    if k:
        return k

    candidates = []
    if key_file:
        candidates.append(Path(key_file))
    candidates.extend([
        Path("OpenAIkey.txt"),
        Path("openai_key.txt"),
        project_dir / "OpenAIkey.txt",
        project_dir / "openai_key.txt",
    ])

    for p in candidates:
        try:
            if p.exists():
                txt = p.read_text().strip()
                if txt:
                    return txt
        except Exception:
            continue
    return ""

# ----------------------------
# REGEX FALLBACK (ROBUST)
# ----------------------------
STOP_TOKENS = {
    "SGRNA", "TARGET", "SEQUENCE", "PRIMERS", "PRIMER", "FORWARD", "REVERSE",
    "ON", "OFF", "TABLE", "SUPPLEMENTAL"
}

def fallback_parse_block(text: str):
    grnas = []
    primers = []

    for line in text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        parts = re.split(r"\s+", ln)
        if len(parts) < 2:
            continue
        gene = parts[0].strip().upper()
        if gene in STOP_TOKENS:
            continue
        seq = parts[1].strip().upper()
        if re.fullmatch(r"[ACGT]{18,25}", seq):
            grnas.append({"target": gene, "sequence": seq})

    current = None
    on = {}
    for line in text.splitlines():
        u = line.strip()
        if not u:
            continue
        m_on = re.match(r"^([A-Za-z0-9_-]+)\s+ON\s+TARGET", u, flags=re.I)
        if m_on:
            if current and on.get("forward") and on.get("reverse"):
                primers.append({"target": current, "on_target": {"forward": on["forward"], "reverse": on["reverse"]}})
            current = m_on.group(1).upper()
            on = {}
            continue
        m_pr = re.match(r"^(forward|reverse)\s+([ACGTacgt]{10,})$", u, flags=re.I)
        if m_pr and current:
            kind = m_pr.group(1).lower()
            seq = re.sub(r"[^ACGT]", "", m_pr.group(2).upper())
            on[kind] = seq

    if current and on.get("forward") and on.get("reverse"):
        primers.append({"target": current, "on_target": {"forward": on["forward"], "reverse": on["reverse"]}})

    seen = set()
    grnas2 = []
    for g0 in grnas:
        if g0["target"] not in seen:
            seen.add(g0["target"])
            grnas2.append(g0)

    return {"grnas": grnas2, "primers": primers}

# ----------------------------
# FASTA UTILITIES (pysam)
# ----------------------------
try:
    import pysam  # type: ignore
    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False

def ensure_fai(fasta_path: Path):
    fai = Path(str(fasta_path) + ".fai")
    if fai.exists():
        return
    info(f"FASTA INDEX NOT FOUND. CREATING: {fai.name}")
    if HAS_PYSAM:
        pysam.faidx(str(fasta_path))
    else:
        subprocess.run(["samtools", "faidx", str(fasta_path)], check=True)

def fetch_seq(fa: Path, region: str) -> str:
    if HAS_PYSAM:
        with pysam.FastaFile(str(fa)) as fasta:
            return fasta.fetch(region=region).upper()
    out = subprocess.check_output(["samtools", "faidx", str(fa), region], text=True)
    seq = "".join([ln.strip() for ln in out.splitlines() if not ln.startswith(">")])
    return seq.upper()

def normalize_region(user_input: str) -> str:
    s = user_input.strip()
    if not s:
        err("EMPTY REGION INPUT")

    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+TO\s+", "-", s, flags=re.I)
    s = s.replace(" ", "")
    u = s.upper()

    if u.startswith("CHR"):
        u = "chr" + u[3:]
    elif re.match(r"^\d+:", u):
        u = "chr" + u

    u = re.sub(r"TO", "-", u)

    m = re.match(r"^(chr[\w]+):(\d+)-(\d+)$", u, flags=re.I)
    if not m:
        err(f"INVALID REGION FORMAT. EXPECT chr:start-end. RECEIVED: {user_input}")

    chrom = m.group(1)
    start = int(m.group(2))
    end = int(m.group(3))
    if start <= 0 or end <= 0 or end < start:
        err(f"INVALID REGION COORDINATES AFTER NORMALIZATION: {chrom}:{start}-{end}")

    return f"{chrom}:{start}-{end}"

# ----------------------------
# GTF GENE COORDS (GENE ONLY)
# ----------------------------
def read_text_maybe_gz(p: Path) -> str:
    if str(p).endswith(".gz"):
        with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    return p.read_text(encoding="utf-8", errors="replace")

def load_gtf_gene_coords(gtf_path: Path):
    gene_map = {}
    gtf_text = read_text_maybe_gz(gtf_path)

    for line in gtf_text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        chrom, source, feature, start, end, score, strand, frame, attrs = parts
        if feature != "gene":
            continue

        gene_name = None
        m = re.search(r'gene_name "([^"]+)"', attrs)
        if m:
            gene_name = m.group(1)
        if not gene_name:
            m = re.search(r'gene_id "([^"]+)"', attrs)
            if m:
                gene_name = m.group(1)

        if not gene_name:
            continue

        gene_map[gene_name.upper()] = (chrom, int(start), int(end), strand)

    return gene_map

def clamp_flank(flank: int) -> int:
    if flank < 0:
        flank = 0
    if flank > 100:
        flank = 100
    return flank

# ----------------------------
# runs.csv / runs.tsv LOADING + CONTROL CLASSIFICATION
# ----------------------------
def find_runs_file(project_dir: Path) -> Path | None:
    candidates = [
        project_dir / "runs.csv",
        project_dir / "runs.tsv",
        project_dir / "runs.txt",
        project_dir.parent / "runs.csv",
        project_dir.parent / "runs.tsv",
        project_dir.parent / "runs.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def load_runs_table(path: Path):
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)

    if not rows or "Run" not in rows[0]:
        err(f"RUNS FILE INVALID (MISSING 'Run' COLUMN): {path}")

    out = {}
    for r0 in rows:
        out[r0["Run"].strip()] = r0
    return out

def heuristic_control(run_row: dict) -> bool:
    fields = [
        run_row.get("LibraryName", ""),
        run_row.get("SampleName", ""),
        run_row.get("Experiment", ""),
        run_row.get("Sample", ""),
    ]
    text = " ".join(fields).upper()
    patterns = [
        " WT", "WT ", "NC", "NEGATIVE", "CONTROL", "UNTREATED", "NO GUIDE", "NOGUIDE",
        "RNPWT", "CAS9WT", "WILDTYPE", "NOTAPPLICABLE"
    ]
    if re.search(r"\bNC\b", text):
        return True
    if any(p in text for p in patterns):
        return True
    return False

def llm_classify_controls(run_rows: dict, api_key: str, model: str):
    payload = []
    for run_id, row in run_rows.items():
        payload.append({
            "Run": run_id,
            "LibraryName": row.get("LibraryName", ""),
            "SampleName": row.get("SampleName", ""),
            "Experiment": row.get("Experiment", ""),
            "Sample": row.get("Sample", ""),
            "LibraryStrategy": row.get("LibraryStrategy", ""),
            "LibraryLayout": row.get("LibraryLayout", ""),
        })

    messages = [
        {"role": "system", "content": LLM_CONTROL_SYSTEM},
        {"role": "user", "content": json.dumps(payload)}
    ]
    return call_openai_json(messages, api_key=api_key, model=model, timeout=60)

# ----------------------------
# SAMPLESHEET VALIDATION (TEMPLATE OPTIONAL)
# ----------------------------
REQUIRED_BASE_COLS = ["sample", "fastq_1", "fastq_2", "reference", "protospacer"]
OPTIONAL_COLS = ["template"]
CANONICAL_COLS = REQUIRED_BASE_COLS + OPTIONAL_COLS

def validate_samplesheet_tsv(path: Path) -> tuple[bool, str, bool]:
    """
    Returns: (ok, msg, has_template)
    - Requires TSV header with tabs.
    - Requires REQUIRED_BASE_COLS.
    - template is optional.
    """
    if not path.exists():
        return False, f"FILE NOT FOUND: {path}", False

    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not text:
        return False, "FILE IS EMPTY", False

    header = text[0].strip()
    if "\t" not in header:
        return False, "HEADER MUST BE TSV (TAB-SEPARATED).", False

    cols = [c.strip() for c in header.split("\t")]
    missing = [c for c in REQUIRED_BASE_COLS if c not in cols]
    if missing:
        return False, f"MISSING REQUIRED COLUMNS: {missing}", False

    if len(text) < 2:
        return False, "NO DATA ROWS FOUND (ONLY HEADER PRESENT).", ("template" in cols)

    return True, "OK", ("template" in cols)

def normalize_samplesheet_to_project(src_tsv: Path, dest_tsv: Path, dest_csv: Path) -> None:
    """
    Reads user TSV (template optional), writes canonical TSV + CSV (template always present; blank if missing).
    """
    ok, msg, has_template = validate_samplesheet_tsv(src_tsv)
    if not ok:
        err(f"SAMPLESHEET VALIDATION FAILED: {msg}")

    with open(src_tsv, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    # Ensure all canonical columns exist
    norm_rows = []
    for r0 in rows:
        out = {}
        for c in REQUIRED_BASE_COLS:
            out[c] = (r0.get(c, "") or "").strip()
        out["template"] = (r0.get("template", "") or "").strip() if has_template else ""
        norm_rows.append(out)

    dest_tsv.parent.mkdir(parents=True, exist_ok=True)

    # Write TSV
    with open(dest_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANONICAL_COLS, delimiter="\t")
        w.writeheader()
        w.writerows(norm_rows)

    # Write CSV
    with open(dest_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANONICAL_COLS, delimiter=",")
        w.writeheader()
        w.writerows(norm_rows)

# ----------------------------
# OUTPUT WRITERS (TSV + CSV)
# ----------------------------
def write_outputs(rows_out: list[dict], tsv_path: Path, csv_path: Path) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANONICAL_COLS, delimiter="\t")
        w.writeheader()
        w.writerows(rows_out)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANONICAL_COLS, delimiter=",")
        w.writeheader()
        w.writerows(rows_out)

# ----------------------------
# MAIN
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="PREPARE nf-core/crisprseq SAMPLESHEET (CONTROL-AWARE, USER-FRIENDLY). TEMPLATE OPTIONAL. TSV+CSV OUTPUT."
    )
    ap.add_argument("project_dir", help="PROJECT DIR, e.g. extracted_data/extracted_data/PRJNA1240319")
    ap.add_argument("--fastq-subdir", default="fastq", help="FASTQ SUBFOLDER INSIDE project_dir (DEFAULT: fastq)")
    ap.add_argument("--hg38", default=_DEFAULT_HG38, help=f"PATH TO hg38 FASTA (DEFAULT: {_DEFAULT_HG38})")
    ap.add_argument("--gtf", default=_DEFAULT_GTF, help=f"PATH TO GTF (DEFAULT: {_DEFAULT_GTF})")
    ap.add_argument("--out", default="samplesheet_final", help="OUTPUT BASENAME INSIDE project_dir (DEFAULT: samplesheet_final). Writes .tsv and .csv")
    ap.add_argument("--gene-flank", type=int, default=0, help="OPTIONAL SMALL FLANK AROUND GENE (MAX 100). DEFAULT: 0")
    ap.add_argument("--openai-model", default="gpt-4.1-mini", help="OPENAI MODEL (DEFAULT: gpt-4.1-mini)")
    ap.add_argument("--openai-key-file", default=None, help="PATH TO KEY FILE (OPTIONAL). EXAMPLE: OpenAIkey.txt")
    ap.add_argument("--no-llm", action="store_true", help="DISABLE LLM EVEN IF KEY IS PRESENT")
    ap.add_argument("--samplesheet", default=None, help="DIRECT TSV SAMPLESHEET PATH (SKIP PROMPTS). template column optional.")
    ap.add_argument("--no-interactive", action="store_true", help="DO NOT PROMPT. REQUIRES --samplesheet OR FAILS.")
    args = ap.parse_args()

    banner_started()

    project_dir = Path(args.project_dir)
    if not project_dir.exists():
        err(f"PROJECT DIRECTORY DOES NOT EXIST: {project_dir}")

    # normalize out basename
    out_base = Path(args.out).name
    if out_base.endswith(".tsv") or out_base.endswith(".csv"):
        out_base = out_base.rsplit(".", 1)[0]

    out_tsv = project_dir / f"{out_base}.tsv"
    out_csv = project_dir / f"{out_base}.csv"

    print(g("SAMPLESHEET FORMAT (TSV):"))
    print(b("Required columns: sample, fastq_1, fastq_2, reference, protospacer"))
    print(b("Optional column:  template"))
    print(b("Canonical output columns: sample\tfastq_1\tfastq_2\treference\tprotospacer\ttemplate"))
    print("")

    # ----------------------------
    # NON-INTERACTIVE MODE
    # ----------------------------
    if args.no_interactive:
        if not args.samplesheet:
            err("NO-INTERACTIVE MODE REQUIRES --samplesheet")
        src = Path(args.samplesheet)
        normalize_samplesheet_to_project(src, out_tsv, out_csv)
        tick(f"WRITTEN TSV: {out_tsv.as_posix()}")
        tick(f"WRITTEN CSV: {out_csv.as_posix()}")
        print("")
        print(g("[PIPELINE COMPLETED]"))
        print(g(f"[CHECK CONFIGURED FILE] {out_tsv.as_posix()}"))
        print(g(f"[CHECK CONFIGURED FILE] {out_csv.as_posix()}"))
        return

    # ----------------------------
    # DIRECT SAMPLESHEET PARAM (SKIP PROMPTS)
    # ----------------------------
    if args.samplesheet:
        src = Path(args.samplesheet)
        normalize_samplesheet_to_project(src, out_tsv, out_csv)
        tick(f"WRITTEN TSV: {out_tsv.as_posix()}")
        tick(f"WRITTEN CSV: {out_csv.as_posix()}")
        print("")
        print(g("[PIPELINE COMPLETED]"))
        print(g(f"[CHECK CONFIGURED FILE] {out_tsv.as_posix()}"))
        print(g(f"[CHECK CONFIGURED FILE] {out_csv.as_posix()}"))
        return

    # ----------------------------
    # ASK: USE OWN SAMPLESHEET?
    # ----------------------------
    if ask_yes_no("DO YOU WANT TO USE/UPLOAD YOUR OWN SAMPLESHEET TSV FILE PATH NOW? (template column optional)", default_yes=True):
        user_path = input("ENTER FULL PATH TO YOUR TSV SAMPLESHEET: ").strip()
        src = Path(user_path)
        normalize_samplesheet_to_project(src, out_tsv, out_csv)
        tick(f"WRITTEN TSV: {out_tsv.as_posix()}")
        tick(f"WRITTEN CSV: {out_csv.as_posix()}")
        print("")
        print(g("[PIPELINE COMPLETED]"))
        print(g(f"[CHECK CONFIGURED FILE] {out_tsv.as_posix()}"))
        print(g(f"[CHECK CONFIGURED FILE] {out_csv.as_posix()}"))
        return

    print("OKAY.")

    # ----------------------------
    # ASK: CREATE SAMPLESHEET?
    # ----------------------------
    if not ask_yes_no("DO YOU WANT ME TO CREATE A SAMPLESHEET FOR YOU NOW?", default_yes=True):
        err("USER CHOSE NOT TO PROVIDE OR CREATE A SAMPLESHEET. EXITING.")

    # ----------------------------
    # FASTQ SCAN + COUNTS
    # ----------------------------
    fastq_dir = project_dir / args.fastq_subdir
    rows, pe, se, missing_r1 = scan_fastqs(fastq_dir)

    tick(f"DETECTED {len(rows)} SAMPLES")
    tick(f"PAIRED-END: {pe}")
    tick(f"SINGLE-END: {se}")
    if missing_r1:
        warn(f"{missing_r1} SAMPLES MISSING R1 (CHECK NAMING *_1.fastq.gz)")

    # ----------------------------
    # RUNS + CONTROL CLASSIFICATION (LLM OR HEURISTICS)
    # ----------------------------
    runs_path = find_runs_file(project_dir)
    runs_rows = {}
    controls = set()

    api_key = read_openai_key("OPENAI_API_KEY", args.openai_key_file, project_dir=project_dir)
    llm_ok = (not args.no_llm) and bool(api_key)

    if not api_key:
        warn("LLM IS DISABLED BECAUSE NO OPENAI API KEY WAS FOUND.")
        warn("FALLING BACK TO HEURISTICS/REGEX WHERE NEEDED.")
    elif args.no_llm:
        warn("LLM DISABLED BY USER (--no-llm). USING HEURISTICS/REGEX WHERE NEEDED.")

    if runs_path:
        info(f"FOUND RUNS FILE: {runs_path}")
        runs_rows = load_runs_table(runs_path)
        present_run_ids = {r0["sample"] for r0 in rows}
        runs_rows_present = {rid: rr for rid, rr in runs_rows.items() if rid in present_run_ids}

        if runs_rows_present:
            if llm_ok:
                info("CLASSIFYING CONTROLS WITH LLM (IF IT FAILS, HEURISTICS WILL BE USED)...")
                try:
                    out = llm_classify_controls(runs_rows_present, api_key=api_key, model=args.openai_model)
                    controls = set(out.get("controls", []) or [])
                    tick(f"LLM CONTROLS DETECTED: {len(controls)}")
                except Exception as e:
                    warn(f"LLM CONTROL CLASSIFICATION FAILED: {e}")
                    warn("FALLING BACK TO HEURISTICS.")
                    controls = {rid for rid, rr in runs_rows_present.items() if heuristic_control(rr)}
                    tick(f"HEURISTIC CONTROLS DETECTED: {len(controls)}")
            else:
                info("LLM UNAVAILABLE; USING HEURISTIC CONTROL DETECTION.")
                controls = {rid for rid, rr in runs_rows_present.items() if heuristic_control(rr)}
                tick(f"HEURISTIC CONTROLS DETECTED: {len(controls)}")
        else:
            warn("RUNS FILE FOUND BUT NONE OF ITS RUN IDS MATCH YOUR FASTQ SAMPLE IDS.")
    else:
        warn("NO runs.csv / runs.tsv FOUND. CONTROL DETECTION SKIPPED (NO CONTROLS).")

    # ----------------------------
    # REFERENCE STRATEGY MENU
    # ----------------------------
    print("""
REFERENCE STRATEGY
------------------
A) GENE-BASED (GENE ONLY)
   - PASTE gRNA TABLE BLOCK
   - SELECT TARGET GENE
   - SCRIPT FETCHES GENE REGION FROM GTF COORDS
   - template will be BLANK unless you later want to add it

B) REGION-BASED (chr:start-end)
   - ACCEPTS MESSY HUMAN INPUT AND NORMALIZES AUTOMATICALLY
   - EXTRACTS REGION SEQUENCE FROM hg38
   - template will be BLANK unless you later want to add it

C) USER-PROVIDED (PASTE SEQUENCE)
   - PASTE REFERENCE SEQUENCE BETWEEN TRIPLE QUOTES
   - PROVIDE PROTOSPACER (gRNA)
   - TEMPLATE IS OPTIONAL (you will be asked if you want to provide it)
""")

    mode = ask_choice("CHOOSE MODE", ["A", "B", "C"])

    hg38 = Path(args.hg38)
    if not hg38.exists():
        err(f"hg38 FASTA NOT FOUND: {hg38} (USE --hg38)")
    ensure_fai(hg38)

    reference_seq = ""
    template_seq = ""  # DEFAULT: BLANK (TEMPLATE OPTIONAL)
    protospacer = ""

    # MODE B
    if mode == "B":
        raw_region = input("ENTER REGION (chr:start-end) [MESSY INPUT ACCEPTED]: ").strip()
        region = normalize_region(raw_region)

        try:
            reference_seq = fetch_seq(hg38, region)
        except subprocess.CalledProcessError:
            err(f"FAILED TO FETCH SEQUENCE FOR REGION: {region}. CHECK CHROM NAME EXISTS IN FASTA (chr15 vs 15).")

        if not reference_seq:
            err(f"EMPTY SEQUENCE RETURNED FOR REGION: {region}")

        protospacer = input("PROTOSPACER (gRNA) SEQUENCE (ACGT...): ").strip().upper()
        protospacer = re.sub(r"[^ACGT]", "", protospacer)
        if not protospacer:
            err("PROTOSPACER IS EMPTY OR INVALID. PROVIDE A VALID ACGT SEQUENCE.")

        # template remains blank unless user later edits samplesheet manually

    # MODE C
    elif mode == "C":
        block = ask_block_triple_quotes("PASTE REFERENCE SEQUENCE (FASTA OR RAW) BETWEEN TRIPLE QUOTES:")
        reference_seq = parse_sequence_from_text(block)

        if not reference_seq:
            err("REFERENCE SEQUENCE IS EMPTY OR INVALID. PASTE A VALID FASTA/SEQUENCE.")

        protospacer = input("PROTOSPACER (gRNA) SEQUENCE (ACGT...): ").strip().upper()
        protospacer = re.sub(r"[^ACGT]", "", protospacer)
        if not protospacer:
            err("PROTOSPACER IS EMPTY OR INVALID. PROVIDE A VALID ACGT SEQUENCE.")

        # TEMPLATE OPTIONAL PROMPT (NO DEFAULT TEMPLATE)
        if ask_yes_no("DO YOU WANT TO PROVIDE A TEMPLATE SEQUENCE? (otherwise it will be blank)", default_yes=False):
            tmp_block = ask_block_triple_quotes("PASTE TEMPLATE SEQUENCE BETWEEN TRIPLE QUOTES:")
            tmp_seq = parse_sequence_from_text(tmp_block)
            if not tmp_seq:
                warn("TEMPLATE WAS PROVIDED BUT PARSED EMPTY. KEEPING TEMPLATE BLANK.")
                template_seq = ""
            else:
                template_seq = tmp_seq
        else:
            template_seq = ""

    # MODE A
    else:
        gtf = Path(args.gtf)
        if not gtf.exists():
            err(f"GTF NOT FOUND: {gtf} (USE --gtf)")

        info("INDEXING GENES FROM GTF (ONE-TIME PER RUN)...")
        gene_map = load_gtf_gene_coords(gtf)
        tick(f"GTF GENES INDEXED: {len(gene_map)}")

        block = ask_block_triple_quotes("PASTE gRNA TABLE BLOCK BETWEEN TRIPLE QUOTES (PRIMERS OPTIONAL):")

        parsed = None
        if llm_ok:
            info("PARSING gRNA BLOCK WITH LLM (IF IT FAILS, REGEX FALLBACK WILL BE USED)...")
            try:
                parsed = call_openai_json(
                    messages=[
                        {"role": "system", "content": LLM_PARSE_SYSTEM},
                        {"role": "user", "content": block},
                    ],
                    api_key=api_key,
                    model=args.openai_model,
                    timeout=60,
                )
            except Exception as e:
                warn(f"LLM PARSING FAILED: {e}")
                warn("FALLING BACK TO REGEX PARSER.")
                parsed = fallback_parse_block(block)
        else:
            warn("LLM UNAVAILABLE/DISABLED → USING REGEX PARSER")
            parsed = fallback_parse_block(block)

        grnas = parsed.get("grnas", []) or []
        if not grnas:
            err("NO gRNAs DETECTED. PASTE BLOCK AGAIN INCLUDING gRNA TABLE.")

        grna_map = {g0["target"].upper(): g0["sequence"].upper() for g0 in grnas if g0.get("target") and g0.get("sequence")}
        tick(f"DETECTED TARGETS (gRNA): {', '.join(sorted(grna_map.keys()))}")

        target = input("WHICH TARGET TO USE (GENE NAME EXACTLY LIKE RAB11A): ").strip().upper()
        if not target:
            err("NO TARGET PROVIDED.")
        if target not in grna_map:
            err(f"TARGET NOT FOUND IN gRNAs: {target}")

        protospacer = grna_map[target]

        if target not in gene_map:
            err(f"GENE NOT FOUND IN GTF: {target}. CHECK GENE SYMBOL OR USE MODE B/C.")

        chrom, gstart, gend, strand = gene_map[target]
        flank = clamp_flank(int(args.gene_flank))
        if flank:
            info(f"USING SMALL FLANK: {flank} BP (MAX 100 ENFORCED)")
        start = max(1, gstart - flank)
        end = gend + flank

        region = f"{chrom}:{start}-{end}"
        info(f"FETCHING GENE REGION (GENE ONLY): {region}")

        try:
            reference_seq = fetch_seq(hg38, region)
        except subprocess.CalledProcessError:
            err(f"FAILED TO FETCH GENE REGION: {region}. CHECK CHROM NAMING IN FASTA (chr15 vs 15).")

        if not reference_seq:
            err(f"EMPTY SEQUENCE RETURNED FOR GENE REGION: {region}")

        # template remains blank unless user later edits samplesheet manually

    # ----------------------------
    # BUILD OUTPUT ROWS
    # ----------------------------
    rows_out = []
    for i, r0 in enumerate(rows, 1):
        tick(f"WRITING {i} / {len(rows)}: {r0['sample']}")
        if r0["sample"] in controls:
            rows_out.append({
                "sample": r0["sample"],
                "fastq_1": r0["fastq_1"],
                "fastq_2": r0.get("fastq_2", ""),
                "reference": "",
                "protospacer": "",
                "template": "",
            })
        else:
            rows_out.append({
                "sample": r0["sample"],
                "fastq_1": r0["fastq_1"],
                "fastq_2": r0.get("fastq_2", ""),
                "reference": reference_seq,
                "protospacer": protospacer,
                "template": template_seq,  # can be blank
            })

    # write TSV + CSV
    write_outputs(rows_out, out_tsv, out_csv)
    tick(f"SAMPLESHEET WRITTEN TSV: {out_tsv.as_posix()}")
    tick(f"SAMPLESHEET WRITTEN CSV: {out_csv.as_posix()}")

    print("")
    print(g("[PIPELINE COMPLETED]"))
    print(g(f"[CHECK CONFIGURED FILE] {out_tsv.as_posix()}"))
    print(g(f"[CHECK CONFIGURED FILE] {out_csv.as_posix()}"))
    print(g("[IF YOU WANT TO RUN AGAIN PLEASE FEEL FREE TO RUN]"))


# ----------------------------
# NON-INTERACTIVE PUBLIC API
# ----------------------------
def generate_samplesheet(
    project_dir: str,
    protospacer: str,
    hg38: str = _DEFAULT_HG38,
    gtf: str = _DEFAULT_GTF,
    target_gene: str = "",
    region: str = "",
    reference_seq: str = "",
    template_seq: str = "",
    gene_flank: int = 0,
    fastq_subdir: str = "fastq",
    out_basename: str = "samplesheet_final",
    output_dir: str = "",
    openai_model: str = "gpt-4.1-mini",
    no_llm: bool = False,
    openai_key_file: str | None = None,
) -> tuple[str, str]:
    """Generate a samplesheet without any interactive prompts.

    Automatically scans FASTQs, classifies controls, resolves the
    reference sequence, and writes both TSV and CSV output files.

    How the reference is resolved (checked in order):

    1. **reference_seq** provided directly → used as-is (like Mode C).
    2. **region** provided (``chr:start-end``) → extracted from hg38
       (like Mode B).
    3. **target_gene** provided → looked up in the GTF, then extracted
       from hg38 (like Mode A).

    Parameters
    ----------
    project_dir : str
        Path to the project directory containing a ``fastq/`` subfolder
        and optionally ``runs.csv`` / ``runs.tsv``.
    protospacer : str
        The guide RNA (gRNA / protospacer) sequence, e.g.
        ``"GGTGGATCCTATTCTAAACG"``. Required.
    hg38 : str, optional
        Path to the hg38 reference genome FASTA (must be indexed).
        Defaults to the copy bundled in ``targeted/reference/``.
    gtf : str, optional
        Path to a GTF gene annotation file. Defaults to the copy
        bundled in ``targeted/reference/``.
    target_gene : str, optional
        Gene symbol (e.g. ``"RAB11A"``). Triggers gene-based reference
        extraction from the GTF + hg38.
    region : str, optional
        Genomic region ``chr:start-end``. Triggers region-based
        reference extraction from hg38.
    reference_seq : str, optional
        A DNA sequence to use directly as the reference.
    template_seq : str, optional
        A DNA template sequence (HDR template). Left blank if not
        provided.
    gene_flank : int, default 0
        Bases of flanking sequence around the gene body (max 100).
    fastq_subdir : str, default ``"fastq"``
        Name of the FASTQ subfolder inside *project_dir*.
    out_basename : str, default ``"samplesheet_final"``
        Base name for output files. Writes ``<basename>.tsv`` and
        ``<basename>.csv``.
    output_dir : str, optional
        Directory to write the samplesheet files into. Defaults to
        *project_dir* when not provided.
    openai_model : str, default ``"gpt-4.1-mini"``
        OpenAI model used for control classification (if key available).
    no_llm : bool, default False
        Force disable LLM even when a key is available.
    openai_key_file : str or None
        Optional path to a file containing the OpenAI API key.

    Returns
    -------
    tuple of (str, str)
        Paths to the generated TSV and CSV files.

    Examples
    --------
    Gene-based (Mode A) — hg38 and gtf default to bundled copies::

        tsv, csv = generate_samplesheet(
            project_dir="extracted_data/PRJNA1240319",
            protospacer="GGTGGATCCTATTCTAAACG",
            target_gene="RAB11A",
        )

    Region-based (Mode B)::

        tsv, csv = generate_samplesheet(
            project_dir="extracted_data/PRJNA1240319",
            protospacer="GGTGGATCCTATTCTAAACG",
            region="chr15:65869459-65891991",
        )

    Direct reference (Mode C)::

        tsv, csv = generate_samplesheet(
            project_dir="extracted_data/PRJNA1240319",
            protospacer="GGTGGATCCTATTCTAAACG",
            reference_seq="ACGTACGT...",
        )
    """
    pdir = Path(project_dir)
    if not pdir.exists():
        err(f"PROJECT DIRECTORY DOES NOT EXIST: {pdir}")

    hg38_path = Path(hg38)
    gtf_path = Path(gtf)

    dest = Path(output_dir) if output_dir else pdir
    dest.mkdir(parents=True, exist_ok=True)
    out_tsv = dest / f"{out_basename}.tsv"
    out_csv = dest / f"{out_basename}.csv"

    protospacer = re.sub(r"[^ACGT]", "", protospacer.strip().upper())
    if not protospacer:
        err("protospacer is required and must be a valid ACGT sequence")

    # --- resolve reference sequence ---
    ref_seq = ""
    if reference_seq:
        ref_seq = re.sub(r"[^ACGTN]", "", reference_seq.strip().upper())
        if not ref_seq:
            err("reference_seq was provided but is empty after cleaning")
        tick(f"USING PROVIDED REFERENCE SEQUENCE ({len(ref_seq)} bp)")

    elif region:
        region = normalize_region(region)
        if not hg38_path.exists():
            err(f"hg38 FASTA NOT FOUND: {hg38_path}")
        ensure_fai(hg38_path)
        try:
            ref_seq = fetch_seq(hg38_path, region)
        except subprocess.CalledProcessError:
            err(f"FAILED TO FETCH SEQUENCE FOR REGION: {region}")
        if not ref_seq:
            err(f"EMPTY SEQUENCE RETURNED FOR REGION: {region}")
        tick(f"EXTRACTED REFERENCE FROM REGION {region} ({len(ref_seq)} bp)")

    elif target_gene:
        if not gtf_path.exists():
            err(f"GTF NOT FOUND: {gtf_path}")
        if not hg38_path.exists():
            err(f"hg38 FASTA NOT FOUND: {hg38_path}")
        ensure_fai(hg38_path)

        info("INDEXING GENES FROM GTF...")
        gene_map = load_gtf_gene_coords(gtf_path)
        tick(f"GTF GENES INDEXED: {len(gene_map)}")

        gene_key = target_gene.strip().upper()
        if gene_key not in gene_map:
            err(f"GENE NOT FOUND IN GTF: {gene_key}")

        chrom, gstart, gend, strand = gene_map[gene_key]
        flank = clamp_flank(gene_flank)
        start = max(1, gstart - flank)
        end = gend + flank
        gene_region = f"{chrom}:{start}-{end}"
        info(f"FETCHING GENE REGION: {gene_region}")

        try:
            ref_seq = fetch_seq(hg38_path, gene_region)
        except subprocess.CalledProcessError:
            err(f"FAILED TO FETCH GENE REGION: {gene_region}")
        if not ref_seq:
            err(f"EMPTY SEQUENCE FOR GENE REGION: {gene_region}")
        tick(f"EXTRACTED REFERENCE FOR {gene_key} ({len(ref_seq)} bp)")
    else:
        err(
            "No reference source provided. Supply one of: "
            "reference_seq, region (chr:start-end), or target_gene"
        )

    template = re.sub(r"[^ACGTN]", "", template_seq.strip().upper()) if template_seq else ""

    # --- scan FASTQs ---
    fastq_dir = pdir / fastq_subdir
    rows, pe, se, missing_r1 = scan_fastqs(fastq_dir)
    tick(f"DETECTED {len(rows)} SAMPLES (PE: {pe}, SE: {se})")

    # --- classify controls ---
    runs_path = find_runs_file(pdir)
    controls: set[str] = set()

    api_key = read_openai_key("OPENAI_API_KEY", openai_key_file, project_dir=pdir)
    llm_ok = (not no_llm) and bool(api_key)

    if runs_path:
        info(f"FOUND RUNS FILE: {runs_path}")
        runs_rows = load_runs_table(runs_path)
        present_ids = {r0["sample"] for r0 in rows}
        runs_present = {rid: rr for rid, rr in runs_rows.items() if rid in present_ids}

        if runs_present:
            if llm_ok:
                info("CLASSIFYING CONTROLS WITH LLM...")
                try:
                    out = llm_classify_controls(runs_present, api_key=api_key, model=openai_model)
                    controls = set(out.get("controls", []) or [])
                    tick(f"LLM CONTROLS DETECTED: {len(controls)}")
                except Exception as e:
                    warn(f"LLM FAILED: {e} — FALLING BACK TO HEURISTICS")
                    controls = {rid for rid, rr in runs_present.items() if heuristic_control(rr)}
                    tick(f"HEURISTIC CONTROLS DETECTED: {len(controls)}")
            else:
                controls = {rid for rid, rr in runs_present.items() if heuristic_control(rr)}
                tick(f"HEURISTIC CONTROLS DETECTED: {len(controls)}")
    else:
        warn("NO runs.csv/runs.tsv FOUND — CONTROL DETECTION SKIPPED")

    # --- build rows ---
    rows_out = []
    for r0 in rows:
        if r0["sample"] in controls:
            rows_out.append({
                "sample": r0["sample"],
                "fastq_1": r0["fastq_1"],
                "fastq_2": r0.get("fastq_2", ""),
                "reference": "",
                "protospacer": "",
                "template": "",
            })
        else:
            rows_out.append({
                "sample": r0["sample"],
                "fastq_1": r0["fastq_1"],
                "fastq_2": r0.get("fastq_2", ""),
                "reference": ref_seq,
                "protospacer": protospacer,
                "template": template,
            })

    write_outputs(rows_out, out_tsv, out_csv)
    tick(f"SAMPLESHEET WRITTEN TSV: {out_tsv.as_posix()}")
    tick(f"SAMPLESHEET WRITTEN CSV: {out_csv.as_posix()}")

    return str(out_tsv), str(out_csv)


if __name__ == "__main__":
    main()
