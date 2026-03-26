#!/usr/bin/env python3
"""
full_main_orchestrator.py (Option-B JSON schema edition)

What changed vs your previous AUTO:
1) COUNTS/DEGS JSONs are NEVER modified. No normalization, no ANY/ALL, no wrapper writes.
2) GL/GC/KG/import JSONs are converted into Option B:
     { PATHWAY: { "UP": {cat:[...]}, "DOWN": {...} }, ... }
   i.e., top-level is pathways, then direction, then entity categories.
3) The old bundle-wrapper normalization (UP/DOWN/ANY/ALL) is removed entirely.

IMPORTANT:
- If mdp_insights currently expects ANY/ALL wrapper JSONs, it must be updated to Option B.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


from .analysis_tools.pathway_classifier import categorize_disease_enrichment_files

# =============================================================================
# Robust error model
# =============================================================================

class OrchestratorError(RuntimeError):
    """Base class for orchestrator failures."""


class UserInputError(OrchestratorError):
    """Bad CLI args / invalid user input."""


class MissingResourceError(OrchestratorError):
    """Required file/module/folder missing."""


class SubprocessError(OrchestratorError):
    """A subprocess returned non-zero."""


# =============================================================================
# Consensus tissues + resolver (COUNTS only)
# =============================================================================

CONSENSUS_TISSUES = sorted({
    "adipose tissue", "adrenal gland", "amygdala", "appendix", "basal ganglia", "bone marrow", "breast",
    "cerebellum", "cerebral cortex", "cervix", "choroid plexus", "colon", "corpus callosum", "ductus deferens",
    "duodenum", "endometrium", "epididymis", "esophagus", "fallopian tube", "gallbladder", "heart muscle",
    "hippocampal formation", "hypothalamus", "kidney", "liver", "lung", "lymph node", "midbrain", "olfactory bulb",
    "ovary", "pancreas", "parathyroid gland", "pituitary gland", "placenta", "pons", "pons and medulla", "prostate",
    "rectum", "retina", "salivary gland", "seminal vesicle", "skeletal muscle", "skin", "small intestine",
    "smooth muscle", "spinal cord", "spleen", "stomach", "testis", "thalamus", "thymus", "thyroid gland",
    "tongue", "tonsil", "urinary bladder", "vagina",
})

FUZZY_MIN_SCORE = 0.86


def _norm_tissue(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _exact_or_none(tissue_in: str) -> Optional[str]:
    if not tissue_in:
        return None
    tnorm = _norm_tissue(tissue_in)
    norm_map = {_norm_tissue(t): t for t in CONSENSUS_TISSUES}
    return norm_map.get(tnorm)


def _best_fuzzy_tissue(query: str) -> Tuple[Optional[str], float]:
    if not query:
        return None, 0.0
    q = _norm_tissue(query)
    best_t = None
    best_s = 0.0
    for c in CONSENSUS_TISSUES:
        s = difflib.SequenceMatcher(None, q, _norm_tissue(c)).ratio()
        if s > best_s:
            best_t, best_s = c, s
    return best_t, best_s


def _llm_pick_tissue(disease: str, user_tissue: str, hint: str = "") -> str:
    """
    LLM fallback: MUST return exactly one item from CONSENSUS_TISSUES.
    Requires openai + OPENAI_API_KEY.
    """

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise MissingResourceError(
            "[ERROR] Tissue unresolved and LLM fallback needed, but OPENAI_API_KEY is not set."
        )

    client = OpenAI(api_key=api_key)

    allowed = "\n".join(f"- {t}" for t in CONSENSUS_TISSUES)
    prompt = f"""
Select the single best-matching tissue label for a disease cohort.

Disease name: {disease}
User-provided tissue (may be empty/invalid): {user_tissue}
Extra hint (may include folder name): {hint}

You MUST choose exactly ONE tissue from this allowed list and output it verbatim (no extra text):
{allowed}
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a bioinformatics assistant. Return only the tissue name from the allowed list, nothing else."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=50,
        )
        out = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise MissingResourceError(f"[ERROR] LLM tissue resolution failed: {e}")

    # Clean up response - extract just tissue name
    out = out.strip()
    # Remove markdown code blocks if present
    if out.startswith("```"):
        parts = out.split("```")
        if len(parts) > 1:
            out = parts[1].strip()
            if out.startswith("text") or out.startswith("plain"):
                out = out[4:].strip()
    if out.endswith("```"):
        out = out.rsplit("```", 1)[0].strip()
    
    # Remove common prefixes/suffixes
    out = re.sub(r"^(The|Selected|Best|Tissue|Label|Name|Answer|Output|Result)[:\s]+", "", out, flags=re.IGNORECASE)
    out = out.strip()
    
    # Extract first line if multiple lines
    if "\n" in out:
        out = out.split("\n")[0].strip()
    
    # Remove trailing punctuation
    out = re.sub(r"[.,;:!?]+$", "", out).strip()

    # Try exact match first
    if out in CONSENSUS_TISSUES:
        return out
    
    # Try fuzzy matching as fallback
    best_match, score = _best_fuzzy_tissue(out)
    if best_match and score >= FUZZY_MIN_SCORE:
        print(f"[tissue] LLM returned '{out}' (not exact); using fuzzy match '{best_match}' (score={score:.3f})")
        return best_match
    
    # Try to find tissue name within response
    for tissue in CONSENSUS_TISSUES:
        if tissue.lower() in out.lower():
            print(f"[tissue] Found '{tissue}' within LLM response '{out}'; using it")
            return tissue
    
        raise UserInputError(
        f"[ERROR] LLM returned invalid tissue '{out}'. Must be one of the consensus tissues. "
        f"Best fuzzy match was '{best_match}' (score={score:.3f})"
        )


def resolve_tissue_or_die(disease: str, tissue_in: str, hint: str = "", allow_llm: bool = True) -> str:
    tissue_in = (tissue_in or "").strip()

    exact = _exact_or_none(tissue_in)
    if exact:
        return exact

    if tissue_in:
        best, score = _best_fuzzy_tissue(tissue_in)
        if best and score >= FUZZY_MIN_SCORE:
            print(f"[tissue] '{tissue_in}' not exact; using fuzzy match '{best}' (score={score:.3f})")
            return best
        print(f"[tissue] '{tissue_in}' not in consensus tissues; fuzzy score={score:.3f} < {FUZZY_MIN_SCORE:.2f}")

    if not allow_llm:
        raise UserInputError(
            "[ERROR] Tissue unresolved and --no-llm-tissue set. Provide a valid tissue "
            "from consensus tissues (example: 'colon', 'pancreas')."
        )

    picked = _llm_pick_tissue(disease=disease, user_tissue=tissue_in, hint=hint)
    print(f"[tissue] using LLM-picked tissue for '{disease}': '{picked}'")
    return picked


# =============================================================================
# Small utils
# =============================================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def wslpath(p: Path) -> str:
    return str(p.resolve())


def _require_dir(p: Path, what: str) -> None:
    if not p.exists() or not p.is_dir():
        raise MissingResourceError(f"[ERROR] {what} not found or not a directory: {p}")


def _require_file(p: Path, what: str) -> None:
    if not p.exists() or not p.is_file():
        raise MissingResourceError(f"[ERROR] {what} not found or not a file: {p}")


def _require_nonempty_dir(p: Path, what: str, glob_pat: str = "*") -> None:
    _require_dir(p, what)
    hits = list(p.glob(glob_pat))
    if not hits:
        raise MissingResourceError(f"[ERROR] {what} is empty (no matches for '{glob_pat}'): {p}")


def _list_input_files(folder: Path) -> List[Path]:
    patterns = ["*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls", "*.json"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(folder.glob(pat)))
    return files


def _safe_slug(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return "unnamed"
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"


def _is_probably_path(s: str) -> bool:
    try:
        return Path(s).expanduser().exists()
    except Exception:
        return False


# =============================================================================
# Streaming subprocess runner (no buffering)
# =============================================================================

@dataclass
class CmdResult:
    code: int
    argv: List[str]


def run_cmd(argv: Sequence[str],
            cwd: Optional[Path] = None,
            env: Optional[Dict[str, str]] = None) -> CmdResult:
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    argv_list = [str(x) for x in argv]
    print(f"[run] {' '.join(argv_list)}")

    proc = subprocess.Popen(
        argv_list,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
    )
    code = proc.wait()
    return CmdResult(code=code, argv=argv_list)


def _module_cli(mod: str) -> List[str]:
    """
    Build a safe ['-m', <module>] argument list.
    Accepts relative modules like '.counts_mdp.run_counts_from_dict'
    and rewrites them to the current package (e.g., 'multi_pathway.…').
    """
    base_pkg = (__package__ or Path(__file__).parent.name)
    mod = (mod or "").strip()
    if mod.startswith("."):
        mod = f"{base_pkg}{mod}"
    return ["-m", mod]


# =============================================================================
# CLI cohort parsing (with typo fixes)
# =============================================================================

def _parse_kv_triple(s: str) -> Dict[str, str]:
    """
    Parse: --cohort "name=...,input=...,tissue=..." (tissue optional).
    Auto-fix: input==/path  -> input=/path (warn).
    """
    out: Dict[str, str] = {}
    raw = (s or "").strip()
    if not raw:
        raise UserInputError("[ERROR] Empty --cohort value.")

    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()

    need = {"name", "input"}
    missing = need - set(out)
    if missing:
        raise UserInputError(f"[ERROR] --cohort missing keys: {', '.join(sorted(missing))} in '{s}'")

    # Auto-fix common typo: input==/path
    inp = out.get("input", "")
    if inp.startswith("="):
        fixed = inp.lstrip("=")
        print(f"[warn] fixed typo in --cohort: input==... → input={fixed}", file=sys.stderr)
        out["input"] = fixed

    p = Path(out["input"]).expanduser().resolve()
    if not p.exists():
        raise MissingResourceError(f"[ERROR] input path not found for cohort '{out['name']}': {p}")

    return out


# =============================================================================
# DEGS column autodetect (optional)
# =============================================================================

_DEG_ID_CAND = ["gene", "genes", "symbol", "gene_symbol", "hgnc_symbol", "feature", "id"]
_DEG_LFC_CAND = ["log2fc", "log2foldchange", "logfc", "lfc", "log2_fold_change", "logfoldchange", "log_fold_change"]
_DEG_Q_CAND = ["padj", "p_adj", "adj_p", "fdr", "q", "qval", "qvalue", "pvalue_adj", "p_value_adj", "adjp", "fdrq"]


def _normalize_header(h: str) -> str:
    h = (h or "").strip().lower()
    h = re.sub(r"\s+", "", h)
    h = h.replace("-", "").replace("_", "")
    return h


def detect_degs_columns_from_file(f: Path) -> Tuple[str, str, str]:
    # Excel -> default (no heavy deps here)
    if f.suffix.lower() in {".xlsx", ".xls"}:
        return ("Gene", "log2FoldChange", "padj")

    delimiter = "\t" if f.suffix.lower() in {".tsv", ".txt"} else ","
    try:
        with f.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
                delimiter = dialect.delimiter
            except Exception:
                pass
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader, [])
    except Exception:
        return ("Gene", "log2FoldChange", "padj")

    norm = [_normalize_header(x) for x in header]

    def pick(cands: List[str]) -> Optional[str]:
        cand_norm = [_normalize_header(x) for x in cands]
        for i, h in enumerate(norm):
            if h in cand_norm:
                return header[i]
        return None

    id_col = pick(_DEG_ID_CAND) or "Gene"
    lfc_col = pick(_DEG_LFC_CAND) or "log2FoldChange"
    q_col = pick(_DEG_Q_CAND) or "padj"
    return (id_col, lfc_col, q_col)


# =============================================================================
# DEGS input selection (folder OR file)
# =============================================================================

def pick_one_degs_file(p: Path) -> Path:
    """
    Accept either:
      - a file path (return it)
      - a folder containing DEG table(s) (choose one safely)
    """
    p = p.expanduser().resolve()
    if p.is_file():
        return p

    _require_dir(p, "DEGs input folder")
    candidates = [x for x in _list_input_files(p) if x.suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx", ".xls"}]

    if not candidates:
        raise MissingResourceError(f"[ERROR] No DEG table found in folder: {p}")

    preferred = [c for c in candidates if "degs" in c.name.lower()]
    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        raise UserInputError(
            "[ERROR] Multiple DEG-like files found (contain 'degs'). "
            "Pick one and pass it explicitly as input=<FILE>:\n  - "
            + "\n  - ".join(str(x) for x in preferred)
        )

    if len(candidates) == 1:
        return candidates[0]

    raise UserInputError(
        "[ERROR] Multiple tables found in DEGs folder. "
        "Pass a specific file path via input=<FILE>.\n  - "
        + "\n  - ".join(str(x) for x in candidates)
    )


# =============================================================================
# GC / GL runners (unchanged)
# =============================================================================

def run_gc(input_dir: Path, out_root: Path, env: Dict[str, str],
           gene_key: str = "gene_symbol", score_key: str = "gene_score") -> Path:
    _require_dir(input_dir, "GC --input")
    ensure_dir(out_root)

    res = run_cmd([
        sys.executable, *_module_cli(".GC_enricher.GC_main"),
        "--in", str(input_dir),
        "--out", str(out_root),
        "--gene-key", gene_key,
        "--score-key", score_key,
    ], env=env)
    if res.code != 0:
        raise SubprocessError("GC run failed.")
    return out_root / "GC_enrich"


def run_gl(input_dir: Path, out_root: Path, env: Dict[str, str]) -> Path:
    _require_dir(input_dir, "GL --input")
    ensure_dir(out_root)

    res = run_cmd([
        sys.executable, *_module_cli(".genelist_mdp.genelist_runner"),
        "--runner", "GL",
        "--input", str(input_dir),
        "--out-root", str(out_root),
    ], env=env)
    if res.code != 0:
        raise SubprocessError("GL run failed.")
    return out_root / "GL_enrich"


def build_json_bundle_for_gl_gc(pipeline_base: Path, env: Dict[str, str]) -> None:
    _require_dir(pipeline_base, "GL/GC base folder (GL_enrich/GC_enrich)")
    res = run_cmd([sys.executable, *_module_cli(".genelist_mdp.json_maker"), str(pipeline_base)], env=env)
    if res.code != 0:
        raise SubprocessError("genelist_mdp.json_maker failed (needed to create jsons_all_folder).")


# =============================================================================
# COUNTS/DEGS unified runner: counts_mdp.run_counts_from_dict
# =============================================================================

def _invoke_counts_from_dict(spec: Dict[str, Any], out_root: Path, env: Dict[str, str], label: str) -> None:
    ensure_dir(out_root)
    spec_file = out_root / f"_{label}_spec.json"
    spec_file.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    res = run_cmd([
        sys.executable, *_module_cli(".counts_mdp.run_counts_from_dict"),
        "--spec", str(spec_file),
        "--out-root", str(out_root),
    ], env=env)

    if res.code != 0:
        raise SubprocessError(f"{label.upper()} run failed (counts_mdp.run_counts_from_dict returned {res.code}).")


def build_counts_spec(cohorts: List[Dict[str, str]], out_root: Path,
                      id_col: str, lfc_col: str, q_col: str, q_max: float) -> Dict[str, Any]:
    spec_cohorts: List[Dict[str, Any]] = []
    for c in cohorts:
        counts_dir = Path(c["input"]).expanduser().resolve()
        _require_dir(counts_dir, f"counts input for cohort '{c['name']}'")

        spec_cohorts.append({
            "name": c["name"].strip(),
            "counts_dir": str(counts_dir),
            "id_col": id_col,
            "lfc_col": lfc_col,
            "q_col": q_col,
            "q_max": float(q_max),
            "tissue": c.get("tissue", "").strip(),
        })
    return {"out_root": str(out_root), "cohorts": spec_cohorts}


def build_degs_spec(cohorts: List[Dict[str, str]], out_root: Path,
                    id_col: str, lfc_col: str, q_col: str, q_max: float) -> Dict[str, Any]:
    """
    DEGS spec MUST use 'degs_file' so counts_mdp reads DE tables directly (no DESeq2/Welch).
    """
    spec_cohorts: List[Dict[str, Any]] = []
    for c in cohorts:
        inp = Path(c["input"]).expanduser().resolve()
        degs_path = pick_one_degs_file(inp)
        _require_file(degs_path, f"DEGs file for cohort '{c['name']}'")

        spec_cohorts.append({
            "name": c["name"].strip(),
            "degs_file": str(degs_path),
            "id_col": id_col,
            "lfc_col": lfc_col,
            "q_col": q_col,
            "q_max": float(q_max),
            "tissue": "",
        })
    return {"out_root": str(out_root), "cohorts": spec_cohorts}


# =============================================================================
# Mode runners (original)
# =============================================================================

def run_counts_mode(args: argparse.Namespace, out_root: Path, env: Dict[str, str]) -> Dict[str, str]:
    resolved_tissues: Dict[str, str] = {}

    if args.cohort:
        cohorts = [_parse_kv_triple(s) for s in args.cohort]
        for c in cohorts:
            p = Path(c["input"]).expanduser().resolve()
            _require_dir(p, f"counts input for cohort '{c['name']}'")

            hint = f"input_folder={p.name}"
            c["tissue"] = resolve_tissue_or_die(
                disease=c["name"],
                tissue_in=c.get("tissue", ""),
                hint=hint,
                allow_llm=(not args.no_llm_tissue),
            )
            resolved_tissues[c["name"]] = c["tissue"]

        (out_root / "resolved_tissues.json").write_text(
            json.dumps(resolved_tissues, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        spec = build_counts_spec(
            cohorts=cohorts,
            out_root=out_root,
            id_col=args.counts_id_col,
            lfc_col=args.counts_lfc_col,
            q_col=args.counts_q_col,
            q_max=args.counts_q_max,
        )
        print(f"[counts] cohorts={len(cohorts)} → counts_mdp.run_counts_from_dict → OUT_ROOT={wslpath(out_root)}")
        _invoke_counts_from_dict(spec, out_root, env=env, label="counts")
        return resolved_tissues

    if not args.input:
        raise UserInputError("--input is required for COUNTS single-cohort")
    if not args.disease_name:
        raise UserInputError("--disease-name is required for COUNTS single-cohort")

    input_dir = Path(args.input).expanduser().resolve()
    _require_dir(input_dir, "COUNTS --input")

    hint = f"input_folder={input_dir.name}"
    resolved = resolve_tissue_or_die(
        disease=args.disease_name.strip(),
        tissue_in=args.tissue.strip(),
        hint=hint,
        allow_llm=(not args.no_llm_tissue),
    )
    resolved_tissues[args.disease_name.strip()] = resolved
    (out_root / "resolved_tissues.json").write_text(
        json.dumps(resolved_tissues, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cohorts = [{"name": args.disease_name.strip(), "input": str(input_dir), "tissue": resolved}]
    spec = build_counts_spec(
        cohorts=cohorts,
        out_root=out_root,
        id_col=args.counts_id_col,
        lfc_col=args.counts_lfc_col,
        q_col=args.counts_q_col,
        q_max=args.counts_q_max,
    )
    print(f"[counts] cohorts=1 → counts_mdp.run_counts_from_dict → OUT_ROOT={wslpath(out_root)}")
    _invoke_counts_from_dict(spec, out_root, env=env, label="counts")
    return resolved_tissues


def run_degs_mode(args: argparse.Namespace, out_root: Path, env: Dict[str, str]) -> None:
    cohorts: List[Dict[str, str]] = []

    if args.cohort:
        for s in args.cohort:
            c = _parse_kv_triple(s)
            cohorts.append({"name": c["name"], "input": c["input"]})
    else:
        if not args.input:
            raise UserInputError("--input is required for DEGS mode (single cohort) OR provide --cohort ...")
        name = (args.disease_name or Path(args.input).expanduser().resolve().stem).strip()
        cohorts.append({"name": name, "input": str(Path(args.input).expanduser().resolve())})

    if args.degs_id_col and args.degs_lfc_col and args.degs_q_col:
        id_col, lfc_col, q_col = args.degs_id_col, args.degs_lfc_col, args.degs_q_col
        print(f"[degs] using user columns: id='{id_col}' lfc='{lfc_col}' q='{q_col}'")
    else:
        first_inp = Path(cohorts[0]["input"]).expanduser().resolve()
        degs_file = pick_one_degs_file(first_inp)
        id_col, lfc_col, q_col = detect_degs_columns_from_file(degs_file)
        print(f"[degs] autodetected from '{degs_file.parent.name if degs_file.parent else degs_file.name}': "
              f"id='{id_col}' lfc='{lfc_col}' q='{q_col}'")

    spec = build_degs_spec(
        cohorts=cohorts,
        out_root=out_root,
        id_col=id_col,
        lfc_col=lfc_col,
        q_col=q_col,
        q_max=args.degs_q_max,
    )
    print(f"[degs] cohorts={len(cohorts)} → counts_mdp.run_counts_from_dict → OUT_ROOT={wslpath(out_root)}")
    _invoke_counts_from_dict(spec, out_root, env=env, label="degs")


# =============================================================================
# Optional enzymes runner
# =============================================================================

def run_enzyme_and_signaling_batch(parent_dir: Path, env: Dict[str, str]) -> None:
    script = Path("enzyme_and_signaling.py")
    if not script.exists():
        print("[warn] enzyme_and_signaling.py not found at repo root; skipping.", file=sys.stderr)
        return
    res = run_cmd([sys.executable, str(script), "--input", str(parent_dir), "--batch"], env=env)
    if res.code != 0:
        raise SubprocessError("enzyme_and_signaling.py failed.")


# =============================================================================
# JSON root resolution (strict)
# =============================================================================

def resolve_json_root(mode: str, out_root: Path) -> Path:
    if mode == "auto":
        p = out_root / "results" / "all_jsons"
        _require_nonempty_dir(p, "AUTO unified JSON root", "*.json")
        return p

    if mode in {"counts", "degs"}:
        p = out_root / "results" / "all_jsons"
        _require_nonempty_dir(p, "COUNTS/DEGS JSON root", "*.json")
        return p

    if mode == "gl":
        p = out_root / "GL_enrich" / "jsons_all_folder"
        _require_nonempty_dir(p, "GL JSON root", "*.json")
        return p

    if mode == "gc":
        p = out_root / "GC_enrich" / "jsons_all_folder"
        _require_nonempty_dir(p, "GC JSON root", "*.json")
        return p

    raise UserInputError(f"[ERROR] Unsupported mode for json root resolution: {mode}")


# =============================================================================
# Pathway classifier (best-effort)
# =============================================================================

def run_pathway_classifier(classifier_root: Path) -> None:


    root_dir = Path(classifier_root).expanduser().resolve()
    if not root_dir.exists():
        print(f"[orchestrator][warn] classifier_root not found (skipping): {root_dir}", file=sys.stderr)
        return

    print(f"[orchestrator] Running pathway_classifier on {wslpath(root_dir)} ...")
    try:
        updated_files = categorize_disease_enrichment_files(root_dir=root_dir)
        try:
            n = len(updated_files)
        except Exception:
            n = -1
        if n >= 0:
            print(f"[orchestrator] pathway_classifier updated {n} file(s).")
        else:
            print("[orchestrator] pathway_classifier completed.")
    except Exception as e:
        print(f"[orchestrator] WARNING: pathway_classifier failed: {e}", file=sys.stderr)


# =============================================================================
# REPORT PREP (unchanged)
# =============================================================================

def _candidate_classifier_roots_for_report(out_root: Path) -> List[Path]:
    roots: List[Path] = []
    roots.append(out_root)
    roots.append(out_root / "GL_enrich")
    roots.append(out_root / "GC_enrich")
    for p in out_root.glob("*_enrich"):
        if p.is_dir():
            roots.append(p)

    seen: Set[str] = set()
    out: List[Path] = []
    for r in roots:
        try:
            key = str(r.resolve())
        except Exception:
            key = str(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _find_gsea_prerank_file(disease_dir: Path) -> Optional[Path]:
    p1 = disease_dir / "gsea_prerank.tsv"
    if p1.exists() and p1.is_file():
        return p1
    p2 = disease_dir / "prerank" / "gsea_prerank.tsv"
    if p2.exists() and p2.is_file():
        return p2
    return None


def ensure_report_has_classified_gsea(out_root: Path) -> None:
    out_root = Path(out_root).expanduser().resolve()
    roots = _candidate_classifier_roots_for_report(out_root)

    missing: List[Tuple[Path, Path]] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            children = list(root.iterdir())
        except Exception:
            continue
        for d in children:
            if not d.is_dir():
                continue
            classified = d / "gsea_prerank_classified.tsv"
            if classified.exists() and classified.is_file():
                continue
            src = _find_gsea_prerank_file(d)
            if src is None:
                continue
            missing.append((d, src))

    if not missing:
        return

    print(f"[orchestrator][report] Missing gsea_prerank_classified.tsv in {len(missing)} disease folder(s).")
    print("[orchestrator][report] Re-running pathway_classifier on candidate roots...")

    for root in roots:
        if root.exists() and root.is_dir():
            run_pathway_classifier(root)

    still_missing: List[Tuple[Path, Path]] = []
    for d, src in missing:
        classified = d / "gsea_prerank_classified.tsv"
        if classified.exists() and classified.is_file():
            continue
        src2 = _find_gsea_prerank_file(d) or src
        still_missing.append((d, src2))

    if not still_missing:
        return

    print(f"[orchestrator][report][warn] Classifier did not produce classified files for {len(still_missing)} folder(s).")
    print("[orchestrator][report][warn] Creating fallback gsea_prerank_classified.tsv by copying gsea_prerank.tsv...")

    for d, src in still_missing:
        classified = d / "gsea_prerank_classified.tsv"
        try:
            classified.write_text(
                src.read_text(encoding="utf-8", errors="replace"),
                encoding="utf-8",
            )
            print(f"[orchestrator][report] wrote: {wslpath(classified)}")
        except Exception as e:
            print(f"[orchestrator][report][warn] failed to write classified for {d.name}: {e}", file=sys.stderr)


# =============================================================================
# Terminal steps (unchanged)
# =============================================================================

def run_pathway_compare(json_root: Path, out_dir: Path, pathways: List[str], direction: str, env: Dict[str, str]) -> None:
    ensure_dir(out_dir)
    argv = [
        sys.executable, "run_pathway_compare.py",
        "--json-root", str(json_root),
        "--out-root", str(out_dir),
        "--direction-mode", direction,
        "-v",
    ]
    for p in pathways:
        argv += ["--pathways", p]
    res = run_cmd(argv, env=env)
    if res.code != 0:
        raise SubprocessError("pathway_compare failed.")


def run_mdp_insights(json_root: Path, out_dir: Path, env: Dict[str, str]) -> None:
    ensure_dir(out_dir)
    argv = [
        sys.executable, *_module_cli(".mdp_insights.main"),
        "--json-root", str(json_root),
        "--out-root", str(out_dir),
        "-v",
    ]
    res = run_cmd(argv, env=env)
    if res.code != 0:
        raise SubprocessError("mdp_insights failed.")


def run_mdp_report_plan2(out_root: Path, env: Dict[str, str], no_llm: bool, q_cutoff: float) -> None:
    argv = [sys.executable, *_module_cli(".report.report"), "--counts-root", str(out_root)]
    if no_llm:
        argv.append("--no-llm")
    argv += ["--q-cutoff", str(q_cutoff)]
    res = run_cmd(argv, env=env)
    if res.code != 0:
        raise SubprocessError("Report generation failed (report.report).")


# =============================================================================
# AUTO mode: Option-B JSON normalization for GL/GC/KG/import only
# =============================================================================

class AutoKind:
    COUNTS = "counts"
    DEGS = "degs"
    GL = "gl"
    GC_INPUT = "gc_input"
    BUNDLE_JSON = "bundle_json"   # importable overlap json (Option B OR legacy dir-wrapped)
    KG_DISEASE = "kg_disease"


def _try_parse_json_file(p: Path) -> Optional[Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _has_dir_keys(d: Dict[str, Any]) -> bool:
    keys_upper = {str(k).strip().upper() for k in d.keys()}
    return bool(keys_upper & {"UP", "DOWN", "ANY", "ALL"})


def _looks_like_option_b_bundle(obj: Any) -> bool:
    """
    Option B signature:
      top-level keys look like pathway names, values are dicts with UP/DOWN or category keys.
    """
    if not isinstance(obj, dict) or not obj:
        return False

    # sample a few items
    checked = 0
    hits = 0
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, dict):
            continue
        checked += 1
        v_keys = {str(x).strip().lower() for x in v.keys()}
        if ("up" in v_keys) or ("down" in v_keys):
            hits += 1
        elif v_keys & {"metabolites", "tf", "epigenetic"}:
            # pathway directly contains categories (directionless)
            hits += 1
        if checked >= 8:
            break
    return checked > 0 and hits >= max(1, checked // 2)


def _looks_like_overlap_bundle(obj: Any) -> bool:
    """
    Accept BOTH:
      - legacy wrapper JSONs: {"UP": {...}, "DOWN": {...}, ...}
      - Option B JSONs: {pathway: {"UP": {...}, "DOWN": {...}}}
    """
    if not isinstance(obj, dict):
        return False
    if _has_dir_keys(obj):
        return True
    return _looks_like_option_b_bundle(obj)


def _write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _snapshot_jsons(folder: Path) -> Set[Path]:
    if not folder.exists() or not folder.is_dir():
        return set()
    return set(folder.glob("*.json"))


def _unique_dest_name(dest_dir: Path, desired_name: str) -> Path:
    ensure_dir(dest_dir)
    base = Path(desired_name).stem
    suffix = Path(desired_name).suffix or ".json"
    cand = dest_dir / f"{base}{suffix}"
    if not cand.exists():
        return cand
    for i in range(2, 5000):
        cand2 = dest_dir / f"{base}__{i}{suffix}"
        if not cand2.exists():
            return cand2
    raise OrchestratorError(f"[ERROR] Could not allocate unique filename in {dest_dir} for '{desired_name}'")


def _dedupe_list(values: List[Any]) -> List[Any]:
    seen: Set[str] = set()
    out: List[Any] = []
    for x in values:
        try:
            key = json.dumps(x, sort_keys=True, default=str)
        except Exception:
            key = str(x)
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def _deep_merge_option_b(a: Any, b: Any) -> Any:
    """
    Merge dict/list in a stable way for Option-B payloads.
    - dict: merge keys recursively
    - list: concat + dedupe
    - scalar: keep 'a' unless empty-ish
    """
    if a is None:
        return b
    if b is None:
        return a

    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, vb in b.items():
            if k in out:
                out[k] = _deep_merge_option_b(out[k], vb)
            else:
                out[k] = vb
        return out

    if isinstance(a, list) and isinstance(b, list):
        return _dedupe_list(list(a) + list(b))

    if a in ("", [], {}):
        return b
    return a


def normalize_to_option_b(obj: Any, force_up_if_directionless: bool = True) -> Dict[str, Any]:
    """
    Convert multiple upstream shapes into Option B:

    Option B target:
      { PATHWAY: { "UP": {...}, "DOWN": {...} }, ... }

    Supported inputs:
    1) Legacy wrapper:
        {"UP": {path: payload, ...}, "DOWN": {...}}
    2) Double-nested weird:
        {"UP": {path: {"UP": {...}, "DOWN": {...}}}}
    3) Directionless by pathway:
        {path: {"metabolites":[...], "tf":[...], ...}}
    4) Already Option B:
        {path: {"UP": {...}, "DOWN": {...}}}
    """
    if not isinstance(obj, dict):
        return {}

    # Case: already Option B (mostly)
    if _looks_like_option_b_bundle(obj) and not _has_dir_keys(obj):
        out: Dict[str, Any] = {}
        for path, node in obj.items():
            if not isinstance(path, str):
                continue
            if not isinstance(node, dict):
                continue
            node_keys_u = {str(k).strip().upper() for k in node.keys()}
            if "UP" in node_keys_u or "DOWN" in node_keys_u:
                up = node.get("UP", {}) if "UP" in node_keys_u else {}
                down = node.get("DOWN", {}) if "DOWN" in node_keys_u else {}
            else:
                # directionless categories at pathway level
                if force_up_if_directionless:
                    up = node
                    down = {}
                else:
                    up = node
                    down = {}
            if not isinstance(up, dict):
                up = {}
            if not isinstance(down, dict):
                down = {}
            out[path] = {"UP": up, "DOWN": down}
        return out

    # Case: legacy wrapper-like at top
    if _has_dir_keys(obj):
        # collect possible directional containers
        def pick_dir(name: str) -> Any:
            for k, v in obj.items():
                if str(k).strip().upper() == name:
                    return v
            return None

        top_up = pick_dir("UP")
        top_down = pick_dir("DOWN")
        top_any = pick_dir("ANY")
        top_all = pick_dir("ALL")

        # for safety: base payload if UP missing
        if top_up is None:
            top_up = top_any if top_any is not None else top_all
        if top_down is None:
            top_down = {}

        out: Dict[str, Any] = {}

        def ingest(direction: str, payload: Any) -> None:
            if not isinstance(payload, dict):
                return
            for path, node in payload.items():
                if not isinstance(path, str):
                    continue

                # node might itself be {"UP":..., "DOWN":...} (double nested)
                if isinstance(node, dict):
                    node_keys_u = {str(k).strip().upper() for k in node.keys()}
                    if "UP" in node_keys_u or "DOWN" in node_keys_u:
                        # ignore outer direction and use inner
                        up = node.get("UP", {}) if "UP" in node_keys_u else {}
                        down = node.get("DOWN", {}) if "DOWN" in node_keys_u else {}
                        out[path] = _deep_merge_option_b(out.get(path, {"UP": {}, "DOWN": {}}),
                                                         {"UP": up if isinstance(up, dict) else {},
                                                          "DOWN": down if isinstance(down, dict) else {}})
                        continue

                # otherwise treat node as categories for THIS direction
                if force_up_if_directionless and direction == "UP":
                    up_payload = node if isinstance(node, dict) else {}
                    down_payload = {}
                else:
                    up_payload = node if (direction == "UP" and isinstance(node, dict)) else {}
                    down_payload = node if (direction == "DOWN" and isinstance(node, dict)) else {}

                base = out.get(path, {"UP": {}, "DOWN": {}})
                merged = _deep_merge_option_b(base, {"UP": up_payload, "DOWN": down_payload})
                out[path] = merged

        ingest("UP", top_up)
        ingest("DOWN", top_down)
        return out

    # Case: dict but neither wrapper nor optionB-ish => could be directionless single-pathway map
    out2: Dict[str, Any] = {}
    for path, node in obj.items():
        if not isinstance(path, str):
            continue
        if not isinstance(node, dict):
            continue
        if force_up_if_directionless:
            out2[path] = {"UP": node, "DOWN": {}}
        else:
            out2[path] = {"UP": node, "DOWN": {}}
    return out2


# =============================================================================
# AUTO detection (mostly unchanged)
# =============================================================================

def _sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
        return dialect.delimiter
    except Exception:
        if sample.count("\t") > sample.count(","):
            return "\t"
        return ","


def _iter_rows_delimited(f: Path, max_rows: int = 200) -> Tuple[List[str], List[List[str]], str]:
    delimiter = "\t" if f.suffix.lower() in {".tsv", ".txt"} else ","
    with f.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(8192)
        delimiter = _sniff_delimiter(sample)
        handle.seek(0)
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])
        rows: List[List[str]] = []
        for _ in range(max_rows):
            try:
                r = next(reader)
            except StopIteration:
                break
            rows.append(r)
    return header, rows, delimiter


def _safe_float(x: str) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"na", "nan", "null", "none"}:
            return None
        return float(s)
    except Exception:
        return None


def _looks_like_gene_symbol(token: str) -> bool:
    s = (token or "").strip()
    # Remove surrounding quotes and trailing commas that often appear in simple gene lists.
    s = s.strip(",").strip("\"'").strip()
    if not s or len(s) > 30:
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9\-\.]{0,29}$", s))


def _header_has_any(cands: Iterable[str], header: List[str]) -> bool:
    norm = [_normalize_header(h) for h in header]
    c_norm = {_normalize_header(c) for c in cands}
    return any(h in c_norm for h in norm)


def _analyze_table_semantics(f: Path) -> Dict[str, Any]:
    if f.suffix.lower() in {".xlsx", ".xls"}:
        return {"ok": True, "header": [], "n_cols": 0, "signals": {"excel": True}}

    header, rows, _delim = _iter_rows_delimited(f, max_rows=200)
    header = header or []
    n_cols = len(header) if header else (len(rows[0]) if rows else 0)

    has_lfc = _header_has_any(_DEG_LFC_CAND, header)
    has_q = _header_has_any(_DEG_Q_CAND, header)

    n_rows = len(rows)
    if n_rows == 0 or n_cols == 0:
        return {
            "ok": True,
            "header": header,
            "n_cols": n_cols,
            "signals": {"empty": True, "has_lfc": has_lfc, "has_q": has_q},
        }

    numeric_counts = [0] * n_cols
    nonnull_counts = [0] * n_cols
    integer_like_counts = [0] * n_cols
    negative_counts = [0] * n_cols
    gene_like_counts = [0] * n_cols

    for r in rows:
        if len(r) < n_cols:
            r = r + [""] * (n_cols - len(r))
        for j in range(n_cols):
            v = r[j] if j < len(r) else ""
            fv = _safe_float(v)
            if fv is None:
                if _looks_like_gene_symbol(v):
                    gene_like_counts[j] += 1
                continue
            nonnull_counts[j] += 1
            numeric_counts[j] += 1
            if fv < 0:
                negative_counts[j] += 1
            if abs(fv - round(fv)) < 1e-8 and fv >= 0:
                integer_like_counts[j] += 1

    numeric_cols: List[int] = []
    for j in range(n_cols):
        nn = max(1, nonnull_counts[j])
        if numeric_counts[j] / nn >= 0.80 and nonnull_counts[j] >= max(3, n_rows // 10):
            numeric_cols.append(j)

    n_numeric_cols = len(numeric_cols)

    gene_col_idx = None
    best_gene = 0
    for j in range(n_cols):
        if j in numeric_cols:
            continue
        if gene_like_counts[j] > best_gene:
            best_gene = gene_like_counts[j]
            gene_col_idx = j

    neg_rate = 0.0
    int_rate = 0.0
    if numeric_cols:
        total_num = 0
        total_neg = 0
        total_int = 0
        for j in numeric_cols:
            total_num += max(1, numeric_counts[j])
            total_neg += negative_counts[j]
            total_int += integer_like_counts[j]
        neg_rate = total_neg / max(1, total_num)
        int_rate = total_int / max(1, total_num)

    gene_list_score = 0.0
    if n_cols <= 2 and n_numeric_cols == 0:
        gene_list_score = gene_like_counts[0] / max(1, n_rows)

    gene_score_like = False
    if 1 <= n_numeric_cols <= 2 and n_cols <= 4:
        gene_score_like = (gene_col_idx is not None) and (n_numeric_cols >= 1)

    counts_like = False
    if n_numeric_cols >= 4 and (gene_col_idx is not None):
        counts_like = (neg_rate <= 0.01) and (int_rate >= 0.60)

    degs_like = False
    if has_lfc:
        degs_like = True
    elif (has_q and n_numeric_cols <= 6 and neg_rate > 0.02):
        degs_like = True

    return {
        "ok": True,
        "header": header,
        "n_cols": n_cols,
        "n_rows_sampled": n_rows,
        "n_numeric_cols": n_numeric_cols,
        "gene_col_idx": gene_col_idx,
        "neg_rate": neg_rate,
        "int_rate": int_rate,
        "signals": {
            "has_lfc": has_lfc,
            "has_q": has_q,
            "degs_like": degs_like,
            "counts_like": counts_like,
            "gene_score_like": gene_score_like,
            "gene_list_score": gene_list_score,
        },
    }


def detect_auto_kind_from_path(p: Path) -> Tuple[str, Dict[str, Any]]:
    p = p.expanduser().resolve()
    if not p.exists():
        return AutoKind.KG_DISEASE, {"reason": "not_a_path"}

    if p.is_dir():
        files = _list_input_files(p)
        if not files:
            raise MissingResourceError(f"[ERROR] AUTO input dir has no supported files: {p}")
        preferred = sorted(files, key=lambda x: (x.suffix.lower() not in {".csv", ".tsv", ".txt"}, x.name.lower()))
        scored: List[Tuple[int, str, Dict[str, Any], Path]] = []
        for f in preferred[:10]:
            k, meta = detect_auto_kind_from_path(f)
            prio = {
                AutoKind.DEGS: 1,
                AutoKind.COUNTS: 2,
                AutoKind.GC_INPUT: 3,
                AutoKind.GL: 4,
                AutoKind.BUNDLE_JSON: 5,
            }.get(k, 99)
            scored.append((prio, k, meta, f))
        scored.sort(key=lambda t: t[0])
        best = scored[0]
        kind = best[1]
        meta = dict(best[2])
        meta["selected_file"] = str(best[3])
        meta["input_dir"] = str(p)
        return kind, meta

    # File
    if p.suffix.lower() == ".json":
        obj = _try_parse_json_file(p)
        if obj is not None and _looks_like_overlap_bundle(obj):
            return AutoKind.BUNDLE_JSON, {"reason": "overlap_bundle_json"}
        return AutoKind.GC_INPUT, {"reason": "json_payload_not_bundle"}

    if p.suffix.lower() in {".csv", ".tsv", ".txt"}:
        meta = _analyze_table_semantics(p)
        sig = meta.get("signals", {})
        if sig.get("degs_like"):
            return AutoKind.DEGS, {"reason": "lfc_or_degs_like", **meta}
        if sig.get("counts_like"):
            return AutoKind.COUNTS, {"reason": "counts_like", **meta}
        if sig.get("gene_score_like"):
            return AutoKind.GC_INPUT, {"reason": "gene_score_like", **meta}
        if sig.get("gene_list_score", 0.0) >= 0.70:
            return AutoKind.GL, {"reason": "gene_list_like", **meta}
        return AutoKind.GC_INPUT, {"reason": "fallback_gc_input", **meta}

    if p.suffix.lower() in {".xlsx", ".xls"}:
        if "deg" in p.name.lower() or "degs" in p.name.lower():
            return AutoKind.DEGS, {"reason": "excel_name_contains_deg"}
        return AutoKind.COUNTS, {"reason": "excel_default_counts"}

    return AutoKind.GC_INPUT, {"reason": f"unknown_suffix_{p.suffix.lower()}"}


def _materialize_file_to_dir(src_file: Path, dest_dir: Path) -> Path:
    _require_file(src_file, "AUTO source file")
    ensure_dir(dest_dir)
    dest = dest_dir / src_file.name
    try:
        dest.write_bytes(src_file.read_bytes())
    except Exception as e:
        raise OrchestratorError(f"[ERROR] Failed to copy input file to staging dir: {src_file} -> {dest} ({e})")
    return dest


def _kg_dir_from_args_or_env(args: argparse.Namespace) -> Optional[Path]:
    raw = (args.kg_dir or "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise MissingResourceError(f"[ERROR] --kg-dir is not a directory: {p}")
        return p
    for env_key in ["MDP_KG_DIR", "GENECARDS_KG_DIR", "KG_DIR"]:
        v = (os.environ.get(env_key) or "").strip()
        if v:
            p = Path(v).expanduser().resolve()
            if p.exists() and p.is_dir():
                return p
    return None


def _find_kg_payload_for_disease(kg_dir: Path, disease: str) -> Path:
    _require_dir(kg_dir, "KG directory")

    slug = _safe_slug(disease).lower()
    patterns = [
        f"*{slug}*.json",
        f"*{slug}*.csv",
        f"*{slug}*.tsv",
        f"*{slug}*.txt",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(sorted(kg_dir.glob(pat)))

    if hits:
        return hits[0]

    slug2 = slug.replace("_", "")
    loose_hits: List[Path] = []
    for ext in ["json", "csv", "tsv", "txt"]:
        loose_hits.extend(sorted(kg_dir.glob(f"*{slug2}*.{ext}")))
    if loose_hits:
        return loose_hits[0]

    raise MissingResourceError(
        "[ERROR] AUTO disease-name-only requires a KG/GeneCards payload file, but none was found.\n"
        f"  Disease: {disease}\n"
        f"  KG dir: {kg_dir}\n"
        "Expected a file matching e.g. '*<disease>*.csv' or '*<disease>*.json'.\n"
        "Fix: export a gene-score payload for this disease into the KG dir, or pass an explicit file input."
    )


def _harvest_and_convert_jsons_option_b(new_jsons: List[Path],
                                       dest_root: Path,
                                       disease: str) -> List[Path]:
    """
    Read each JSON, convert to Option B, write to dest_root as <disease>[__i].json.
    """
    written: List[Path] = []
    for i, src in enumerate(sorted(new_jsons)):
        _require_file(src, "Harvest source JSON")
        obj = _try_parse_json_file(src)
        if obj is None:
            print(f"[auto][warn] could not parse JSON (skipping): {src}", file=sys.stderr)
            continue

        norm = normalize_to_option_b(obj, force_up_if_directionless=True)

        desired = f"{_safe_slug(disease)}.json" if len(new_jsons) == 1 else f"{_safe_slug(disease)}__{i+1}.json"
        dest = _unique_dest_name(dest_root, desired)
        _write_json(dest, norm)
        written.append(dest)
    return written


@dataclass
class AutoRunRecord:
    disease: str
    kind: str
    source_tag: str
    input_ref: str
    staging_dir: str
    produced_jsons: List[str]
    classifier_root: str
    ok: bool
    error: str = ""


def _run_counts_one(disease: str,
                    counts_dir: Path,
                    tissue_in: str,
                    args: argparse.Namespace,
                    out_root: Path,
                    env: Dict[str, str]) -> Dict[str, str]:
    hint = f"input_folder={counts_dir.name}"
    tissue_resolved = resolve_tissue_or_die(
        disease=disease,
        tissue_in=tissue_in,
        hint=hint,
        allow_llm=(not args.no_llm_tissue),
    )
    resolved_tissues = {disease: tissue_resolved}
    (out_root / "resolved_tissues.json").write_text(
        json.dumps(resolved_tissues, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cohorts = [{"name": disease, "input": str(counts_dir), "tissue": tissue_resolved}]
    spec = build_counts_spec(
        cohorts=cohorts,
        out_root=out_root,
        id_col=args.counts_id_col,
        lfc_col=args.counts_lfc_col,
        q_col=args.counts_q_col,
        q_max=args.counts_q_max,
    )
    print(f"[auto][counts] cohorts=1 → counts_mdp.run_counts_from_dict → OUT_ROOT={wslpath(out_root)}")
    _invoke_counts_from_dict(spec, out_root, env=env, label=f"auto_counts_{_safe_slug(disease)}")
    return resolved_tissues


def _run_degs_one(disease: str,
                  degs_input: Path,
                  args: argparse.Namespace,
                  out_root: Path,
                  env: Dict[str, str]) -> None:
    if args.degs_id_col and args.degs_lfc_col and args.degs_q_col:
        id_col, lfc_col, q_col = args.degs_id_col, args.degs_lfc_col, args.degs_q_col
        print(f"[auto][degs] using user columns: id='{id_col}' lfc='{lfc_col}' q='{q_col}'")
    else:
        degs_file = pick_one_degs_file(degs_input)
        id_col, lfc_col, q_col = detect_degs_columns_from_file(degs_file)
        print(f"[auto][degs] autodetected from '{degs_file.name}': id='{id_col}' lfc='{lfc_col}' q='{q_col}'")

    cohorts = [{"name": disease, "input": str(degs_input)}]
    spec = build_degs_spec(
        cohorts=cohorts,
        out_root=out_root,
        id_col=id_col,
        lfc_col=lfc_col,
        q_col=q_col,
        q_max=args.degs_q_max,
    )
    print(f"[auto][degs] cohorts=1 → counts_mdp.run_counts_from_dict → OUT_ROOT={wslpath(out_root)}")
    _invoke_counts_from_dict(spec, out_root, env=env, label=f"auto_degs_{_safe_slug(disease)}")


def run_auto_mode(args: argparse.Namespace, out_root: Path, env: Dict[str, str]) -> List[AutoRunRecord]:
    unified_root = ensure_dir(out_root / "results" / "all_jsons")
    staging_root = ensure_dir(out_root / "_inputs")
    kg_dir = _kg_dir_from_args_or_env(args)

    raw_items: List[str] = []
    raw_items.extend(args.item or [])
    raw_items.extend(args.cohort or [])

    if not raw_items:
        raise UserInputError(
            "[ERROR] AUTO mode requires at least one --item (or --cohort).\n"
            "Examples:\n"
            "  --mode auto --out-root OUT --item /path/to/counts.csv --item 'Ulcerative Colitis'\n"
            "  --mode auto --out-root OUT --item name=IBD,input=/path/to/counts.csv,tissue=colon --item Parkinson\n"
        )

    candidates: List[Dict[str, Any]] = []
    disease_only: List[str] = []
    counts_like_paths_without_name: List[Path] = []

    for s in raw_items:
        s = (s or "").strip()
        if not s:
            continue

        if ("=" in s) and ("input" in s) and ("name" in s):
            kv = _parse_kv_triple(s)
            inp = Path(kv["input"]).expanduser().resolve()
            kind, meta = detect_auto_kind_from_path(inp)
            candidates.append({
                "disease": kv["name"].strip(),
                "input": str(inp),
                "input_path": inp,
                "tissue": kv.get("tissue", "").strip(),
                "kind": kind,
                "meta": meta,
                "explicit_name": True,
            })
            continue

        if _is_probably_path(s):
            inp = Path(s).expanduser().resolve()
            kind, meta = detect_auto_kind_from_path(inp)
            disease_guess = inp.stem
            candidates.append({
                "disease": disease_guess.strip(),
                "input": str(inp),
                "input_path": inp,
                "tissue": "",
                "kind": kind,
                "meta": meta,
                "explicit_name": False,
            })
            if kind == AutoKind.COUNTS:
                counts_like_paths_without_name.append(inp)
            continue

        disease_only.append(s.strip())

    if len(counts_like_paths_without_name) == 1 and disease_only:
        counts_path = counts_like_paths_without_name[0]
        for c in candidates:
            if (not c.get("explicit_name")) and c.get("kind") == AutoKind.COUNTS and c.get("input_path") == counts_path:
                assigned = disease_only.pop(0)
                print(f"[auto] mapping counts input '{counts_path.name}' -> disease='{assigned}'")
                c["disease"] = assigned
                break

    for d in disease_only:
        candidates.append({
            "disease": d,
            "input": d,
            "input_path": None,
            "tissue": "",
            "kind": AutoKind.KG_DISEASE,
            "meta": {"reason": "disease_name_only"},
            "explicit_name": True,
        })

    manifest: List[AutoRunRecord] = []

    for item in candidates:
        disease = (item["disease"] or "").strip() or "UnnamedDisease"
        kind = item["kind"]
        tissue_in = (item.get("tissue") or "").strip()
        input_path: Optional[Path] = item.get("input_path")
        meta = item.get("meta") or {}

        record = AutoRunRecord(
            disease=disease,
            kind=kind,
            source_tag="",
            input_ref=str(item.get("input") or ""),
            staging_dir="",
            produced_jsons=[],
            classifier_root="",
            ok=False,
            error="",
        )

        try:
            # ---------------------------
            # KG disease-name-only → GC → json_maker → normalize to Option B → write into unified_root
            # ---------------------------
            if kind == AutoKind.KG_DISEASE:
                stage_dir = ensure_dir(staging_root / _safe_slug(disease) / "kg_gc")
                payload = None
                source = ""
                try:
                    # Prefer local KG dir if provided
                    if kg_dir is not None:
                        payload = _find_kg_payload_for_disease(kg_dir, disease)
                        _materialize_file_to_dir(payload, stage_dir)
                        source = "kg_gc"
                    else:
                        # Fall back to live GeneCards API (genecards_api/app.py)
                        from .genecards_api.app import get_kg_data
                        resp = get_kg_data(disease)
                        if not isinstance(resp, dict) or resp.get("status") != "success":
                            raise MissingResourceError(f"[ERROR] GeneCards API failed for '{disease}': {resp}")
                        data = resp.get("data") or {}
                        if isinstance(data, dict) and "data" in data:
                            data = data["data"]
                        if not isinstance(data, dict) or not data.get("gene_symbol"):
                            raise MissingResourceError(f"[ERROR] GeneCards API returned no gene_symbol list for '{disease}'")
                        out_json = stage_dir / f"{_safe_slug(disease)}.json"
                        _write_json(out_json, {"data": data})
                        payload = out_json
                        source = "kg_gc_api"
                except Exception as e:
                    raise

                if payload is None:
                    raise MissingResourceError(f"[ERROR] No KG payload available for '{disease}'.")

                record.staging_dir = str(stage_dir)
                record.source_tag = source or "kg_gc"

                src_json_root = out_root / "GC_enrich" / "jsons_all_folder"
                before = _snapshot_jsons(src_json_root)

                gc_base = run_gc(stage_dir, out_root, env=env, gene_key=args.gene_key, score_key=args.score_key)
                record.classifier_root = str(gc_base)
                run_pathway_classifier(gc_base)

                build_json_bundle_for_gl_gc(gc_base, env=env)

                after = _snapshot_jsons(src_json_root)
                new_jsons = sorted(list(after - before))
                if not new_jsons:
                    slug = _safe_slug(disease).lower()
                    new_jsons = sorted([p for p in src_json_root.glob("*.json") if slug in p.stem.lower()])

                written = _harvest_and_convert_jsons_option_b(new_jsons, unified_root, disease)
                record.produced_jsons = [str(p) for p in written]
                record.ok = True
                manifest.append(record)
                continue

            if input_path is None:
                raise UserInputError(f"[ERROR] AUTO internal error: missing input_path for item: {item}")

            selected_file = Path(meta["selected_file"]).expanduser().resolve() if isinstance(meta, dict) and meta.get("selected_file") else None

            # ---------------------------
            # Import JSON bundle (Option B or legacy) → normalize to Option B → write into unified_root
            # ---------------------------
            if kind == AutoKind.BUNDLE_JSON:
                record.source_tag = "import"
                obj = _try_parse_json_file(input_path)
                if obj is None:
                    raise UserInputError(f"[ERROR] Could not parse JSON bundle: {input_path}")

                norm = normalize_to_option_b(obj, force_up_if_directionless=True)
                dest = _unique_dest_name(unified_root, f"{_safe_slug(disease)}.json")
                _write_json(dest, norm)

                record.produced_jsons = [str(dest)]
                record.ok = True
                manifest.append(record)
                continue

            # ---------------------------
            # DEGS: run counts_mdp, DO NOT touch produced JSONs
            # ---------------------------
            if kind == AutoKind.DEGS:
                degs_inp = input_path
                if input_path.is_dir() and selected_file:
                    degs_inp = input_path
                record.source_tag = "degs"
                record.staging_dir = str(degs_inp)

                json_root = ensure_dir(out_root / "results" / "all_jsons")
                before = _snapshot_jsons(json_root)

                _run_degs_one(disease=disease, degs_input=degs_inp, args=args, out_root=out_root, env=env)
                record.classifier_root = str(out_root)
                run_pathway_classifier(out_root)

                after = _snapshot_jsons(json_root)
                new_jsons = sorted(list(after - before))
                record.produced_jsons = [str(p) for p in new_jsons]  # untouched
                record.ok = True
                manifest.append(record)
                continue

            # ---------------------------
            # COUNTS: run counts_mdp, DO NOT touch produced JSONs
            # ---------------------------
            if kind == AutoKind.COUNTS:
                if input_path.is_file():
                    stage_dir = ensure_dir(staging_root / _safe_slug(disease) / "counts")
                    _materialize_file_to_dir(input_path, stage_dir)
                    counts_dir = stage_dir
                else:
                    counts_dir = input_path
                record.source_tag = "counts"
                record.staging_dir = str(counts_dir)

                json_root = ensure_dir(out_root / "results" / "all_jsons")
                before = _snapshot_jsons(json_root)

                _run_counts_one(disease=disease, counts_dir=counts_dir, tissue_in=tissue_in,
                                args=args, out_root=out_root, env=env)
                record.classifier_root = str(out_root)
                run_pathway_classifier(out_root)

                after = _snapshot_jsons(json_root)
                new_jsons = sorted(list(after - before))
                record.produced_jsons = [str(p) for p in new_jsons]  # untouched
                record.ok = True
                manifest.append(record)
                continue

            # ---------------------------
            # GL: run GL + json_maker → normalize new JSONs to Option B → write into unified_root
            # ---------------------------
            if kind == AutoKind.GL:
                if input_path.is_file():
                    stage_dir = ensure_dir(staging_root / _safe_slug(disease) / "gl")
                    _materialize_file_to_dir(input_path, stage_dir)
                    gl_dir = stage_dir
                else:
                    gl_dir = input_path
                record.source_tag = "gl"
                record.staging_dir = str(gl_dir)

                src_json_root = out_root / "GL_enrich" / "jsons_all_folder"
                before = _snapshot_jsons(src_json_root)

                gl_base = run_gl(gl_dir, out_root, env=env)
                record.classifier_root = str(gl_base)
                run_pathway_classifier(gl_base)

                build_json_bundle_for_gl_gc(gl_base, env=env)

                after = _snapshot_jsons(src_json_root)
                new_jsons = sorted(list(after - before))
                if not new_jsons:
                    slug = _safe_slug(disease).lower()
                    new_jsons = sorted([p for p in src_json_root.glob("*.json") if slug in p.stem.lower()])

                written = _harvest_and_convert_jsons_option_b(new_jsons, unified_root, disease)
                record.produced_jsons = [str(p) for p in written]
                record.ok = True
                manifest.append(record)
                continue

            # ---------------------------
            # GC_INPUT: run GC + json_maker → normalize new JSONs to Option B → write into unified_root
            # ---------------------------
            if kind == AutoKind.GC_INPUT:
                if input_path.is_file():
                    stage_dir = ensure_dir(staging_root / _safe_slug(disease) / "gc")
                    _materialize_file_to_dir(input_path, stage_dir)
                    gc_dir = stage_dir
                else:
                    gc_dir = input_path
                record.source_tag = "gc"
                record.staging_dir = str(gc_dir)

                src_json_root = out_root / "GC_enrich" / "jsons_all_folder"
                before = _snapshot_jsons(src_json_root)

                gc_base = run_gc(gc_dir, out_root, env=env, gene_key=args.gene_key, score_key=args.score_key)
                record.classifier_root = str(gc_base)
                run_pathway_classifier(gc_base)

                build_json_bundle_for_gl_gc(gc_base, env=env)

                after = _snapshot_jsons(src_json_root)
                new_jsons = sorted(list(after - before))
                if not new_jsons:
                    slug = _safe_slug(disease).lower()
                    new_jsons = sorted([p for p in src_json_root.glob("*.json") if slug in p.stem.lower()])

                written = _harvest_and_convert_jsons_option_b(new_jsons, unified_root, disease)
                record.produced_jsons = [str(p) for p in written]
                record.ok = True
                manifest.append(record)
                continue

            raise UserInputError(f"[ERROR] AUTO unsupported kind: {kind}")

        except Exception as e:
            record.ok = False
            record.error = str(e)
            manifest.append(record)
            print(f"[auto][warn] item failed for disease='{disease}' kind='{kind}': {e}", file=sys.stderr)
            continue

    manifest_path = out_root / "RUN_MANIFEST.json"
    _write_json(manifest_path, {
        "mode": "auto",
        "out_root": str(out_root),
        "unified_json_root": str(unified_root),
        "items": [
            {
                "disease": r.disease,
                "kind": r.kind,
                "source_tag": r.source_tag,
                "input_ref": r.input_ref,
                "staging_dir": r.staging_dir,
                "classifier_root": r.classifier_root,
                "produced_jsons": r.produced_jsons,
                "ok": r.ok,
                "error": r.error,
            }
            for r in manifest
        ],
    })
    print(f"[auto] Wrote run manifest: {wslpath(manifest_path)}")

    return manifest


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="COUNTS/DEGS/GL/GC runner → classifier → (pathway_compare|mdp_insights) → report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mode", required=True, choices=["counts", "degs", "gl", "gc", "auto"])
    ap.add_argument("--input", help="Input folder/file for single-cohort modes.")
    ap.add_argument("--out-root", required=True, help="Base output folder for the selected mode.")

    ap.add_argument("--cohort", action="append", default=[], help="Repeatable: name=<NAME>,input=<DIR|FILE>[,tissue=<TISSUE>]")

    # AUTO: mixed inputs (paths or disease names)
    ap.add_argument(
        "--item",
        action="append",
        default=[],
        help="AUTO mode: repeatable mixed items. Each item can be a PATH, a disease name string, or 'name=...,input=...[,tissue=...]'.",
    )
    ap.add_argument(
        "--kg-dir",
        default="",
        help="AUTO mode: directory holding inbuilt KG/GeneCards payload files for disease-name-only items "
             "(can also be set via env MDP_KG_DIR / GENECARDS_KG_DIR).",
    )

    # counts single
    ap.add_argument("--disease-name", default="", help="Cohort name for single-cohort runs (counts/degs).")
    ap.add_argument("--tissue", default="", help="COUNTS single: tissue label (optional; auto-resolve).")
    ap.add_argument("--no-llm-tissue", action="store_true", help="Disable LLM tissue fallback.")

    # counts spec knobs
    ap.add_argument("--counts-id-col", default="Gene")
    ap.add_argument("--counts-lfc-col", default="log2FoldChange")
    ap.add_argument("--counts-q-col", default="")
    ap.add_argument("--counts-q-max", type=float, default=0.05)

    # degs knobs
    ap.add_argument("--degs-id-col", default="", help="Override gene id column (else autodetect).")
    ap.add_argument("--degs-lfc-col", default="", help="Override LFC column (else autodetect).")
    ap.add_argument("--degs-q-col", default="", help="Override q/padj column (else autodetect).")
    ap.add_argument("--degs-q-max", type=float, default=0.10)

    # gc/gl knobs
    ap.add_argument("--gene-key", default="gene_symbol")
    ap.add_argument("--score-key", default="gene_score")

    # extra steps
    ap.add_argument("--run-enzymes", action="store_true")

    # terminal
    ap.add_argument("--pathways", action="append", default=[])
    ap.add_argument("--direction-mode", default="both", choices=["both", "up", "down"])
    ap.add_argument("--pc-out", default="PC_out")
    ap.add_argument("--insights-out", default="INSIGHTS_out")

    # report
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--report-no-llm", action="store_true")
    ap.add_argument("--report-q-cutoff", type=float, default=0.05)

    return ap.parse_args()


# =============================================================================
# main
# =============================================================================

def _install_signal_handlers() -> None:
    def _handler(sig: int, _frame: Any) -> None:
        raise KeyboardInterrupt()
    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


def main() -> int:
    _install_signal_handlers()
    args = parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    ensure_dir(out_root)

    forced_env = {"OUT_ROOT": str(out_root), "MDP_OUT_ROOT": str(out_root)}

    print(f"[orchestrator] MODE={args.mode.upper()}")
    print(f"[orchestrator] OUT_ROOT={wslpath(out_root)}")

    # classifier root for non-AUTO
    if args.mode == "gl":
        classifier_root = out_root / "GL_enrich"
    elif args.mode == "gc":
        classifier_root = out_root / "GC_enrich"
    else:
        classifier_root = out_root

    resolved_tissues: Dict[str, str] = {}
    auto_manifest: List[AutoRunRecord] = []

    # 1) Run pipeline
    if args.mode == "gc":
        if not args.input:
            raise UserInputError("--input is required for GC mode")
        gc_base = run_gc(Path(args.input).expanduser().resolve(), out_root, env=forced_env,
                         gene_key=args.gene_key, score_key=args.score_key)
        build_json_bundle_for_gl_gc(gc_base, env=forced_env)
        # NOTE: In GC-only mode, jsons_all_folder remains as produced. (AUTO is where we unify/convert.)

    elif args.mode == "gl":
        if not args.input:
            raise UserInputError("--input is required for GL mode")
        gl_base = run_gl(Path(args.input).expanduser().resolve(), out_root, env=forced_env)
        build_json_bundle_for_gl_gc(gl_base, env=forced_env)

    elif args.mode == "counts":
        resolved_tissues = run_counts_mode(args, out_root, env=forced_env)

    elif args.mode == "degs":
        run_degs_mode(args, out_root, env=forced_env)

    elif args.mode == "auto":
        auto_manifest = run_auto_mode(args, out_root, env=forced_env)

    else:
        raise UserInputError("Unsupported mode.")

    # 2) classifier
    if args.mode != "auto":
        if classifier_root.exists():
            run_pathway_classifier(classifier_root)
        else:
            print(f"[orchestrator][warn] classifier_root not found (skipping): {classifier_root}", file=sys.stderr)

    # 3) Strict JSON root
    json_root = resolve_json_root(args.mode, out_root)
    print(f"[orchestrator] JSON root (STRICT): {wslpath(json_root)}")

    # 4) Optional enzymes
    if args.run_enzymes:
        if args.mode in {"counts", "degs", "auto"}:
            print("[orchestrator] Running enzyme_and_signaling.py (batch) on OUT_ROOT...")
            run_enzyme_and_signaling_batch(out_root, env=forced_env)
        else:
            print("[orchestrator] NOTE: --run-enzymes ignored for GL/GC.")

    # 5) Terminal
    if args.pathways:
        pc_out = Path(args.pc_out)
        if not pc_out.is_absolute():
            pc_out = out_root / pc_out
        print(f"[orchestrator] Running pathway_compare → {wslpath(pc_out)}")
        run_pathway_compare(json_root=json_root, out_dir=pc_out, pathways=args.pathways,
                            direction=args.direction_mode, env=forced_env)
    else:
        ins_out = Path(args.insights_out)
        if not ins_out.is_absolute():
            ins_out = out_root / ins_out
        print(f"[orchestrator] Running mdp_insights → {wslpath(ins_out)}")
        run_mdp_insights(json_root=json_root, out_dir=ins_out, env=forced_env)

    # 6) Report
    if args.skip_report:
        print("[orchestrator] Skipping report generation (--skip-report).")
    else:
        ensure_report_has_classified_gsea(out_root)
        print("[orchestrator] Generating final MDP Report (HTML + PDF)...")
        run_mdp_report_plan2(out_root=out_root, env=forced_env,
                             no_llm=bool(args.report_no_llm), q_cutoff=float(args.report_q_cutoff))

    # Summary
    if resolved_tissues:
        print("[orchestrator] Resolved tissues:")
        for k, v in sorted(resolved_tissues.items()):
            print(f"  - {k}: {v}")

    if auto_manifest:
        ok = sum(1 for r in auto_manifest if r.ok)
        bad = sum(1 for r in auto_manifest if not r.ok)
        print(f"[orchestrator] AUTO items completed: ok={ok} failed={bad}")
        for r in auto_manifest:
            status = "OK" if r.ok else "FAIL"
            print(f"  - [{status}] {r.disease} ({r.kind}) -> {len(r.produced_jsons)} json(s)")

    print("[orchestrator] DONE.")
    return 0



def run_pipeline(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root).expanduser().resolve()
    ensure_dir(out_root)

    forced_env = {"OUT_ROOT": str(out_root), "MDP_OUT_ROOT": str(out_root)}

    print(f"[orchestrator] MODE={args.mode.upper()}")
    print(f"[orchestrator] OUT_ROOT={wslpath(out_root)}")

    # classifier root for non-AUTO
    if args.mode == "gl":
        classifier_root = out_root / "GL_enrich"
    elif args.mode == "gc":
        classifier_root = out_root / "GC_enrich"
    else:
        classifier_root = out_root

    resolved_tissues: Dict[str, str] = {}
    auto_manifest: List[AutoRunRecord] = []

    # 1) Run pipeline
    if args.mode == "gc":
        if not args.input:
            raise UserInputError("--input is required for GC mode")
        gc_base = run_gc(Path(args.input).expanduser().resolve(), out_root, env=forced_env,
                         gene_key=args.gene_key, score_key=args.score_key)
        build_json_bundle_for_gl_gc(gc_base, env=forced_env)
        # NOTE: In GC-only mode, jsons_all_folder remains as produced. (AUTO is where we unify/convert.)

    elif args.mode == "gl":
        if not args.input:
            raise UserInputError("--input is required for GL mode")
        gl_base = run_gl(Path(args.input).expanduser().resolve(), out_root, env=forced_env)
        build_json_bundle_for_gl_gc(gl_base, env=forced_env)

    elif args.mode == "counts":
        resolved_tissues = run_counts_mode(args, out_root, env=forced_env)

    elif args.mode == "degs":
        run_degs_mode(args, out_root, env=forced_env)

    elif args.mode == "auto":
        auto_manifest = run_auto_mode(args, out_root, env=forced_env)

    else:
        raise UserInputError("Unsupported mode.")

    # 2) classifier
    if args.mode != "auto":
        if classifier_root.exists():
            run_pathway_classifier(classifier_root)
        else:
            print(f"[orchestrator][warn] classifier_root not found (skipping): {classifier_root}", file=sys.stderr)

    # 3) Strict JSON root
    json_root = resolve_json_root(args.mode, out_root)
    print(f"[orchestrator] JSON root (STRICT): {wslpath(json_root)}")

    # 4) Optional enzymes
    if args.run_enzymes:
        if args.mode in {"counts", "degs", "auto"}:
            print("[orchestrator] Running enzyme_and_signaling.py (batch) on OUT_ROOT...")
            run_enzyme_and_signaling_batch(out_root, env=forced_env)
        else:
            print("[orchestrator] NOTE: --run-enzymes ignored for GL/GC.")

    # 5) Terminal
    if args.pathways:
        pc_out = Path(args.pc_out)
        if not pc_out.is_absolute():
            pc_out = out_root / pc_out
        print(f"[orchestrator] Running pathway_compare → {wslpath(pc_out)}")
        run_pathway_compare(json_root=json_root, out_dir=pc_out, pathways=args.pathways,
                            direction=args.direction_mode, env=forced_env)
    else:
        ins_out = Path(args.insights_out)
        if not ins_out.is_absolute():
            ins_out = out_root / ins_out
        print(f"[orchestrator] Running mdp_insights → {wslpath(ins_out)}")
        run_mdp_insights(json_root=json_root, out_dir=ins_out, env=forced_env)

    # 6) Report
    if args.skip_report:
        print("[orchestrator] Skipping report generation (--skip-report).")
    else:
        ensure_report_has_classified_gsea(out_root)
        print("[orchestrator] Generating final MDP Report (HTML + PDF)...")
        run_mdp_report_plan2(out_root=out_root, env=forced_env,
                             no_llm=bool(args.report_no_llm), q_cutoff=float(args.report_q_cutoff))

    # Summary
    if resolved_tissues:
        print("[orchestrator] Resolved tissues:")
        for k, v in sorted(resolved_tissues.items()):
            print(f"  - {k}: {v}")

    if auto_manifest:
        ok = sum(1 for r in auto_manifest if r.ok)
        bad = sum(1 for r in auto_manifest if not r.ok)
        print(f"[orchestrator] AUTO items completed: ok={ok} failed={bad}")
        for r in auto_manifest:
            status = "OK" if r.ok else "FAIL"
            print(f"  - [{status}] {r.disease} ({r.kind}) -> {len(r.produced_jsons)} json(s)")

    print("[orchestrator] DONE.")
    return 0
    


def run_full_pipeline(
    mode: Optional[str] = "auto",
    input: Optional[Path | str] = None,
    out_root: Optional[Path | str] = "output",
    cohort: Optional[Sequence[str]] = None,
    item: Optional[Sequence[str]] = None,
    kg_dir: str = "",
    disease_name: str = "",
    tissue: str = "",
    no_llm_tissue: bool = False,
    counts_id_col: str = "Gene",
    counts_lfc_col: str = "log2FoldChange",
    counts_q_col: str = "",
    counts_q_max: float = 0.05,
    degs_id_col: str = "",
    degs_lfc_col: str = "",
    degs_q_col: str = "",
    degs_q_max: float = 0.10,
    gene_key: str = "gene_symbol",
    score_key: str = "gene_score",
    run_enzymes: bool = True,
    pathways: Optional[Sequence[str]] = None,
    direction_mode: str = "both",
    pc_out: str = "PC_out",
    insights_out: str = "INSIGHTS_out",
    skip_report: bool = False,
    report_no_llm: bool = False,
    report_q_cutoff: float = 0.05,
    env: Optional[Dict[str, str]] = None,
) -> int:
    """
    Convenience wrapper so callers can supply the same parameters as main()
    via explicit named arguments (no **kwargs required).
    """
    args = argparse.Namespace(
        mode=mode,
        input=input,
        out_root=out_root,
        cohort=cohort,
        item=item,
        kg_dir=kg_dir,
        disease_name=disease_name,
        tissue=tissue,
        no_llm_tissue=no_llm_tissue,
        counts_id_col=counts_id_col,
        counts_lfc_col=counts_lfc_col,
        counts_q_col=counts_q_col,
        counts_q_max=counts_q_max,
        degs_id_col=degs_id_col,
        degs_lfc_col=degs_lfc_col,
        degs_q_col=degs_q_col,
        degs_q_max=degs_q_max,
        gene_key=gene_key,
        score_key=score_key,
        run_enzymes=run_enzymes,
        pathways=pathways,
        direction_mode=direction_mode,
        pc_out=pc_out,
        insights_out=insights_out,
        skip_report=skip_report,
        report_no_llm=report_no_llm,
        report_q_cutoff=report_q_cutoff,
        env=env,
    )

    return run_pipeline(args)



# if __name__ == "__main__":
#     try:
#         raise SystemExit(main())
#     except KeyboardInterrupt:
#         print("\n[orchestrator] Interrupted by user (Ctrl+C).", file=sys.stderr)
#         raise SystemExit(130)
#     except OrchestratorError as e:
#         print(str(e), file=sys.stderr)
#         raise SystemExit(2)
#     except Exception as e:
#         print(f"[FATAL] Unhandled exception: {e}", file=sys.stderr)
#         raise SystemExit(1)
