from __future__ import annotations
import json, logging, re, unicodedata, difflib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_jsons(json_root: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    files = sorted(json_root.glob("*.json"))
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and data:
                out[fp.stem] = data
            else:
                logging.warning(f"[skip] {fp.name}: empty or not a dict")
        except Exception as e:
            logging.error(f"[error] reading {fp.name}: {e}")
    return out

def read_pathway_list(inline: str = "", file_path: str = "", csv_col: str = "") -> List[str]:
    vals: List[str] = []
    if inline:
        vals.extend([v.strip() for v in inline.split(",") if v.strip()])
    if file_path:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            logging.error(f"pathways-file not found: {p}")
        elif p.suffix.lower() in {".txt", ".list"}:
            with p.open("r", encoding="utf-8") as fh:
                vals.extend([ln.strip() for ln in fh if ln.strip()])
        else:
            df = pd.read_csv(p)
            col = csv_col or df.columns[0]
            vals.extend([str(x).strip() for x in df[col].dropna().tolist()])
    # unique preserving order
    seen = set(); out = []
    for v in vals:
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out

# Backward-compatible alias (used by main.py)
def read_pathways_from_cli_or_file(inline: str = "", file_path: str = "", csv_col: str = "") -> List[str]:
    return read_pathway_list(inline=inline, file_path=file_path, csv_col=csv_col)

# ---- optional safe writers (no-ops if you already import from elsewhere) ----
def safe_write_csv(df: pd.DataFrame, path: Path) -> Optional[Path]:
    try:
        ensure_dir(path.parent)
        df.to_csv(path, index=False)
        return path
    except Exception as e:
        logging.error(f"[write_csv] {path}: {e}")
        return None

def safe_write_xlsx(dfs: Dict[str, pd.DataFrame], path: Path) -> Optional[Path]:
    try:
        ensure_dir(path.parent)
        with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
            for name, df in dfs.items():
                df.to_excel(xw, sheet_name=str(name)[:31], index=False)
        return path
    except Exception as e:
        logging.error(f"[write_xlsx] {path}: {e}")
        return None

# ---- name/ID matching ----
_ID_PATTERNS = [
    re.compile(r"\bKEGG:(\d+)\b", re.I),
    re.compile(r"\bhsa(\d{5})\b", re.I),
    re.compile(r"\bR-HSA-(\d+)\b", re.I),
    re.compile(r"\bWP(\d+)\b", re.I),
]
_punct_re = re.compile(r"[^\w\s]+")

def _ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _norm_text(s: str) -> str:
    s = _ascii_fold(s or "")
    s = s.lower().strip()
    s = _punct_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _extract_ids(s: str) -> List[str]:
    out = []
    for rx in _ID_PATTERNS:
        m = rx.search(s or "")
        if m:
            out.append(m.group(0).upper())
    return out

def build_pathway_alias_map(all_json: Dict[str, dict]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      name_key_to_canonical: normalized name -> canonical label
      id_to_canonical:       ID token (e.g. 'R-HSA-12345') -> canonical label
    """
    name_key_to_canonical: Dict[str, str] = {}
    id_to_canonical: Dict[str, str] = {}

    labels: List[str] = []
    for _, obj in all_json.items():
        if not isinstance(obj, dict):
            continue
        for pathway_label in obj.keys():
            if pathway_label not in labels:
                labels.append(pathway_label)

    for lab in labels:
        canon = str(lab)
        name_key = _norm_text(canon)
        name_key_to_canonical.setdefault(name_key, canon)
        for idtok in _extract_ids(canon):
            id_to_canonical.setdefault(idtok.upper(), canon)

    return name_key_to_canonical, id_to_canonical

def resolve_requested_pathways(
    requested: List[str],
    alias_by_name: Dict[str, str],
    alias_by_id: Dict[str, str],
    *,
    enable_id_matching: bool = True,
    enable_name_fallback: bool = True,
    fail_on_missing: bool = False,
) -> Tuple[List[str], List[str]]:
    resolved: List[str] = []
    missing: List[str] = []
    seen = set()

    for token in requested:
        raw = token.strip()
        if not raw:
            continue

        # try ID match first
        if enable_id_matching:
            ids = _extract_ids(raw)
            if not ids and re.match(r"^(KEGG:\d+|R-HSA-\d+|hsa\d{5}|WP\d+)$", raw, re.I):
                ids = [raw]
            matched = False
            for idt in ids:
                canon = alias_by_id.get(idt.upper())
                if canon and canon not in seen:
                    resolved.append(canon); seen.add(canon); matched = True; break
            if matched:
                continue
            # raw as exact id token
            canon = alias_by_id.get(raw.upper())
            if canon and canon not in seen:
                resolved.append(canon); seen.add(canon); continue

        # name fallback
        if enable_name_fallback:
            key = _norm_text(raw)
            canon = alias_by_name.get(key)
            if canon and canon not in seen:
                resolved.append(canon); seen.add(canon); continue

        missing.append(raw)

    if missing:
        msg = f"Unresolved pathways (not found in JSON labels): {missing}"
        if fail_on_missing:
            raise ValueError(msg)
        logging.warning(msg)

    return resolved, missing

# ------------------- FUZZY RESOLUTION (ROBUST) -------------------

def _token_set_ratio(a: str, b: str) -> float:
    A = set(_norm_text(a).split())
    B = set(_norm_text(b).split())
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def _collect_all_labels(alias_by_name: Dict[str, str]) -> List[str]:
    seen = set(); out: List[str] = []
    for lab in alias_by_name.values():
        if lab not in seen:
            seen.add(lab); out.append(lab)
    return out

def resolve_requested_pathways_fuzzy(
    requested: List[str],
    alias_by_name: Dict[str, str],
    alias_by_id: Dict[str, str],
    *,
    min_score: float = 0.66,
    topk: int = 5,
    enable_id_matching: bool = True,
) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[str, str]]:
    """
    Robust resolver:
      1) ID tokens (KEGG:nnnn, R-HSA-nnnnn, hsaNNNNN, WPnnnn)
      2) Exact normalized name
      3) Fuzzy (token-set + difflib), accept if score>=min_score

    Returns:
      resolved_labels, unresolved_raw_inputs, suggestions (raw->[labels]), mapping (raw->resolved_label)
    """
    all_labels = _collect_all_labels(alias_by_name)

    resolved: List[str] = []
    unresolved: List[str] = []
    suggestions: Dict[str, List[str]] = {}
    mapping: Dict[str, str] = {}
    seen = set()

    for raw in requested:
        q = (raw or "").strip()
        if not q:
            continue

        # 1) ID-first
        matched = False
        if enable_id_matching:
            ids = _extract_ids(q)
            if not ids and re.match(r"^(KEGG:\d+|R-HSA-\d+|hsa\d{5}|WP\d+)$", q, re.I):
                ids = [q]
            for idt in ids:
                lab = alias_by_id.get(idt.upper())
                if lab and lab not in seen:
                    resolved.append(lab); seen.add(lab); mapping[q] = lab; matched = True
                    break
            if matched:
                continue
            lab = alias_by_id.get(q.upper())
            if lab and lab not in seen:
                resolved.append(lab); seen.add(lab); mapping[q] = lab
                continue

        # 2) exact normalized name
        key = _norm_text(q)
        lab = alias_by_name.get(key)
        if lab and lab not in seen:
            resolved.append(lab); seen.add(lab); mapping[q] = lab
            continue

        # 3) fuzzy names
        scored = []
        nq = _norm_text(q)
        for lbl in all_labels:
            s1 = _token_set_ratio(nq, lbl)
            s2 = difflib.SequenceMatcher(None, _norm_text(lbl), nq).ratio()
            s = max(s1, s2)
            scored.append((s, lbl))
        scored.sort(key=lambda x: (-x[0], x[1]))
        cand = [lbl for s, lbl in scored[:topk]]
        suggestions[q] = cand

        if scored and scored[0][0] >= min_score:
            lab = scored[0][1]
            if lab not in seen:
                resolved.append(lab); seen.add(lab); mapping[q] = lab
        else:
            unresolved.append(q)

    return resolved, unresolved, suggestions, mapping
