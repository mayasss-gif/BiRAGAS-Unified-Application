from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd

from mdp_engine.exceptions import DataError, ValidationError
from mdp_engine.logging_utils import get_logger

log = get_logger("mdp_engine.engines.causal_pathway_context")


@dataclass(frozen=True)
class ContextPaths:
    out_dir: Path
    pkn_edges: Path
    causal_edges: Path
    causal_nodes: Path
    drivers_ranked: Path
    mechanism_cards_json: Path
    mechanism_cards_tsv: Path
    manifest: Path
    skipped_flag: Path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _cohort_dir(out_root: Path, disease: str) -> Path:
    d1 = out_root / disease
    d2 = out_root / "cohorts" / disease
    if d1.exists():
        return d1
    if d2.exists():
        return d2
    raise DataError(f"Cannot find cohort folder for '{disease}' in {out_root} (tried {d1} and {d2}).")


def _find_engine1_tf_activity(out_root: Path, disease: str) -> Optional[Path]:
    p = out_root / "engines" / "causal_pathway_features" / disease / "tf_activity.tsv"
    return p if p.exists() else None


def _find_engine1_footprints(out_root: Path, disease: str) -> Optional[Path]:
    p = out_root / "engines" / "causal_pathway_features" / disease / "pathway_footprint_activity.tsv"
    return p if p.exists() else None


def _find_engine2_confounding(out_root: Path, disease: str) -> Optional[Path]:
    p = out_root / "engines" / "confounding" / disease / "confounding_report.tsv"
    return p if p.exists() else None


def _find_pkn_edges(out_root: Path) -> Optional[Path]:
    candidates = [
        out_root / "engines" / "pkn_cache" / "pkn_edges.tsv",
        out_root / "data" / "omnipath_cache" / "edges.tsv",
        out_root / "pkn_edges.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _read_indexed_table(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        raise DataError(f"Missing file: {path}")
    suf = path.suffix.lower()
    sep = "\t" if suf in {".tsv", ".txt"} else ","
    try:
        df = pd.read_csv(path, sep=sep, index_col=0)
    except Exception as e:
        raise DataError(f"Failed reading indexed table: {path} ({e})") from e
    if df is None or df.empty:
        raise DataError(f"Empty table: {path}")
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df


def _normalize_pkn_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}

    src = low.get("source") or low.get("src") or low.get("from") or low.get("a")
    dst = low.get("target") or low.get("dst") or low.get("to") or low.get("b")
    if src is None or dst is None:
        raise DataError("PKN edges must include source/target (or src/dst).")

    df = df.rename(columns={src: "source", dst: "target"})
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()

    sign_col = low.get("sign")
    inter_col = low.get("interaction") or low.get("effect")

    if sign_col is not None:
        s = pd.to_numeric(df[sign_col], errors="coerce").fillna(0.0)
        df["sign"] = s
    elif inter_col is not None:
        txt = df[inter_col].astype(str).str.lower()
        df["sign"] = 0.0
        df.loc[txt.str.contains("activ"), "sign"] = 1.0
        df.loc[txt.str.contains("inhib"), "sign"] = -1.0
    else:
        df["sign"] = 0.0

    df["sign"] = df["sign"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df = df[(df["source"] != "") & (df["target"] != "")]
    df = df.drop_duplicates(subset=["source", "target", "sign"])
    return df[["source", "target", "sign"]]


def _build_incoming(edges: pd.DataFrame) -> Dict[str, List[Tuple[str, int]]]:
    inc: Dict[str, List[Tuple[str, int]]] = {}
    for _, r in edges.iterrows():
        s = str(r["source"])
        t = str(r["target"])
        sg = int(r["sign"])
        if sg == 0:
            continue
        inc.setdefault(t, []).append((s, sg))
    return inc


def _bfs_upstream(incoming: Dict[str, List[Tuple[str, int]]], tf: str, max_steps: int) -> Dict[Tuple[str, int], int]:
    """
    BFS upstream from TF using incoming adjacency.
    State: (node, sign_to_tf) where sign_to_tf is product of signs along the path.
    Returns best depth for each state.
    """
    start = (tf, 1)
    best: Dict[Tuple[str, int], int] = {start: 0}
    q = deque([(tf, 1, 0)])

    while q:
        node, sign_to_tf, depth = q.popleft()
        if depth >= max_steps:
            continue
        for src, esign in incoming.get(node, []):
            new_sign = int(sign_to_tf * esign)
            st = (src, new_sign)
            nd = depth + 1
            if st not in best or nd < best[st]:
                best[st] = nd
                q.append((src, new_sign, nd))
    return best


def _normalize_tf_columns(cols: List[str]) -> List[str]:
    out: List[str] = []
    for c in cols:
        s = str(c).strip()
        if s.startswith("TF:"):
            s = s.replace("TF:", "", 1)
        out.append(s.upper())
    return out


def _footprint_consistency_for_tf(
    *,
    tf_series: pd.Series,
    footprints_df: pd.DataFrame,
    min_samples: int = 6,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Minimal, no-surprises footprint consistency check:
      - Align samples between TF activity and footprint activity
      - Compute Pearson correlation between TF vector and each footprint column
      - Return (best_abs_corr_signed, best_col_name)
    If cannot compute robustly, returns (None, None).
    """
    if tf_series is None or tf_series.empty:
        return None, None
    if footprints_df is None or footprints_df.empty:
        return None, None

    # align by sample IDs
    common = tf_series.index.intersection(footprints_df.index)
    if len(common) < int(min_samples):
        return None, None

    y = pd.to_numeric(tf_series.loc[common], errors="coerce")
    if y.dropna().nunique() < 2:
        return None, None

    X = footprints_df.loc[common].apply(pd.to_numeric, errors="coerce")
    # drop constant columns
    nunq = X.nunique(dropna=True)
    X = X.loc[:, nunq >= 2]
    if X.shape[1] == 0:
        return None, None

    corr = X.corrwith(y, axis=0)
    corr = corr.dropna()
    if corr.empty:
        return None, None

    best_col = corr.abs().idxmax()
    best_corr = float(corr.loc[best_col])
    return best_corr, str(best_col)


def run_contextualization_engine_for_disease(
    out_root: Path,
    disease: str,
    *,
    pkn_edges_path: Optional[Path] = None,
    max_steps: int = 3,
    top_tfs: int = 30,
    confound_penalty_threshold: float = 0.40,
    strict: bool = False,
) -> ContextPaths:
    """
    Engine 3: directed signed plausibility routes consistent with TF activity and a signed PKN.

    Outputs:
      OUT_ROOT/engines/causal_pathway_context/<Disease>/
        pkn_edges.tsv
        causal_subnetwork_edges.tsv
        causal_subnetwork_nodes.tsv
        drivers_ranked.tsv
        mechanism_cards.json + mechanism_cards.tsv
        ENGINE_MANIFEST.json
        SKIPPED.txt (if skipped)

    Robust behavior:
      - Missing TF activity or PKN -> SKIPPED (unless strict=True)
      - No subnetwork formed -> SKIPPED (unless strict=True)

    Confidence behavior (updated, minimal):
      - High/Medium now requires *measured footprint consistency* (not just footprints file presence).
        Consistency is currently defined as best absolute correlation between TF per-sample activity
        and any footprint activity column after sample alignment.
    """
    out_root = Path(out_root).resolve()
    if not out_root.exists():
        raise ValidationError(f"out_root does not exist: {out_root}")

    _ = _cohort_dir(out_root, disease)  # validate disease exists

    out_dir = out_root / "engines" / "causal_pathway_context" / disease
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = ContextPaths(
        out_dir=out_dir,
        pkn_edges=out_dir / "pkn_edges.tsv",
        causal_edges=out_dir / "causal_subnetwork_edges.tsv",
        causal_nodes=out_dir / "causal_subnetwork_nodes.tsv",
        drivers_ranked=out_dir / "drivers_ranked.tsv",
        mechanism_cards_json=out_dir / "mechanism_cards.json",
        mechanism_cards_tsv=out_dir / "mechanism_cards.tsv",
        manifest=out_dir / "ENGINE_MANIFEST.json",
        skipped_flag=out_dir / "SKIPPED.txt",
    )

    tf_path = _find_engine1_tf_activity(out_root, disease)
    if tf_path is None:
        msg = f"Engine3 skipped: missing Engine1 tf_activity.tsv for {disease}"
        if strict:
            raise DataError(msg)
        log.warning(msg)
        _atomic_write_text(paths.skipped_flag, msg + "\n")
        _atomic_write_text(paths.manifest, json.dumps({
            "engine": "causal_pathway_context",
            "version": "1.0.0",
            "status": "skipped",
            "reason": msg,
            "outputs": {"skipped_flag": str(paths.skipped_flag)},
        }, indent=2))
        return paths

    pkn_path = Path(pkn_edges_path) if pkn_edges_path is not None else _find_pkn_edges(out_root)
    if pkn_path is None or not pkn_path.exists():
        msg = "Engine3 skipped: PKN edges file not found."
        if strict:
            raise DataError(msg)
        log.warning(msg)
        _atomic_write_text(paths.skipped_flag, msg + "\n")
        _atomic_write_text(paths.manifest, json.dumps({
            "engine": "causal_pathway_context",
            "version": "1.0.0",
            "status": "skipped",
            "reason": msg,
            "inputs": {"tf_activity": str(tf_path)},
            "outputs": {"skipped_flag": str(paths.skipped_flag)},
        }, indent=2))
        return paths

    footprints_path = _find_engine1_footprints(out_root, disease)
    footprints_available = footprints_path is not None and footprints_path.exists()

    # Load TF activity
    tf_df_raw = _read_indexed_table(tf_path)
    tf_df = tf_df_raw.copy()
    tf_df.columns = _normalize_tf_columns(list(tf_df.columns))
    tf_df = tf_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if tf_df.empty:
        msg = f"Engine3 skipped: TF activity table empty: {tf_path}"
        if strict:
            raise DataError(msg)
        log.warning(msg)
        _atomic_write_text(paths.skipped_flag, msg + "\n")
        _atomic_write_text(paths.manifest, json.dumps({
            "engine": "causal_pathway_context",
            "version": "1.0.0",
            "status": "skipped",
            "reason": msg,
            "inputs": {"tf_activity": str(tf_path), "pkn_edges": str(pkn_path)},
            "outputs": {"skipped_flag": str(paths.skipped_flag)},
        }, indent=2))
        return paths

    tf_mean = tf_df.mean(axis=0)
    tf_mean = tf_mean.sort_values(key=lambda s: s.abs(), ascending=False)
    tf_keep = list(tf_mean.index[: int(top_tfs)])

    # Load footprints (optional)
    footprints_df: Optional[pd.DataFrame] = None
    footprints_load_error: Optional[str] = None
    if footprints_available and footprints_path is not None:
        try:
            footprints_df = _read_indexed_table(footprints_path)
            footprints_df = footprints_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        except Exception as e:
            footprints_df = None
            footprints_load_error = str(e)

    # Confounding penalties (optional)
    conf_path = _find_engine2_confounding(out_root, disease)
    conf_pen: Dict[str, float] = {}
    if conf_path is not None:
        try:
            cdf = pd.read_csv(conf_path, sep="\t")
            if "feature" in cdf.columns and "penalty" in cdf.columns:
                for _, r in cdf.iterrows():
                    feat = str(r["feature"])
                    if feat.startswith("TF:"):
                        tfname = feat.replace("TF:", "", 1).strip().upper()
                        try:
                            conf_pen[tfname] = float(r.get("penalty", 0.0))
                        except Exception:
                            conf_pen[tfname] = 0.0
        except Exception:
            # keep best-effort; don't fail engine
            pass

    # Load and normalize PKN
    raw_edges = pd.read_csv(pkn_path, sep="\t" if pkn_path.suffix.lower() in {".tsv", ".txt"} else ",")
    edges = _normalize_pkn_edges(raw_edges)
    edges.to_csv(paths.pkn_edges, sep="\t", index=False)

    incoming = _build_incoming(edges)

    sub_edges_set: Set[Tuple[str, str, int]] = set()
    driver_score: Dict[str, float] = {}
    node_sign_votes: Dict[str, List[int]] = {}
    cards: List[Dict[str, object]] = []

    # thresholds (kept conservative)
    min_abs_corr_for_support = 0.20

    for tf in tf_keep:
        tf_act = float(tf_mean.loc[tf])
        tf_dir = 1 if tf_act > 0 else (-1 if tf_act < 0 else 0)

        best = _bfs_upstream(incoming, tf, max_steps=int(max_steps))
        route_support = int(len(best) > 1)

        nodes_in = {n for (n, _sg) in best.keys()}
        for node in list(nodes_in):
            for src, esign in incoming.get(node, []):
                if src in nodes_in and esign != 0:
                    sub_edges_set.add((src, node, int(esign)))

        for (node, sign_to_tf), depth in best.items():
            if node == tf or depth <= 0:
                continue

            implied = int(tf_dir * sign_to_tf) if tf_dir != 0 else 0
            if implied != 0:
                node_sign_votes.setdefault(node, []).append(implied)

            contrib = abs(tf_act) / float(depth)
            driver_score[node] = driver_score.get(node, 0.0) + contrib

        penalty = float(conf_pen.get(tf, 0.0))

        # NEW: measured footprint consistency check (not just file existence)
        fp_best_corr: Optional[float] = None
        fp_best_pathway: Optional[str] = None
        fp_support = False

        if footprints_df is not None and tf in tf_df.columns:
            fp_best_corr, fp_best_pathway = _footprint_consistency_for_tf(
                tf_series=tf_df[tf],
                footprints_df=footprints_df,
                min_samples=6,
            )
            if fp_best_corr is not None and abs(fp_best_corr) >= float(min_abs_corr_for_support):
                fp_support = True

        # Confidence gating: High/Medium requires *supporting* footprint consistency now
        if fp_support and route_support and penalty < 0.30 and abs(tf_act) >= 1.0:
            conf_label = "High"
        elif fp_support and route_support and penalty < float(confound_penalty_threshold):
            conf_label = "Medium"
        else:
            conf_label = "Low"

        notes = [
            "Directed route evidence is prior-knowledge-consistent support (PKN), not statistical causality proof.",
            "High/Medium confidence requires measured footprint consistency (not only file presence).",
        ]
        if footprints_load_error:
            notes.append(f"Footprints file present but could not be loaded: {footprints_load_error}")

        cards.append({
            "type": "TF_route",
            "tf": tf,
            "tf_mean_activity": tf_act,
            "tf_direction": "UP" if tf_dir > 0 else ("DOWN" if tf_dir < 0 else "ZERO"),
            "footprints_available": bool(footprints_available),
            "footprints_support": bool(fp_support),
            "footprints_best_abs_corr": float(fp_best_corr) if fp_best_corr is not None else None,
            "footprints_best_feature": fp_best_pathway,
            "route_support": bool(route_support),
            "confounding_penalty": penalty,
            "confidence": conf_label,
            "notes": notes,
        })

    if not sub_edges_set:
        msg = "Engine3 skipped: No causal subnetwork edges formed (check PKN coverage / symbol mapping)."
        if strict:
            raise DataError(msg)
        log.warning(msg)
        _atomic_write_text(paths.skipped_flag, msg + "\n")
        _atomic_write_text(paths.manifest, json.dumps({
            "engine": "causal_pathway_context",
            "version": "1.0.0",
            "status": "skipped",
            "reason": msg,
            "inputs": {"tf_activity": str(tf_path), "pkn_edges": str(pkn_path)},
            "outputs": {"skipped_flag": str(paths.skipped_flag), "pkn_edges": str(paths.pkn_edges)},
        }, indent=2))
        return paths

    sub_edges = pd.DataFrame(list(sub_edges_set), columns=["source", "target", "sign"])
    sub_edges.to_csv(paths.causal_edges, sep="\t", index=False)

    nodes = sorted(set(sub_edges["source"]) | set(sub_edges["target"]))
    degree = pd.concat([sub_edges["source"].value_counts(), sub_edges["target"].value_counts()], axis=1).fillna(0).sum(axis=1).to_dict()

    node_rows: List[Dict[str, object]] = []
    for n in nodes:
        votes = node_sign_votes.get(n, [])
        implied_sign = 0
        if votes:
            implied_sign = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)

        node_rows.append({
            "node": n,
            "driver_score": float(driver_score.get(n, 0.0)),
            "degree": float(degree.get(n, 0.0)),
            "implied_sign": int(implied_sign),
        })

    nodes_df = pd.DataFrame(node_rows).sort_values(["driver_score", "degree"], ascending=[False, False])
    nodes_df.to_csv(paths.causal_nodes, sep="\t", index=False)

    drivers = nodes_df[nodes_df["driver_score"] > 0].copy().reset_index(drop=True)
    drivers.to_csv(paths.drivers_ranked, sep="\t", index=False)

    paths.mechanism_cards_json.write_text(json.dumps({"disease": disease, "cards": cards}, indent=2), encoding="utf-8")
    pd.DataFrame(cards).to_csv(paths.mechanism_cards_tsv, sep="\t", index=False)

    manifest = {
        "engine": "causal_pathway_context",
        "version": "1.0.0",
        "status": "ok",
        "inputs": {
            "out_root": str(out_root),
            "disease": disease,
            "tf_activity": str(tf_path),
            "pathway_footprints": str(footprints_path) if footprints_path else None,
            "pkn_edges": str(pkn_path),
            "confounding_report": str(conf_path) if conf_path else None,
        },
        "params": {
            "max_steps": int(max_steps),
            "top_tfs": int(top_tfs),
            "confound_penalty_threshold": float(confound_penalty_threshold),
            "strict": bool(strict),
            "footprints_min_abs_corr_for_support": float(min_abs_corr_for_support),
        },
        "outputs": {
            "pkn_edges": str(paths.pkn_edges),
            "causal_edges": str(paths.causal_edges),
            "causal_nodes": str(paths.causal_nodes),
            "drivers_ranked": str(paths.drivers_ranked),
            "mechanism_cards_json": str(paths.mechanism_cards_json),
            "mechanism_cards_tsv": str(paths.mechanism_cards_tsv),
        },
        "notes": [
            "This is mechanistic plausibility contextualization (signed PKN + TF constraints), not statistical causality proof.",
            "High/Medium confidence requires measured footprint consistency (best abs corr vs footprint activity).",
        ],
    }
    _atomic_write_text(paths.manifest, json.dumps(manifest, indent=2))

    if paths.skipped_flag.exists():
        try:
            paths.skipped_flag.unlink()
        except Exception:
            pass

    return paths
