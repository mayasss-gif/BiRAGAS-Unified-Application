#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evidence ranking + concise clinical bullets for MDP Plan2 reports (NO BioLinkBERT).

Goal:
- Normalize noisy driver strings from overlap JSON (TF/epi/metabolite labels).
- Rank drivers for a given pathway using evidence_weight (overlap gene support).
- Generate clean clinical bullets (2–3) for:
    "Same pathway, different mechanism (top 3)"

CONSTRAINTS:
- MUST NOT invent mechanistic claims.
- Bullets MUST only reference entities present in overlap JSON for that pathway.
- Bullets MUST be evidence-first and conservative (association only).
- Output must be plain text (NO markdown ** **).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


# -----------------------------
# Non-human filtering (entities)
# -----------------------------
NON_HUMAN_PAT = re.compile(r"\b(mouse|mice|rat|mm9|mm10|murine|c2c12)\b", flags=re.IGNORECASE)


# -----------------------------
# Normalization
# -----------------------------
_HISTONE_MARK_PAT = re.compile(r"^(H[234]K\d+(AC|ME\d?))$", flags=re.IGNORECASE)
_STUDY_ID_PAT = re.compile(r"\b(\d{7,8})\b")
_SEP_PAT = re.compile(r"[\|\(\)\[\]\{\}:]+")
_NONWORD_PAT = re.compile(r"[^A-Za-z0-9_\-]+")
_MULTI_WS = re.compile(r"\s+")


def _clean(s: str) -> str:
    s = str(s or "").strip()
    s = _SEP_PAT.sub(" ", s)
    s = _STUDY_ID_PAT.sub(" ", s)
    s = _MULTI_WS.sub(" ", s).strip()
    return s


def normalize_tf_label(raw: str) -> str:
    s = _clean(raw)
    if not s:
        return ""
    toks = s.split()
    head = toks[0] if toks else ""
    if _HISTONE_MARK_PAT.match(head):
        return head.upper()
    head = _NONWORD_PAT.sub("", head).upper().strip()
    return head


def normalize_epigenetic_label(raw: str) -> str:
    s = _clean(raw)
    if not s:
        return ""
    toks = s.split()
    for t in toks:
        t2 = _NONWORD_PAT.sub("", t)
        if _HISTONE_MARK_PAT.match(t2):
            return t2.upper()
    head = _NONWORD_PAT.sub("", toks[0]).upper().strip()
    return head


def normalize_metabolite_label(raw: str) -> str:
    s = _clean(raw)
    if not s:
        return ""
    return s.strip()


def normalize_entity(raw: str, entity_type: str) -> str:
    et = (entity_type or "").strip().lower()
    if et == "tf":
        return normalize_tf_label(raw)
    if et == "epigenetic":
        return normalize_epigenetic_label(raw)
    if et == "metabolite":
        return normalize_metabolite_label(raw)
    return _clean(raw).strip()


# -----------------------------
# Micro-glossary (safe, conservative)
# -----------------------------
_EPI_MARK_GLOSS = {
    "H3K27AC": "active regulatory chromatin signal (often linked to higher transcriptional activity)",
    "H3K4ME3": "promoter/TSS-linked chromatin signal",
    "H3K4ME1": "enhancer-linked chromatin signal (context-dependent)",
    "H3K36ME3": "gene-body/elongation-linked chromatin signal",
    "H3K27ME3": "repressive polycomb-linked chromatin signal",
}

_TF_GLOSS = {
    "SPI1": "PU.1 lineage/immune regulator (association only)",
    "STAT3": "JAK/STAT inflammatory transcription effector (association only)",
    "MYC": "growth/metabolic transcription amplifier (association only)",
    "FOS": "immediate-early response TF (association only)",
    "YY1": "broad transcription/chromatin regulator (association only)",
    "VDR": "nuclear receptor axis (association only)",
    "TAF1": "general transcription factor complex component (association only)",
    "GABPA": "ETS-family transcription regulator (association only)",
}


def _norm_key(x: str) -> str:
    return str(x or "").upper().replace("-", "").replace("_", "").strip()


def entity_micro_gloss(entity: str, entity_type: str) -> str:
    e = str(entity or "").strip()
    if not e:
        return ""
    et = (entity_type or "").lower().strip()

    if et == "epigenetic":
        k = _norm_key(e)
        for mk, desc in _EPI_MARK_GLOSS.items():
            if _norm_key(mk) == k:
                return desc
        if "H3" in e.upper():
            return "histone modification mark; indicates chromatin-state differences (association only)"
        return "epigenetic/chromatin feature; indicates regulatory-state differences (association only)"

    if et == "tf":
        k = e.upper().strip()
        if k in _TF_GLOSS:
            return _TF_GLOSS[k]
        return "transcription regulator signal (association only)"

    if et == "metabolite":
        return "metabolite signal; may reflect metabolic-state differences (association only)"

    return ""


# -----------------------------
# Evidence aggregation
# -----------------------------
def _extract_overlap_score(row: Any) -> float:
    if not isinstance(row, dict):
        return 1.0

    key_candidates = [
        "overlap_genes", "overlapGenes", "overlap_gene_set",
        "overlap_count", "n_overlap", "nOverlaps", "overlap",
        "shared_genes", "n_shared", "intersection_size",
        "gene_overlap", "overlap_n", "k",
    ]
    for k in key_candidates:
        if k in row:
            v = row.get(k)
            if isinstance(v, (list, tuple, set)):
                return float(len(v))
            if isinstance(v, str):
                toks = [t for t in re.split(r"[,\s;]+", v.strip()) if t]
                if len(toks) >= 2:
                    return float(len(toks))
                try:
                    return float(v)
                except Exception:
                    return 1.0
            try:
                return float(v)
            except Exception:
                return 1.0

    if "genes" in row:
        v = row.get("genes")
        if isinstance(v, (list, tuple, set)):
            return float(len(v))
        if isinstance(v, str):
            toks = [t for t in re.split(r"[,\s;]+", v.strip()) if t]
            if toks:
                return float(len(toks))

    return 1.0


def _get_pathway_bucket(overlap_obj: Dict[str, Any], pathway_name: str) -> Dict[str, Any]:
    if not overlap_obj or not pathway_name:
        return {}
    if pathway_name in overlap_obj:
        v = overlap_obj.get(pathway_name) or {}
        return v if isinstance(v, dict) else {}
    low = str(pathway_name).strip().lower()
    for k in overlap_obj.keys():
        if str(k).strip().lower() == low:
            v = overlap_obj.get(k) or {}
            return v if isinstance(v, dict) else {}
    return {}


def aggregate_normalized_entities_for_pathway(
    overlap_obj: Dict[str, Any],
    pathway: str,
    entity_type: str,
    direction_mode: str = "ANY",
    human_only: bool = True,
    nonhuman_pat: Optional[re.Pattern] = None,
) -> Dict[str, float]:
    """
    normalized_entity -> aggregated evidence score for this pathway.
    """
    if not overlap_obj or not pathway:
        return {}

    d = _get_pathway_bucket(overlap_obj, pathway)
    if not d:
        return {}

    def ok_entity(name: str) -> bool:
        if not name:
            return False
        if not human_only:
            return True
        if nonhuman_pat is None:
            return True
        return nonhuman_pat.search(name) is None

    def pull(dir_key: str) -> List[Dict[str, Any]]:
        dd = (d.get(dir_key, {}) or {})
        arr = dd.get(entity_type, [])
        return arr if isinstance(arr, list) else []

    if direction_mode == "UP":
        rows = pull("UP")
    elif direction_mode == "DOWN":
        rows = pull("DOWN")
    else:
        rows = pull("UP") + pull("DOWN")

    raw_best: Dict[str, float] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        raw_ent = str(r.get("entity", "")).strip()
        if not ok_entity(raw_ent):
            continue
        sc = _extract_overlap_score(r)
        raw_best[raw_ent] = max(raw_best.get(raw_ent, 0.0), float(sc))

    norm_scores: Dict[str, float] = {}
    for raw_ent, sc in raw_best.items():
        norm = normalize_entity(raw_ent, entity_type=entity_type)
        if not norm:
            continue
        norm_scores[norm] = norm_scores.get(norm, 0.0) + float(sc)

    return norm_scores


def rank_by_evidence(entity_scores: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
    if not entity_scores:
        return []
    out = [(k, float(v)) for k, v in entity_scores.items()]
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k]


# -----------------------------
# Bullet generator (CLINICAL, 3 bullets, plain text)
# -----------------------------
@dataclass
class ContrastNarrative:
    bullets: List[str]
    debug: Dict[str, Any]


def _fmt_names(items: List[Tuple[str, float]], n: int) -> List[str]:
    return [x[0] for x in items[:n]]


def _first_or_none(items: List[Tuple[str, float]]) -> Optional[str]:
    return items[0][0] if items else None


def build_contrast_bullets(
    pathway: str,
    disease_a: str,
    disease_b: str,
    json_a: Dict[str, Any],
    json_b: Dict[str, Any],
    nonhuman_pat: Optional[re.Pattern] = NON_HUMAN_PAT,
    max_entities_each: int = 2,
) -> ContrastNarrative:
    """
    Produce exactly 3 plain-text bullets:
      1) Shared scaffold (top shared TF/epi/met)
      2) Clinical contrast: A-specific vs B-specific (handles missing)
      3) Actionable validation step

    Evidence-bound, association-only.
    """

    a_tf = aggregate_normalized_entities_for_pathway(json_a, pathway, "tf", "ANY", True, nonhuman_pat)
    b_tf = aggregate_normalized_entities_for_pathway(json_b, pathway, "tf", "ANY", True, nonhuman_pat)
    a_epi = aggregate_normalized_entities_for_pathway(json_a, pathway, "epigenetic", "ANY", True, nonhuman_pat)
    b_epi = aggregate_normalized_entities_for_pathway(json_b, pathway, "epigenetic", "ANY", True, nonhuman_pat)
    a_met = aggregate_normalized_entities_for_pathway(json_a, pathway, "metabolite", "ANY", True, nonhuman_pat)
    b_met = aggregate_normalized_entities_for_pathway(json_b, pathway, "metabolite", "ANY", True, nonhuman_pat)

    sa_tf, sb_tf = set(a_tf.keys()), set(b_tf.keys())
    sa_epi, sb_epi = set(a_epi.keys()), set(b_epi.keys())
    sa_met, sb_met = set(a_met.keys()), set(b_met.keys())

    sh_tf = sa_tf & sb_tf
    sh_epi = sa_epi & sb_epi
    sh_met = sa_met & sb_met

    only_a_tf = sa_tf - sb_tf
    only_b_tf = sb_tf - sa_tf
    only_a_epi = sa_epi - sb_epi
    only_b_epi = sb_epi - sa_epi
    only_a_met = sa_met - sb_met
    only_b_met = sb_met - sa_met

    sh_tf_r = rank_by_evidence({k: a_tf.get(k, 0.0) + b_tf.get(k, 0.0) for k in sh_tf}, top_k=3)
    sh_epi_r = rank_by_evidence({k: a_epi.get(k, 0.0) + b_epi.get(k, 0.0) for k in sh_epi}, top_k=2)
    sh_met_r = rank_by_evidence({k: a_met.get(k, 0.0) + b_met.get(k, 0.0) for k in sh_met}, top_k=2)

    a_leads = (
        rank_by_evidence({k: a_tf.get(k, 0.0) for k in only_a_tf}, top_k=3)
        or rank_by_evidence({k: a_epi.get(k, 0.0) for k in only_a_epi}, top_k=3)
        or rank_by_evidence({k: a_met.get(k, 0.0) for k in only_a_met}, top_k=3)
    )
    b_leads = (
        rank_by_evidence({k: b_tf.get(k, 0.0) for k in only_b_tf}, top_k=3)
        or rank_by_evidence({k: b_epi.get(k, 0.0) for k in only_b_epi}, top_k=3)
        or rank_by_evidence({k: b_met.get(k, 0.0) for k in only_b_met}, top_k=3)
    )

    shared_parts = []
    if sh_tf_r:
        shared_parts.append("shared TFs: " + ", ".join(_fmt_names(sh_tf_r, 3)))
    if sh_epi_r:
        shared_parts.append("shared chromatin marks: " + ", ".join(_fmt_names(sh_epi_r, 2)))
    if sh_met_r:
        shared_parts.append("shared metabolites: " + ", ".join(_fmt_names(sh_met_r, 2)))

    if not shared_parts:
        shared_parts.append("no shared drivers captured in overlap evidence at current thresholds")

    a_top = _fmt_names(a_leads, max_entities_each)
    b_top = _fmt_names(b_leads, max_entities_each)

    if a_top:
        a_desc = f"{disease_a} shows disease-leaning drivers such as {', '.join(a_top)}"
        a_gl = entity_micro_gloss(a_top[0], "tf") if a_top else ""
        if a_gl:
            a_desc += f" ({a_gl})"
    else:
        a_desc = f"{disease_a} has no clear disease-unique drivers in the overlap set for this pathway"

    if b_top:
        b_desc = f"{disease_b} shows disease-leaning drivers such as {', '.join(b_top)}"
        b_gl = entity_micro_gloss(b_top[0], "tf") if b_top else ""
        if b_gl:
            b_desc += f" ({b_gl})"
    else:
        b_desc = f"{disease_b} has no clear disease-unique drivers in the overlap set for this pathway"

    bullets = [
        f"{pathway}: overlap evidence supports shared pathway engagement across {disease_a} and {disease_b} ({'; '.join(shared_parts)}). Association only.",
        f"Clinical contrast under the same pathway label: {a_desc}; whereas {b_desc}. This suggests different upstream control layers mapping to the same pathway annotation (association only).",
        f"Actionable next step: reproduce these shared vs disease-leaning driver sets in an independent cohort, and verify whether leading-edge/overlap genes (if available upstream) align with the listed TF/chromatin/metabolite signals.",
    ]

    debug = {
        "pathway": pathway,
        "shared_counts": {"tf": len(sh_tf), "epigenetic": len(sh_epi), "metabolite": len(sh_met)},
        "a_unique_counts": {"tf": len(only_a_tf), "epigenetic": len(only_a_epi), "metabolite": len(only_a_met)},
        "b_unique_counts": {"tf": len(only_b_tf), "epigenetic": len(only_b_epi), "metabolite": len(only_b_met)},
        "top_shared": {"tf": _fmt_names(sh_tf_r, 3), "epigenetic": _fmt_names(sh_epi_r, 2), "metabolite": _fmt_names(sh_met_r, 2)},
        "top_a_unique": a_top,
        "top_b_unique": b_top,
    }

    return ContrastNarrative(bullets=bullets, debug=debug)
