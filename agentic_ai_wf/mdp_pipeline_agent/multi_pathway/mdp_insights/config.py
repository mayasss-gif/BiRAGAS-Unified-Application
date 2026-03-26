from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class InsightsConfig:
    # core limits
    top_k: int = 20
    dpi: int = 300

    # shared views
    exclude_hmdb_in_shared: bool = False
    hub_cap: int = 30
    min_shared_gene_intersection: int = 2

    # significance & direction controls
    sig_cap: float = 300.0
    min_sig_for_label: float = 2.0
    direction_mode: str = "both"
    borrow_any_into_directional: str = "up"  # treat ANY as UP by default

    # Top-200 tables
    top_n: int = 200
    pis_w_prevalence: float = 0.35
    pis_w_evidence: float = 0.35
    pis_w_specificity: float = 0.15
    pis_w_support: float = 0.15

    tps_w_cross_pathway: float = 0.30
    tps_w_cross_disease: float = 0.35
    tps_w_evidence: float = 0.30
    tps_w_directionality: float = 0.05

    # modules & similarity
    max_modules_k: int = 30

    # species filters
    exclude_mouse_epigenetic_suffix: bool = True
    mouse_suffix_regex: str = r"(?i)(?:^|[\s_\-()])mm(?:\d{1,2})?$"

    # NEW — filter TFs that look mouse-specific anywhere in the name
    exclude_mouse_tf: bool = True
    tf_mouse_regex: str = r"(?i)\b(mouse|murine|mus\s*musculus)\b|\([\s]*mouse[\s]*\)|\[[\s]*mouse[\s]*\]"
