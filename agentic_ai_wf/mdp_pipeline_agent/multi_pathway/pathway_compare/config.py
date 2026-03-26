from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PCConfig:
    # plotting / limits
    top_k: int = 30
    dpi: int = 300
    max_upset_intersections: int = 100

    # significance & direction
    sig_cap: float = 300.0
    direction_mode: str = "both"                 # any|up|down|both
    borrow_any_into_directional: str = "up"      # none|up|down|both

    # behavior
    fail_on_missing_pathway: bool = False

    # name/ID matching toggles
    enable_id_matching: bool = True
    enable_name_fallback: bool = True

    # entity cleaning toggles
    clean_mouse_tags_in_tf: bool = True
    clean_mm_suffix_in_epigenetic: bool = True
