"""
RNABaseEditEngine — dCas13 RNA Base Editing (A-to-I via ADAR2, C-to-U via APOBEC)
=====================================================================================
Predicts RNA base editing outcomes using catalytically dead Cas13 (dCas13)
fused to deaminase domains. No DNA alteration — reversible, non-heritable edits.

Capabilities:
    - A-to-I editing (ADAR2dd fusion): Recodes codons, alters splicing
    - C-to-U editing (APOBEC1 fusion): Creates premature stops, recodes
    - Edit window prediction (optimal positions within spacer)
    - Bystander editing risk assessment
    - Codon change prediction (amino acid impact)
    - Splice site disruption/creation prediction
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("biragas_crispr.rna.base_edit")

CODON_TABLE = {
    'UUU':'F','UUC':'F','UUA':'L','UUG':'L','CUU':'L','CUC':'L','CUA':'L','CUG':'L',
    'AUU':'I','AUC':'I','AUA':'I','AUG':'M','GUU':'V','GUC':'V','GUA':'V','GUG':'V',
    'UCU':'S','UCC':'S','UCA':'S','UCG':'S','CCU':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACU':'T','ACC':'T','ACA':'T','ACG':'T','GCU':'A','GCC':'A','GCA':'A','GCG':'A',
    'UAU':'Y','UAC':'Y','UAA':'*','UAG':'*','CAU':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAU':'N','AAC':'N','AAA':'K','AAG':'K','GAU':'D','GAC':'D','GAA':'E','GAG':'E',
    'UGU':'C','UGC':'C','UGA':'*','UGG':'W','CGU':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGU':'S','AGC':'S','AGA':'R','AGG':'R','GGU':'G','GGC':'G','GGA':'G','GGG':'G',
}


@dataclass
class BaseEditSite:
    """A predicted RNA base editing site."""
    position: int = 0
    original_base: str = ""
    edited_base: str = ""
    edit_type: str = ""           # A-to-I or C-to-U
    efficiency: float = 0.0       # predicted editing rate
    bystander_risk: float = 0.0   # risk of nearby unwanted edits
    codon_original: str = ""
    codon_edited: str = ""
    aa_original: str = ""
    aa_edited: str = ""
    is_recoding: bool = False
    is_stop_codon: bool = False
    splice_impact: str = "none"   # none, donor_disrupted, acceptor_disrupted, created
    in_edit_window: bool = False

    def to_dict(self) -> Dict:
        return {
            'position': self.position, 'edit': f"{self.original_base}→{self.edited_base}",
            'type': self.edit_type, 'efficiency': round(self.efficiency, 2),
            'bystander_risk': round(self.bystander_risk, 2),
            'codon': f"{self.codon_original}→{self.codon_edited}",
            'amino_acid': f"{self.aa_original}→{self.aa_edited}",
            'recoding': self.is_recoding, 'stop_codon': self.is_stop_codon,
            'splice_impact': self.splice_impact,
        }


@dataclass
class BaseEditPrediction:
    """Complete prediction for a dCas13 base editing experiment."""
    gene: str = ""
    guide_sequence: str = ""
    edit_type: str = ""
    target_region: str = ""
    edit_sites: List[BaseEditSite] = field(default_factory=list)
    primary_edit: Optional[BaseEditSite] = None
    total_editable_bases: int = 0
    bystander_sites: int = 0
    overall_efficiency: float = 0.0
    specificity_score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'gene': self.gene, 'guide': self.guide_sequence,
            'edit_type': self.edit_type,
            'primary_edit': self.primary_edit.to_dict() if self.primary_edit else None,
            'total_editable': self.total_editable_bases,
            'bystander_sites': self.bystander_sites,
            'efficiency': round(self.overall_efficiency, 2),
            'specificity': round(self.specificity_score, 3),
            'all_sites': [s.to_dict() for s in self.edit_sites],
        }


class RNABaseEditEngine:
    """
    dCas13-ADAR2/APOBEC RNA base editing prediction engine.

    Edit types:
        A-to-I (read as G): dCas13-ADAR2dd → adenosine deamination
        C-to-U: dCas13-APOBEC1 → cytidine deamination

    Edit window: positions 20-30 from 5' end of spacer (for REPAIR/RESCUE systems)
    """

    # Optimal edit window within spacer (positions from 5' end)
    ADAR2_WINDOW = (18, 30)   # A-to-I optimal positions
    APOBEC_WINDOW = (15, 25)  # C-to-U optimal positions

    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}
        logger.info("RNABaseEditEngine initialized (A-to-I + C-to-U)")

    def predict_a_to_i(self, guide_seq: str, target_rna: str,
                       gene: str = "", cds_start: int = 0) -> BaseEditPrediction:
        """Predict A-to-I editing outcomes for a dCas13-ADAR2 guide."""
        return self._predict_edits(guide_seq, target_rna, gene, cds_start,
                                    edit_type="A-to-I", target_base='A',
                                    edited_base='I', window=self.ADAR2_WINDOW)

    def predict_c_to_u(self, guide_seq: str, target_rna: str,
                       gene: str = "", cds_start: int = 0) -> BaseEditPrediction:
        """Predict C-to-U editing outcomes for a dCas13-APOBEC1 guide."""
        return self._predict_edits(guide_seq, target_rna, gene, cds_start,
                                    edit_type="C-to-U", target_base='C',
                                    edited_base='U', window=self.APOBEC_WINDOW)

    def _predict_edits(self, guide_seq, target_rna, gene, cds_start,
                       edit_type, target_base, edited_base, window) -> BaseEditPrediction:
        guide = guide_seq.upper().replace('T', 'U')
        target = target_rna.upper().replace('T', 'U')

        # Find guide binding region on target
        guide_rc = ''.join({'A':'U','U':'A','G':'C','C':'G'}.get(c, c) for c in guide)
        bind_pos = target.find(guide_rc)
        if bind_pos < 0:
            bind_pos = 0

        sites = []
        primary = None

        for i, base in enumerate(guide_rc):
            abs_pos = bind_pos + i
            if abs_pos >= len(target):
                break

            if target[abs_pos] == target_base:
                in_window = window[0] <= i <= window[1]

                # Efficiency based on position within edit window
                if in_window:
                    center = (window[0] + window[1]) / 2
                    dist = abs(i - center) / ((window[1] - window[0]) / 2)
                    eff = max(10, 80 * (1.0 - dist * 0.7))
                else:
                    eff = max(2, 15 * (1.0 - abs(i - window[0]) / len(guide)))

                # Sequence context affects efficiency
                if i > 0 and i < len(guide_rc) - 1:
                    five_p = target[abs_pos - 1] if abs_pos > 0 else 'N'
                    three_p = target[abs_pos + 1] if abs_pos < len(target) - 1 else 'N'
                    # ADAR2 prefers UAG > AAG > CAG (5'-neighbor preference)
                    if edit_type == "A-to-I":
                        if five_p == 'U': eff *= 1.3
                        elif five_p == 'A': eff *= 1.0
                        elif five_p == 'C': eff *= 0.7
                        elif five_p == 'G': eff *= 0.5

                # Codon impact
                codon_orig = codon_edit = aa_orig = aa_edit = ""
                is_recoding = is_stop = False
                if cds_start <= abs_pos:
                    codon_pos = (abs_pos - cds_start) % 3
                    codon_start = abs_pos - codon_pos
                    if codon_start >= 0 and codon_start + 3 <= len(target):
                        codon_orig = target[codon_start:codon_start + 3]
                        codon_list = list(codon_orig)
                        codon_list[codon_pos] = 'G' if edit_type == "A-to-I" else edited_base
                        codon_edit = ''.join(codon_list)
                        aa_orig = CODON_TABLE.get(codon_orig, '?')
                        aa_edit = CODON_TABLE.get(codon_edit, '?')
                        is_recoding = aa_orig != aa_edit
                        is_stop = aa_edit == '*'

                # Splice site check
                splice = "none"
                local = target[max(0, abs_pos-2):abs_pos+3]
                if 'GU' in local and edit_type == "A-to-I":
                    splice = "donor_proximal"
                elif 'AG' in local:
                    splice = "acceptor_proximal"

                site = BaseEditSite(
                    position=abs_pos, original_base=target_base,
                    edited_base='G' if edit_type == "A-to-I" else edited_base,
                    edit_type=edit_type, efficiency=min(95, eff),
                    bystander_risk=0.0 if in_window else min(50, eff * 0.5),
                    codon_original=codon_orig, codon_edited=codon_edit,
                    aa_original=aa_orig, aa_edited=aa_edit,
                    is_recoding=is_recoding, is_stop_codon=is_stop,
                    splice_impact=splice, in_edit_window=in_window,
                )
                sites.append(site)

                if in_window and (primary is None or site.efficiency > primary.efficiency):
                    primary = site

        bystanders = sum(1 for s in sites if not s.in_edit_window and s.efficiency > 10)

        specificity = 1.0
        if sites:
            total_eff = sum(s.efficiency for s in sites)
            primary_eff = primary.efficiency if primary else 0
            specificity = primary_eff / max(total_eff, 1)

        return BaseEditPrediction(
            gene=gene, guide_sequence=guide_seq, edit_type=edit_type,
            target_region=target[bind_pos:bind_pos+len(guide)] if bind_pos >= 0 else "",
            edit_sites=sites, primary_edit=primary,
            total_editable_bases=len(sites), bystander_sites=bystanders,
            overall_efficiency=primary.efficiency if primary else 0,
            specificity_score=specificity,
        )

    def find_best_edit_sites(self, target_rna: str, edit_type: str = "A-to-I",
                              gene: str = "", n_sites: int = 5) -> List[BaseEditPrediction]:
        """Find optimal editing sites across the entire transcript."""
        target = target_rna.upper().replace('T', 'U')
        target_base = 'A' if edit_type == "A-to-I" else 'C'
        gl = 22  # CasRx guide length

        candidates = []
        for i in range(len(target) - gl):
            region = target[i:i + gl]
            # Count target bases in edit window
            window = self.ADAR2_WINDOW if edit_type == "A-to-I" else self.APOBEC_WINDOW
            w_start = min(window[0], gl - 1)
            w_end = min(window[1], gl)
            window_bases = sum(1 for j in range(w_start, w_end) if j < len(region) and region[j] == target_base)
            if window_bases > 0:
                guide = ''.join({'A':'U','U':'A','G':'C','C':'G'}.get(c, c) for c in region)
                candidates.append((guide, window_bases, i))

        candidates.sort(key=lambda x: -x[1])
        results = []
        for guide, _, pos in candidates[:n_sites * 2]:
            pred = self._predict_edits(guide, target, gene, 0, edit_type,
                                        target_base, 'G' if edit_type == "A-to-I" else 'U',
                                        self.ADAR2_WINDOW if edit_type == "A-to-I" else self.APOBEC_WINDOW)
            if pred.primary_edit:
                results.append(pred)

        results.sort(key=lambda p: -(p.overall_efficiency * p.specificity_score))
        return results[:n_sites]
