"""
ClinicalNarrativeGenerator — AI-Generated Clinical Summary
=============================================================
Generates human-readable clinical narratives from analysis results,
covering DNA knockout, RNA knockdown, combination therapy, causality
validation, safety assessment, and recommended next steps.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("biragas_crispr.reporting.narrative")


class ClinicalNarrativeGenerator:
    """Generate clinical narratives from BiRAGAS analysis results."""

    def generate(self, report_data: Dict, disease: str = "Disease") -> str:
        """Generate comprehensive clinical narrative."""
        sections = []

        # Opening
        sections.append(self._opening(report_data, disease))

        # Knockout findings
        ko = report_data.get('knockout', report_data.get('dna_stages', {}).get('knockout', {}))
        if ko:
            sections.append(self._knockout_section(ko, disease))

        # Combination findings
        combos = report_data.get('combinations', {})
        if combos:
            sections.append(self._combination_section(combos))

        # RNA findings
        rna = report_data.get('rna_stages', {})
        be = report_data.get('base_editing', {})
        if rna or be:
            sections.append(self._rna_section(rna, be))

        # Causality validation
        caus = report_data.get('causality', {})
        if caus:
            sections.append(self._causality_section(caus))

        # Safety
        sections.append(self._safety_section(report_data))

        # Recommendations
        sections.append(self._recommendations(report_data, disease))

        return "\n\n".join(s for s in sections if s)

    def _opening(self, data, disease) -> str:
        scale = data.get('scale', {})
        n_genes = scale.get('genes', 'N/A')
        total_configs = scale.get('total_configs', 'N/A')
        total_combos = scale.get('total_billions', scale.get('total_combos', 'N/A'))

        return (
            f"BIRAGAS CRISPR COMPLETE CLINICAL REPORT — {disease}\n"
            f"{'='*60}\n"
            f"This analysis applied the BiRAGAS CRISPR Complete v3.0 platform "
            f"(unified DNA + RNA) to {disease}, examining {n_genes} genes across "
            f"{total_configs} knockout/knockdown configurations and predicting "
            f"up to {total_combos} billion pairwise combinations using 12 synergy models."
        )

    def _knockout_section(self, ko, disease) -> str:
        top = ko.get('top_15', ko.get('top_5', []))
        if not top:
            return ""

        n_total = ko.get('total_predicted', len(top))
        n_configs = ko.get('total_configs', n_total * 11)
        top_gene = top[0].get('gene', 'N/A')
        top_score = top[0].get('ensemble', 0)

        genes_str = ', '.join(t.get('gene', '') for t in top[:5])

        return (
            f"KNOCKOUT/KNOCKDOWN PREDICTIONS\n"
            f"{'-'*40}\n"
            f"The 7-method ensemble (Topological + Bayesian + Monte Carlo + Pathway + "
            f"Feedback + ODE + Mutual Information) predicted knockout effects for "
            f"{n_total} genes ({n_configs:,} total configurations).\n\n"
            f"Top target: {top_gene} (ensemble score: {top_score:.4f}). "
            f"The top 5 targets are: {genes_str}. "
            f"These genes represent the most impactful knockout candidates for {disease} "
            f"based on causal network analysis."
        )

    def _combination_section(self, combos) -> str:
        parts = []
        for cls_key, cls_name in [('dna_x_dna', 'DNA×DNA'), ('rna_x_rna', 'RNA×RNA'), ('dna_x_rna', 'DNA×RNA')]:
            cls_data = combos.get(cls_key, {})
            n = cls_data.get('count', 0)
            syn = cls_data.get('synergistic', 0)
            if n > 0:
                top = cls_data.get('top_5', [])
                top_str = ''
                if top:
                    t = top[0]
                    top_str = f" Top pair: {t['genes'][0]}+{t['genes'][1]} (synergy={t.get('synergy', 0):.4f})."
                parts.append(f"  {cls_name}: {n} pairs analyzed, {syn} synergistic.{top_str}")

        triple = combos.get('triple_kras_pi3k_mir21', {})
        triple_str = ""
        if triple:
            triple_str = (f"\n\n3-WAY COMBINATION: {'+'.join(triple.get('genes', []))} "
                          f"(synergy={triple.get('synergy', 0):.4f}, class={triple.get('combination_class', '')}). "
                          f"This cross-modal triple targets multiple compensation pathways simultaneously.")

        return (
            f"COMBINATION SYNERGY ANALYSIS (12-Model Cross-Modal)\n"
            f"{'-'*40}\n"
            f"12 synergy models (6 classical + 6 cross-modal) predicted:\n"
            + '\n'.join(parts) + triple_str
        )

    def _rna_section(self, rna, be) -> str:
        parts = ["RNA-LEVEL ANALYSIS", "-" * 40]

        if rna:
            cas13 = rna.get('cas13_guides', {})
            if cas13:
                parts.append(f"Cas13 RNA guide design: {cas13.get('genes', 0)} genes, "
                             f"nucleases: {', '.join(cas13.get('nucleases', []))}")
            ci = rna.get('crispri_crispra', {})
            if ci:
                parts.append(f"CRISPRi/CRISPRa: {ci.get('designs', 0)} designs "
                             f"({ci.get('crispri', 0)} CRISPRi, {ci.get('crispra', 0)} CRISPRa)")
            nc = rna.get('noncoding', {})
            if nc:
                parts.append(f"Non-coding RNA: {nc.get('analyzed', 0)} ncRNAs analyzed "
                             f"({nc.get('lncrna', 0)} lncRNA, {nc.get('mirna', 0)} miRNA)")

        if be:
            parts.append(f"RNA Base Editing target: {be.get('target', 'N/A')}")
            a2i = be.get('a_to_i_sites', [])
            if a2i:
                parts.append(f"  A-to-I edit sites found: {len(a2i)}")

        return '\n'.join(parts)

    def _causality_section(self, caus) -> str:
        n_run = caus.get('modules_run', 0)
        n_fail = caus.get('modules_failed', 0)

        p7 = caus.get('phases', {}).get('phase7', {}).get('details', {}).get('report', {})
        narrative = p7.get('narrative', '')

        return (
            f"CAUSALITY FRAMEWORK VALIDATION (28 Modules × 7 Phases)\n"
            f"{'-'*40}\n"
            f"All {n_run} modules executed: {n_run - n_fail} passed, {n_fail} failed.\n\n"
            f"{narrative}"
        )

    def _safety_section(self, data) -> str:
        caus = data.get('causality', {})
        p5 = caus.get('phases', {}).get('phase5', {}).get('details', {}).get('safety', {})
        safe = p5.get('safe', 0)
        risky = p5.get('risky', 0)

        return (
            f"SAFETY ASSESSMENT\n"
            f"{'-'*40}\n"
            f"Safety classification: {safe} genes classified as safe targets, "
            f"{risky} flagged with safety concerns (Core Essential or high resistance risk). "
            f"Recommend avoiding Core Essential genes for therapeutic targeting."
        )

    def _recommendations(self, data, disease) -> str:
        gaps = data.get('causality', {}).get('phases', {}).get('phase7', {}).get('details', {}).get('gap_analysis', {})
        gap_text = ""
        if gaps and gaps.get('top_gaps'):
            top_gaps = gaps['top_gaps'][:5]
            gap_lines = [f"  - {g['gene']} ({g['priority']}): missing {', '.join(g['missing'][:2])}" for g in top_gaps]
            gap_text = "\n\nRecommended validation experiments:\n" + '\n'.join(gap_lines)

        return (
            f"RECOMMENDATIONS & NEXT STEPS\n"
            f"{'-'*40}\n"
            f"1. Validate top knockout targets with in vitro CRISPR screens\n"
            f"2. Test cross-modal DNA+RNA combinations in cell line models\n"
            f"3. Confirm RNA knockdown targets with Cas13d in relevant {disease} cell lines\n"
            f"4. Assess combination synergy using dose-response matrix experiments\n"
            f"5. Perform Perturb-seq to validate predicted transcriptomic effects"
            + gap_text
        )
