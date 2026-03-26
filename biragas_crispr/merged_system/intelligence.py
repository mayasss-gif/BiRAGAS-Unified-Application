"""
intelligence.py — consolidated AI/LLM layer for the molecular causal discovery pipeline.

Sections:
    1. ClaudeClient        — Anthropic API wrapper with local fallback engines
    2. ClarificationEngine — interactive pre-run intake interview
    3. IntentClassifier    — classifies query into I_01..I_07
    4. LiteraturePipeline  — 8-stage literature evidence pipeline
    5. ResultSynthesiser   — synthesises the final analysis result
    6. StepNarrator        — generates 3-sentence narrations for pipeline steps

All imports from agent.py are guarded with TYPE_CHECKING to prevent circular imports.
"""
from __future__ import annotations

import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from agent import (
        ParsedIntent, LitBrief, LitClaim, LiteraturePaper,
        FileAuditResult, FinalResult, ClarificationResult,
    )

log = logging.getLogger("causal_platform")

# ── Constants (mirrored from agent to avoid circular import) ──────────────────
ANTHROPIC_MODEL      = "claude-sonnet-4-20250514"
MAX_TOKENS_INTENT    = 800
MAX_TOKENS_LIT       = 1000
MAX_TOKENS_NARRATE   = 400
MAX_TOKENS_RESULT    = 1200
MAX_TOKENS_CLAIMS    = 1000
MAX_TOKENS_BRIEF     = 1000

PUBMED_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC_BASE    = "https://www.ebi.ac.uk/europepmc/webservices/rest"
S2_BASE      = "https://api.semanticscholar.org/graph/v1"

LIT_MAX_PAPERS = 30
LIT_TOP_K      = 15
LIT_TIMEOUT    = 12.0

_EMAIL = "platform@causal.bio"

try:
    import anthropic as _anthropic_sdk
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


# =============================================================================
# SECTION 1 — ClaudeClient
# =============================================================================

class _LocalIntentEngine:
    """Rule-based intent classifier used when no API key is available."""

    _INTENTS = [
        ("I_01", "Causal Drivers Discovery",
         ["M12", "M13", "M14", "M15", "M_DC"],
         [("causal driver", 5), ("what causes", 4), ("causal mechanism", 4),
          ("upstream regulator", 4), ("find cause", 3), ("identify cause", 3),
          ("causal gene", 3), ("drives", 2), ("discover", 1)]),

        ("I_02", "Directed Causality X->Y",
         ["M12", "M_DC"],
         [("does .+ cause", 6), ("causally affect", 6), ("x.*cause.*y", 5),
          ("is .+ upstream", 5), ("causal effect of", 4), ("does .+ drive", 4),
          ("causal link between", 3), ("affect", 1)]),

        ("I_03", "Intervention / Actionability",
         ["M_IS", "M_PI"],
         [("drug target", 5), ("which target", 5), ("inhibit", 4), ("therapeutic", 4),
          ("intervention", 4), ("actionable", 4), ("treat", 3), ("drug", 3),
          ("clinical", 2), ("suppress", 2)]),

        ("I_04", "Comparative Causality",
         ["M12", "M12b", "M13", "M14", "M15", "M_DC", "DELTA"],
         [("compare", 5), ("difference between", 5), ("responder.*non.responder", 5),
          ("group a.*group b", 5), ("early.*late", 4), ("stratif", 4),
          ("versus", 3), ("across cohort", 3), ("between groups", 3)]),

        ("I_05", "Counterfactual / What-If",
         ["M_IS", "M_DC"],
         [("what if", 6), ("simulate", 6), ("counterfactual", 6), ("what happen", 5),
          ("predict.*effect", 5), ("if we.*inhibit", 5), ("knock.*down.*by", 5),
          ("what would", 4), ("scenario", 3)]),

        ("I_06", "Evidence Inspection / Explain",
         ["M15", "M14"],
         [("why is .+ ranked", 6), ("explain.*evidence", 6), ("what evidence", 5),
          ("evidence for", 5), ("justify", 4), ("evidence support", 4),
          ("which evidence", 4), ("evidence behind", 3), ("explain", 2)]),

        ("I_07", "Standard Association Analysis",
         [],
         [("differentially expressed", 5), ("deg", 4), ("pathway enrichment", 5),
          ("association", 4), ("correlation", 4), ("gsea", 4), ("go enrichment", 4),
          ("kegg", 3), ("top gene", 2)]),
    ]

    _DOMAIN_KEYWORDS = {
        "oncology":     ["cancer", "tumor", "oncogen", "carcinoma", "metastasis", "KRAS", "TP53"],
        "immunology":   ["immune", "autoimmune", "inflammat", "lupus", "SLE", "T cell", "B cell",
                         "cytokine", "interferon", "JAK", "STAT"],
        "neuroscience": ["neuron", "brain", "neurodegenera", "alzheimer", "parkinson",
                         "dopamine", "synaptic", "cortex"],
        "metabolism":   ["metabol", "insulin", "diabetes", "glucose", "lipid", "obesity",
                         "adipose", "AMPK", "mTOR"],
        "aging":        ["aging", "ageing", "senescen", "longevity", "telomer", "FOXO"],
        "cardiology":   ["cardiac", "heart", "cardiomyopathy", "arrhythmia", "myocardial"],
        "development":  ["development", "differentiation", "stem cell", "embryo", "pluripotent"],
        "microbiome":   ["microbiome", "microbiota", "gut bacteria", "16S", "metagenom"],
        "pharmacology": ["drug", "compound", "inhibitor", "agonist", "pharmacol"],
    }

    def classify(self, query: str) -> dict:
        q = query.lower()
        scores: dict[str, float] = {}
        for iid, iname, chain, kws in self._INTENTS:
            score = 0.0
            for pattern, weight in kws:
                if re.search(pattern, q):
                    score += weight
            scores[iid] = score

        best_id = max(scores, key=lambda k: scores[k])
        best_score = scores[best_id]
        max_possible = 30.0
        confidence = 0.65 + min(best_score / max_possible, 1.0) * 0.33

        defn = next(d for d in self._INTENTS if d[0] == best_id)
        _, iname, chain, _ = defn

        domain, phenotype = self._extract_context(query)
        gene_x, gene_y, intervention, groups, magnitude = self._extract_entities(query)

        needs_clarif = False
        clarif_q = None
        if best_id == "I_02" and not gene_x:
            needs_clarif = True
            clarif_q = "Which gene/protein is X (the potential cause), and what is Y (the outcome)?"
        elif best_id == "I_04" and not groups:
            needs_clarif = True
            clarif_q = "Which two groups should be compared? (e.g., responders vs non-responders)"
        elif best_id == "I_05" and not intervention:
            needs_clarif = True
            clarif_q = "Which gene/drug should be simulated, and to what level? (e.g., 80% inhibition of KRAS)"

        requires = {
            "expression":        best_id in ("I_01", "I_02", "I_04", "I_07"),
            "phenotype_labels":  best_id in ("I_01", "I_02", "I_04", "I_07"),
            "gwas_eqtl":         best_id in ("I_01", "I_02", "I_03"),
            "perturbation":      best_id in ("I_01", "I_03"),
            "temporal":          best_id in ("I_01", "I_02", "I_04"),
            "prior_network":     best_id in ("I_01", "I_02"),
            "intervention_data": best_id in ("I_03", "I_05"),
        }

        fallbacks = {
            "I_01": "Association-only ranking with explicit disclaimer",
            "I_02": "Graph-only hypothesis with uncertainty flags",
            "I_03": "Observed-only ranking with context caveats",
            "I_04": "Single-run results with stratification note",
            "I_05": "Observed-only results labelled as simulated",
            "I_06": "Partial evidence card from available artifacts",
            "I_07": "DEGs and pathways only",
        }

        parallel = []
        if best_id == "I_04":
            parallel = ["DAGBuilder group A and B run in parallel after shared Bio Prep"]
        elif best_id == "I_01":
            parallel = ["T_05 (temporal) + T_08 (GWAS/MR) run in parallel during Bio/Intervention Prep"]

        return {
            "intent_id":              best_id,
            "intent_name":            iname,
            "confidence":             round(confidence, 3),
            "needs_clarification":    needs_clarif,
            "clarifying_question":    clarif_q,
            "context": {
                "domain":              domain,
                "phenotype":           phenotype,
                "tissue_or_cell_type": self._extract_tissue(query),
                "organism":            self._extract_organism(query),
                "outcome_variable":    None,
                "cohort_description":  None,
            },
            "entities": {
                "gene_x":       gene_x,
                "gene_y":       gene_y,
                "intervention": intervention,
                "magnitude":    magnitude,
                "groups":       groups,
                "comparison":   groups,
                "pathway":      self._extract_pathway(query),
            },
            "requires":               requires,
            "needs_literature_first": best_id in ("I_01", "I_02", "I_03"),
            "requires_existing_dag":  best_id in ("I_03", "I_05", "I_06"),
            "routing_summary":        self._routing_summary(best_id, iname, domain, phenotype),
            "module_chain":           chain,
            "parallel_blocks":        parallel,
            "fallback":               fallbacks.get(best_id, "Association-only ranking"),
        }

    def _extract_context(self, query: str):
        q = query.lower()
        domain, phenotype = "molecular biology", ""
        for dom, kws in self._DOMAIN_KEYWORDS.items():
            if any(kw.lower() in q for kw in kws):
                domain = dom
                break
        m = re.search(
            r"\b(?:of|in|for|drive|cause)\s+([A-Za-z][A-Za-z0-9 _\-]{2,40}?)(?:\s+(?:in|using|with|from|cohort|patient|dataset|study|analysis)|$)",
            query, re.I,
        )
        if m:
            phenotype = m.group(1).strip()
        return domain, phenotype

    def _extract_tissue(self, query: str):
        tissues = ["blood", "pbmc", "kidney", "liver", "lung", "brain", "heart",
                   "muscle", "adipose", "skin", "colon", "breast", "pancreas",
                   "cell line", "k562", "jurkat", "hela", "single.cell"]
        for t in tissues:
            if re.search(t, query, re.I):
                return t.replace(".", " ")
        return None

    def _extract_organism(self, query: str) -> str:
        if re.search(r"\bmouse\b|\bmurine\b|\bMus musculus\b", query, re.I):
            return "mouse"
        if re.search(r"\bzebrafish\b|\bdanio\b", query, re.I):
            return "zebrafish"
        if re.search(r"\byeast\b|\bS\. cerevisiae\b", query, re.I):
            return "yeast"
        return "human"

    def _extract_entities(self, query: str):
        genes = re.findall(r'\b([A-Z][A-Z0-9]{1,7}(?:\d+)?)\b', query)
        gene_x = genes[0] if genes else None
        gene_y = genes[1] if len(genes) > 1 else None
        m_int = re.search(r'\b(inhibit(?:ion|ing)?|knock(?:down|out)|overexpress(?:ion)?|activat(?:e|ion))\b', query, re.I)
        intervention = m_int.group(1) if m_int else gene_x
        m_mag = re.search(r'(\d+)\s*%', query)
        magnitude = f"{m_mag.group(1)}%" if m_mag else None
        m_grp = re.search(r'(responders?|non.responders?|control|treated|early|late|group\s*[AB12])', query, re.I)
        groups = m_grp.group(1) if m_grp else None
        return gene_x, gene_y, intervention, groups, magnitude

    def _extract_pathway(self, query: str):
        known = ["MAPK", "PI3K", "AKT", "mTOR", "Wnt", "Notch", "Hedgehog", "JAK.STAT",
                 "NF.kB", "TGF.beta", "p53", "RAS", "VEGF", "HIF", "Hippo"]
        for pw in known:
            if re.search(pw, query, re.I):
                return pw
        return None

    def _routing_summary(self, iid: str, iname: str, domain: str, phenotype: str) -> str:
        pheno = phenotype or "the phenotype of interest"
        dom   = domain or "molecular biology"
        summaries = {
            "I_01": f"Full causal driver discovery in {dom} — {pheno}: all 4 phases run.",
            "I_02": f"Targeted X->Y causal test in {dom} — {pheno}: bio-prep then do-calculus.",
            "I_03": f"Intervention/drug target prioritisation for {pheno}: reads existing DAG.",
            "I_04": f"Comparative causal analysis for {pheno}: dual DAG build + DELTA.",
            "I_05": f"Counterfactual simulation for {pheno}: reads existing DAG.",
            "I_06": f"Evidence inspection for {pheno}: read-only, no new compute.",
            "I_07": f"Association analysis for {pheno}: DEG + pathway only.",
        }
        return summaries.get(iid, f"{iname}: {pheno}")


class _LocalClarificationEngine:
    """Rule-based clarification when no API key is available."""

    _ORGANISMS = ["human", "mouse", "rat", "zebrafish", "drosophila", "yeast",
                  "arabidopsis", "c. elegans", "pig", "non-human primate"]
    _TISSUES   = ["liver", "lung", "brain", "cortex", "heart", "kidney", "colon",
                  "breast", "skin", "blood", "pbmc", "neuron", "hepatocyte",
                  "fibroblast", "macrophage", "t cell", "b cell", "nk cell",
                  "stem cell", "organoid", "tumor", "stroma", "endothel"]

    def generate(self, query: str, file_types: list, intent_name: str) -> dict:
        ql = query.lower()
        questions: list[str] = []
        blocking:  list[int] = []

        if not any(o in ql for o in self._ORGANISMS):
            questions.append("1. What organism is your data from? (e.g. human, mouse, rat)")
            blocking.append(len(questions))

        if not any(t in ql for t in self._TISSUES):
            questions.append(
                f"{len(questions)+1}. What tissue or cell type is your data from? "
                "(e.g. colon epithelium, PBMC, primary hepatocytes)"
            )

        has_meta = any(t in file_types for t in ("metadata", "phenotype"))
        has_expr = "expression" in file_types
        if has_expr and not has_meta:
            questions.append(
                f"{len(questions)+1}. Do you have a sample metadata file with "
                "case/control or treatment labels? (needed for DESeq2 and pathway enrichment)"
            )
            blocking.append(len(questions))

        if not file_types:
            questions.append(
                f"{len(questions)+1}. Which data files do you have available to upload? "
                "(a) Expression/counts matrix  (b) Sample metadata with phenotype labels  "
                "(c) GWAS / MR results  (d) CRISPR screen data  "
                "(e) eQTL data  (f) Temporal / longitudinal data"
            )

        intent_l = intent_name.lower()
        if "directed" in intent_l or "i_02" in intent_l:
            has_arrow = any(tok in ql for tok in ["->", " to ", " drive ", " affect "])
            if not has_arrow:
                questions.append(
                    f"{len(questions)+1}. Which specific gene or protein X do you want to test "
                    "as the causal driver, and what is the outcome Y?"
                )
                blocking.append(len(questions))

        if "counterfactual" in intent_l or "simulate" in ql or "what if" in ql:
            if not re.search(r"\d+\s*%|knockdown|knockout|overexpress", ql):
                questions.append(
                    f"{len(questions)+1}. What is the intervention magnitude you want to simulate? "
                    "(e.g. 80% inhibition, complete knockout, 2-fold overexpression)"
                )

        if not questions or len(questions) < 2:
            questions.append(
                f"{len(questions)+1}. What is your primary analysis goal? "
                "(a) Identify all causal drivers of the phenotype  "
                "(b) Prioritise druggable/therapeutic targets  "
                "(c) Understand the mechanistic pathway  "
                "(d) Compare causal structure between two groups"
            )

        can_proceed = len(blocking) == 0
        return {
            "questions": questions[:4],
            "can_proceed_without_answers": can_proceed,
            "blocking_question_indices": blocking[:4],
        }


class _LocalClaimEngine:
    """Returns empty claims (no real abstracts without API)."""

    def extract(self, abstracts_text: str) -> dict:
        return {"claims": []}


class _LocalBriefEngine:
    """Template planning brief for local fallback."""

    def build(self, context_str: str) -> dict:
        return {
            "inferred_context":       "molecular biology phenotype",
            "prior_evidence_summary": "Literature search skipped — no API key available.",
            "causal_vs_associative":  "mixed",
            "recommended_modules":    ["M12", "M13", "M14", "M15", "M_DC"],
            "data_gaps":              ["expression matrix", "metadata with phenotype labels"],
            "supervisor_brief":       (
                "No API key available — running with local fallback engines. "
                "Causal analysis will proceed with available data. "
                "Upload expression and metadata files to unlock full pipeline. "
                "Set ANTHROPIC_API_KEY for richer literature and synthesis outputs."
            ),
        }


class _LocalNarrateEngine:
    """Returns step description as narration fallback."""

    def narrate(self, step: dict) -> str:
        sid  = step.get("id", "")
        desc = step.get("desc", "")
        label = step.get("label", sid)
        templates = {
            "T_00": (
                "Cohort Data Retrieval searches GEO/SRA for a matching dataset using the disease name from the query. "
                "It retrieves raw count matrices and metadata so the pipeline has expression data to work with. "
                "The retrieved files flow directly into T_01 normalization."
            ),
            "T_01": (
                "Expression Normalization applies log2-CPM transformation and QC filtering to the raw count matrix. "
                "It removes low-quality genes and samples, producing a clean normalized matrix. "
                "The normalized matrix (expr_norm.parquet) feeds into T_02, T_04, and T_05."
            ),
            "T_02": (
                "Differential Expression Analysis (DESeq2) computes case-vs-control gene-level statistics. "
                "It identifies which genes are significantly up- or down-regulated between phenotype groups. "
                "The resulting DEG table drives pathway enrichment (T_03) and the DAGBuilder (M12)."
            ),
            "T_03": (
                "Pathway Enrichment runs ORA and GSEA across GO, KEGG, Reactome and MSigDB. "
                "It translates the gene-level DEG signal into biological pathway context. "
                "Enriched pathway sets are used by M12 as structural priors for the causal graph."
            ),
            "T_04": (
                "Cell-Type Deconvolution estimates immune and stromal cell-type fractions per sample. "
                "It uses CIBERSORT, BisQue (if sc_data uploaded), or built-in LM22 markers. "
                "Cell fractions flow into M12 as an immunological context layer."
            ),
            "T_04b": (
                "Single-Cell RNA-seq Pipeline performs QC, normalization, clustering, and cell-type annotation. "
                "It produces a reference atlas used by T_04 in BisQue deconvolution mode. "
                "Outputs also provide pseudobulk expression for M12."
            ),
            "T_05": (
                "Temporal / Pseudotime Pipeline fits an impulse model and computes Granger causality edges. "
                "It provides gene-level temporal ordering evidence for directed edge assignment in the DAG. "
                "Granger edges and temporal fits flow into M12 as directional priors."
            ),
            "T_06": (
                "CRISPR / Perturbation Pipeline computes ACE scores from DepMap or uploaded CRISPR data. "
                "It identifies which genes are context-essential in cell lines matching the disease. "
                "CRISPR scores are weighted at 0.25 by M15 EvidenceAggregator."
            ),
            "T_07": (
                "Prior Knowledge Network assembles mechanistic edges from SIGNOR, STRING, and KEGG. "
                "It provides biochemically validated interaction priors to constrain the causal graph. "
                "Network edges are weighted at 0.15 by M15 and used by M12 for edge orientation."
            ),
            "T_08": (
                "GWAS / eQTL / MR Preprocessing fetches genetic association evidence from public databases. "
                "It identifies genetic instruments and computes MR-based causal effect estimates. "
                "Genetic evidence carries the highest weight (0.30) in M15 EvidenceAggregator."
            ),
            "M12": (
                "DAGBuilder runs PC/FCI/GES consensus structure learning across all evidence streams. "
                "It applies MR constraints as hard edge direction priors and bootstraps 1000x for stability. "
                "The output consensus_causal_dag.json is the central artifact for all downstream modules."
            ),
            "M13": (
                "DAGValidator tests edge stability across bootstrap replicates and checks PPI/pathway coherence. "
                "It removes unstable edges below the confidence threshold to produce the validated DAG. "
                "The validated DAG is the input for M14 centrality scoring."
            ),
            "M14": (
                "CentralityCalculator computes kME, betweenness, and PageRank for each DAG node. "
                "It combines these into a causal_importance_score and assigns Tier 1/2/3 rankings. "
                "Ranked targets feed M15 evidence aggregation and the final result synthesis."
            ),
            "M15": (
                "EvidenceAggregator fuses six evidence streams with weights: genetic(0.30), perturbation(0.25), "
                "temporal(0.20), network(0.15), expression(0.05), immuno(0.05). "
                "It also flags conflicting evidence across streams for each causal edge."
            ),
            "M_DC": (
                "DoCalculusEngine applies Judea Pearl's Backdoor Criterion to test edge directionality. "
                "It identifies valid adjustment sets to remove confounding and estimates causal effects. "
                "Results confirm or refute the causal direction for each top-ranked candidate."
            ),
            "M_IS": (
                "InSilicoSimulator propagates a do(X=0) intervention through the validated DAG. "
                "It predicts downstream expression changes, compensation pathways, and resistance mechanisms. "
                "Outputs guide M_PI drug target prioritisation."
            ),
            "M_PI": (
                "PharmaInterventionEngine scores each ranked target against DrugBank, ChEMBL, DepMap, and LINCS. "
                "It balances predicted therapeutic efficacy against systemic safety signals. "
                "The output Target Product Profile summarises druggability for each candidate."
            ),
            "DELTA": (
                "Delta Analysis compares the two group-specific DAGs to identify shared and group-specific edges. "
                "It computes delta causal_importance_scores to reveal rewired regulatory architecture. "
                "Conserved edges represent robust drivers; group-specific edges reveal context-dependent mechanisms."
            ),
        }
        return templates.get(sid, desc or f"{label}: processing step.")


class _LocalSynthesisEngine:
    """Template final result for local fallback."""

    def synthesise(self, context: str) -> dict:
        return {
            "headline":           "Analysis pipeline executed — modules pending implementation.",
            "analyzed_context":   "molecular biology phenotype of interest",
            "top_findings":       [
                "Expression normalization and QC completed.",
                "Causal modules (M12-M_DC) are registered and awaiting implementation.",
                "Upload additional data files to improve evidence coverage.",
            ],
            "tier1_candidates":   [],
            "tier2_candidates":   [],
            "actionable_targets": [],
            "evidence_quality": {
                "streams_present":    ["expression"],
                "streams_missing":    ["genetic", "perturbation", "temporal", "network", "immuno"],
                "overall_confidence": "low",
                "note": "Causal modules not yet implemented — results are preliminary.",
            },
            "caveats": [
                "M12 DAGBuilder and downstream modules are not yet implemented.",
                "Set ANTHROPIC_API_KEY for richer synthesis and literature search.",
            ],
            "next_experiments":    ["Implement M12 DAGBuilder in tool_registry.py"],
            "missing_data_impact": ["All causal evidence streams need implementation"],
        }


class LocalClient:
    """
    Deterministic rule-based client used when no Anthropic API key is available.
    Routes requests to the appropriate local engine by fingerprinting the system prompt.
    """

    def __init__(self):
        self._intent    = _LocalIntentEngine()
        self._clarif    = _LocalClarificationEngine()
        self._claims    = _LocalClaimEngine()
        self._brief     = _LocalBriefEngine()
        self._narrate   = _LocalNarrateEngine()
        self._synthesis = _LocalSynthesisEngine()

    def complete(self, user_msg: str, system_prompt: str, max_tokens: int, as_json: bool = True):
        sys_l = system_prompt.lower()
        if "intake agent" in sys_l:
            return self._clarif.generate(user_msg, [], "")
        if "intent" in sys_l and "i_01" in sys_l:
            return self._intent.classify(user_msg)
        if "claim extractor" in sys_l:
            return self._claims.extract(user_msg)
        if "planning brief" in sys_l:
            return self._brief.build(user_msg)
        if "narrating a live" in sys_l:
            # user_msg contains step info — try to parse step id
            step_id = ""
            m = re.search(r"Step:\s*\S+\s*\((\w+)\)", user_msg)
            if m:
                step_id = m.group(1)
            return self._narrate.narrate({"id": step_id, "desc": user_msg[:120]})
        if "synthesising the final" in sys_l:
            return self._synthesis.synthesise(user_msg)
        # Default
        return {"raw": user_msg[:200]}


class ClaudeClient:
    """
    Anthropic API wrapper with transparent local fallback.

    When ANTHROPIC_API_KEY is set and anthropic SDK is installed -> real API.
    Otherwise -> LocalClient (rule-based, no HTTP, no cost).
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if _ANTHROPIC_AVAILABLE and key:
            self._client = _anthropic_sdk.Anthropic(api_key=key)
            self._local  = None
            log.info("ClaudeClient: using Anthropic API (model=%s)", ANTHROPIC_MODEL)
        else:
            self._client = None
            self._local  = LocalClient()
            reason = "anthropic SDK not installed" if not _ANTHROPIC_AVAILABLE else "no API key"
            log.info("ClaudeClient: using local fallback engines (%s)", reason)

    def complete(
        self,
        user_msg:      str,
        system_prompt: str,
        max_tokens:    int,
        as_json:       bool = True,
    ):
        """
        Call the API (or local fallback) and return a dict (as_json=True) or str.
        Never raises — on failure returns {} or "" respectively.
        """
        if self._local is not None:
            result = self._local.complete(user_msg, system_prompt, max_tokens, as_json)
            if not as_json:
                return result if isinstance(result, str) else json.dumps(result)
            return result if isinstance(result, dict) else {}

        try:
            msg = self._client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = msg.content[0].text.strip()
            if not as_json:
                return text
            # Strip markdown code fences if present
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            return json.loads(text)
        except Exception as exc:
            log.warning("ClaudeClient.complete failed: %s", exc)
            return {} if as_json else ""


# =============================================================================
# SECTION 2 — ClarificationEngine
# =============================================================================

_CLARIF_SYSTEM = """\
You are the intake agent for a molecular causal discovery platform.

Given the user's query and the files they have uploaded, generate 2-4 concise
clarifying questions to collect everything needed to run the analysis correctly.

Focus ONLY on genuinely ambiguous or missing information:
1. Confirm disease/phenotype and specific outcome variable (if not explicit)
2. Organism and tissue/cell type (if not mentioned)
3. Which optional data types they still plan to upload (GWAS, CRISPR, temporal,
   eQTL, prior network) — do NOT ask about files already uploaded
4. Specific gene/entity names for I_02 (X->Y) or I_05 (simulate X) queries
5. Analysis goal: full driver landscape vs druggable targets only

Rules:
- If something is already clear from the query, do NOT ask about it.
- Never ask about expression matrix or metadata if already uploaded.
- Keep each question to one sentence. Number them 1-4.
- If the query is fully unambiguous and files cover everything critical,
  return an empty list and set can_proceed_without_answers to true.

Return ONLY valid JSON (no markdown):
{
  "questions": ["1. ...", "2. ...", ...],
  "can_proceed_without_answers": true/false,
  "blocking_question_indices": [1, 2]
}"""


@dataclass
class ClarificationResult:
    questions:        list = field(default_factory=list)
    answers:          list = field(default_factory=list)
    enriched_query:   str  = ""
    enriched_context: dict = field(default_factory=dict)
    can_proceed:      bool = True
    skipped:          bool = False


class ClarificationEngine:
    """Generates clarifying questions and builds enriched context from answers."""

    _ORGANISMS = ["human", "mouse", "rat", "zebrafish", "drosophila", "yeast",
                  "arabidopsis", "c. elegans", "pig", "non-human primate"]
    _TISSUES   = ["liver", "lung", "brain", "cortex", "heart", "kidney", "colon",
                  "breast", "skin", "blood", "pbmc", "neuron", "hepatocyte",
                  "fibroblast", "macrophage", "t cell", "b cell", "nk cell",
                  "stem cell", "organoid", "tumor", "stroma", "endothel"]

    def __init__(self, claude: ClaudeClient):
        self._claude = claude
        self._local  = _LocalClarificationEngine()

    def generate_questions(
        self,
        query:       str,
        file_types:  list,
        intent_name: str = "",
    ) -> list:
        files_str = ", ".join(file_types) if file_types else "none uploaded yet"
        user_msg  = (
            f"Query: {query}\n"
            f"Files already uploaded: {files_str}\n"
            f"Detected intent: {intent_name}"
        )
        try:
            resp = self._claude.complete(user_msg, _CLARIF_SYSTEM, 500)
            if isinstance(resp, dict):
                return resp.get("questions", [])
        except Exception as exc:
            log.debug("ClarificationEngine API failed: %s — using local engine", exc)
        resp = self._local.generate(query, file_types, intent_name)
        return resp.get("questions", [])

    def build_result(
        self,
        original_query: str,
        questions:      list,
        answers:        list,
        skipped:        bool = False,
    ) -> ClarificationResult:
        if skipped or not any(a.strip() for a in answers):
            return ClarificationResult(
                questions=questions, answers=answers,
                enriched_query=original_query,
                enriched_context={},
                can_proceed=True, skipped=True,
            )
        qa_lines = []
        for q, a in zip(questions, answers):
            if a.strip():
                qa_lines.append(f"{q.strip()} -> {a.strip()}")
        enriched_query = original_query
        if qa_lines:
            enriched_query += "\n[User clarifications: " + "; ".join(qa_lines) + "]"
        ctx = self._extract_context(original_query + " " + " ".join(answers))
        return ClarificationResult(
            questions=questions, answers=answers,
            enriched_query=enriched_query, enriched_context=ctx,
            can_proceed=True, skipped=False,
        )

    @staticmethod
    def _extract_context(text: str) -> dict:
        tl = text.lower()
        ctx: dict = {}
        organisms = ["human", "mouse", "rat", "zebrafish", "drosophila", "yeast",
                     "arabidopsis", "c. elegans", "pig"]
        tissues   = ["liver", "lung", "brain", "cortex", "heart", "kidney", "colon",
                     "breast", "skin", "blood", "pbmc", "neuron", "hepatocyte",
                     "fibroblast", "macrophage", "t cell", "b cell", "nk cell",
                     "stem cell", "organoid", "tumor"]
        for org in organisms:
            if org in tl:
                ctx["organism"] = org
                break
        for tissue in tissues:
            if tissue in tl:
                ctx["tissue_or_cell_type"] = tissue
                break
        if any(w in tl for w in ["drug", "druggable", "therapeutic", "target"]):
            ctx["goal"] = "druggable_targets"
        elif any(w in tl for w in ["mechanism", "pathway", "how does"]):
            ctx["goal"] = "mechanistic"
        elif any(w in tl for w in ["compare", "difference", "group", "responder"]):
            ctx["goal"] = "comparative"
        else:
            ctx["goal"] = "full_driver_landscape"
        mag = re.search(r"(\d+)\s*%", text)
        if mag:
            ctx["intervention_magnitude"] = f"{mag.group(1)}%"
        for kw in ["knockdown", "knockout", "overexpress", "inhibit", "activat"]:
            if kw in tl:
                ctx["intervention_type"] = kw
                break
        return ctx


# =============================================================================
# SECTION 3 — IntentClassifier
# =============================================================================

_INTENT_SYSTEM = """\
You are the Supervisor Agent for a molecular causal discovery platform.
The platform works for ANY molecular domain: oncology, immunology, neuroscience,
metabolism, aging, developmental biology, drug response, microbiome, plant biology,
or any other -omics study. Never assume a specific disease or organism.

Extract domain context entirely from the user's query.

INTENT TYPES:
I_01 = Causal Drivers Discovery — identify what causally drives a phenotype/trait
I_02 = Directed Causality X->Y — test if entity X causally affects outcome Y
I_03 = Intervention / Actionability — rank therapeutic or experimental targets
I_04 = Comparative Causality — compare causal mechanisms across groups/conditions/tissues
I_05 = Counterfactual / What-If — simulate perturbations and predict effects
I_06 = Evidence Inspection — explain rankings, evidence quality, data gaps
I_07 = Standard Association Analysis — DEGs, pathways, correlations (no causal claims)

ROUTING RULES:
- I_01: needs cohort expression + phenotype labels; optionally genetics, perturbation, temporal
- I_02: needs specific X and Y entities; targeted causal test
- I_03: needs existing DAG or intervention data; reads from prior run
- I_04: needs group labels or two comparable conditions
- I_05: needs existing DAG; reads from prior run
- I_06: read-only; no new compute unless artifacts missing
- I_07: no causality modules — associations only

Return ONLY valid JSON (no markdown, no prose):
{
  "intent_id": "I_0X",
  "intent_name": "...",
  "confidence": 0.0-1.0,
  "needs_clarification": true/false,
  "clarifying_question": "specific question to resolve ambiguity, or null",
  "context": {
    "domain": "molecular/biological domain",
    "phenotype": "specific phenotype or outcome",
    "tissue_or_cell_type": "tissue/cell type or null",
    "organism": "human/mouse/etc or null",
    "outcome_variable": "specific outcome column or trait or null",
    "cohort_description": "brief description or null"
  },
  "entities": {
    "gene_x": null, "gene_y": null,
    "intervention": null, "magnitude": null,
    "groups": null, "comparison": null, "pathway": null
  },
  "requires": {
    "expression": true/false, "phenotype_labels": true/false,
    "gwas_eqtl": true/false, "perturbation": true/false,
    "temporal": true/false, "prior_network": true/false,
    "intervention_data": true/false
  },
  "needs_literature_first": true/false,
  "requires_existing_dag": true/false,
  "routing_summary": "one concise sentence",
  "module_chain": ["M12", "M13", "..."],
  "parallel_blocks": ["describe parallelizable groups"],
  "fallback": "fallback if critical data is missing"
}
"""

_DEFAULT_MODULE_CHAIN = ["M12", "M13", "M14", "M15", "M_DC"]


class IntentClassifier:
    """
    Classifies a user query into I_01-I_07 and returns a fully populated ParsedIntent.
    Falls back to I_01 on API failure.
    """

    def __init__(self, claude: ClaudeClient):
        self._claude = claude

    def classify(self, query: str):
        # Import here to avoid circular import at module load time
        from agent import ParsedIntent, IntentID

        try:
            resp = self._claude.complete(query, _INTENT_SYSTEM, MAX_TOKENS_INTENT)
            return ParsedIntent(
                intent_id              = IntentID(resp.get("intent_id", "I_01")),
                intent_name            = resp.get("intent_name", "Causal Drivers Discovery"),
                confidence             = float(resp.get("confidence", 0.7)),
                needs_clarification    = bool(resp.get("needs_clarification", False)),
                clarifying_question    = resp.get("clarifying_question"),
                context                = resp.get("context", {}),
                entities               = resp.get("entities", {}),
                requires               = resp.get("requires", {}),
                needs_literature_first = bool(resp.get("needs_literature_first", True)),
                requires_existing_dag  = bool(resp.get("requires_existing_dag", False)),
                routing_summary        = resp.get("routing_summary", ""),
                module_chain           = resp.get("module_chain", []),
                parallel_blocks        = resp.get("parallel_blocks", []),
                fallback               = resp.get("fallback", "Association-only ranking with disclaimer"),
            )
        except Exception as exc:
            log.warning("Intent classification failed: %s — defaulting to I_01", exc)
            return ParsedIntent(
                intent_id           = IntentID.I_01,
                intent_name         = "Causal Drivers Discovery",
                confidence          = 0.6,
                needs_clarification = False,
                clarifying_question = None,
                routing_summary     = "Default causal discovery route.",
                module_chain        = _DEFAULT_MODULE_CHAIN,
                fallback            = "Association-only ranking with disclaimer",
            )


# =============================================================================
# SECTION 4 — LiteraturePipeline
# =============================================================================

_SYS_PARSE = """\
You are the query parser for a molecular causal discovery platform.
Extract structured information from the user's query.
Return ONLY valid JSON (no markdown):
{
  "domain": "molecular biology domain",
  "phenotype": "specific phenotype or outcome",
  "tissue_or_cell_type": "relevant tissue/cell type or null",
  "organism": "human/mouse/etc or null",
  "key_genes": ["list of gene/protein names"],
  "key_pathways": ["relevant pathways"],
  "search_terms": ["3-5 PubMed MeSH-style search queries"],
  "study_type_filter": "causal/interventional/mechanistic"
}"""

_SYS_CLAIMS = """\
You are a molecular biology claim extractor for a causal inference platform.
Extract all directional causal or regulatory claims from the provided abstracts.
Return ONLY valid JSON (no markdown):
{
  "claims": [
    {
      "entity_x": "gene/protein/pathway name",
      "relation": "activates|inhibits|regulates|drives|causes|binds|phosphorylates|mediates",
      "entity_y": "gene/protein/pathway/phenotype name",
      "direction": "forward|reverse|bidirectional|unknown",
      "evidence_type": "genetic|perturbation|association|mechanistic|clinical",
      "strength": "strong|moderate|weak",
      "pmid": "PMID or null",
      "confidence": 0.0-1.0,
      "quote": "short supporting quote <=80 chars"
    }
  ]
}
Extract only genuine molecular claims. Skip general background statements."""

_SYS_CONFLICTS = """\
You are a molecular biology evidence analyst.
Given a list of causal claims, identify the most important conflicts or contradictions.
Return ONLY valid JSON: {"additional_conflicts": ["list of conflict descriptions <=100 chars each"]}"""

_SYS_BRIEF = """\
You are the literature planning brief builder for a molecular causal inference platform.
Synthesise the provided evidence into a planning brief for the supervisor agent.
Return ONLY valid JSON (no markdown):
{
  "inferred_context": "domain/phenotype inferred",
  "prior_evidence_summary": "2-3 sentences on what the literature shows",
  "causal_vs_associative": "causal/associative/mixed",
  "recommended_modules": ["module IDs most supported by evidence"],
  "data_gaps": ["data types needed but likely missing"],
  "supervisor_brief": "3-4 sentence executive summary"
}"""


class LiteraturePipeline:
    """Executes all 8 literature stages and returns a LitBrief."""

    def __init__(self, claude: ClaudeClient):
        self._claude = claude

    def run(self, query: str, intent_name: str, entities: dict):
        from agent import LiteraturePaper, LitClaim, LitBrief

        log.info("[LIT_01] Parsing query...")
        parsed_query = self._parse_query(query, intent_name, entities)

        log.info("[LIT_02] Expanding entities...")
        entity_list = self._expand_entities(parsed_query)

        log.info("[LIT_03] Building and executing search queries...")
        search_queries = self._build_search_queries(parsed_query, entity_list)
        raw_hits: list[dict] = []
        queries_used: list[str] = []

        for q in search_queries:
            pmids = self._search_pubmed(q)
            if pmids:
                queries_used.append(f"PubMed: {q}")
                for pmid in pmids[:8]:
                    raw_hits.append({"pmid": pmid, "source": "pubmed", "title": "", "abstract": ""})
            epmc = self._search_europepmc(q)
            if epmc:
                queries_used.append(f"EPMC: {q}")
                raw_hits.extend(epmc)

        if search_queries:
            s2 = self._search_semantic_scholar(search_queries[0])
            if s2:
                raw_hits.extend(s2)

        log.info("[LIT_04] Deduplicating %d raw hits...", len(raw_hits))
        papers = self._deduplicate_and_rank(raw_hits, LiteraturePaper)

        log.info("[LIT_05] Fetching abstracts for %d papers...", len(papers))
        papers = self._fetch_abstracts(papers)
        papers_with_content = [p for p in papers if p.abstract or p.title]

        log.info("[LIT_06] Extracting claims from %d papers...", len(papers_with_content))
        claims = self._extract_claims(papers_with_content)

        log.info("[LIT_07] Grading evidence and detecting conflicts...")
        high_conf_edges, conflicts, conflict_rate = self._grade_evidence(claims)

        log.info("[LIT_08] Building supervisor planning brief...")
        brief_dict = self._build_planning_brief(
            parsed_query, papers_with_content, claims, high_conf_edges, conflicts, conflict_rate,
        )

        brief = LitBrief(
            inferred_context       = brief_dict.get("inferred_context", parsed_query.get("domain", "")),
            key_entities           = entity_list[:12],
            search_queries_used    = queries_used[:6],
            papers_found           = len(raw_hits),
            papers_processed       = len(papers_with_content),
            claims                 = [self._dict_to_claim(c, LitClaim) for c in claims],
            high_confidence_edges  = high_conf_edges,
            conflicts              = conflicts,
            causal_vs_associative  = brief_dict.get("causal_vs_associative", "mixed"),
            prior_evidence_summary = brief_dict.get("prior_evidence_summary", ""),
            recommended_modules    = brief_dict.get("recommended_modules", []),
            data_gaps              = brief_dict.get("data_gaps", []),
            supervisor_brief       = brief_dict.get("supervisor_brief", ""),
            conflict_rate          = conflict_rate,
        )
        log.info(
            "[LIT] Complete: %d papers, %d claims, %d high-conf edges, %d conflicts",
            brief.papers_processed, len(brief.claims),
            len(brief.high_confidence_edges), len(brief.conflicts),
        )
        return brief

    def _parse_query(self, query: str, intent_name: str, entities: dict) -> dict:
        try:
            return self._claude.complete(
                f"Query: {query}\nIntent: {intent_name}\nEntities: {entities}",
                _SYS_PARSE, 500,
            )
        except Exception as exc:
            log.debug("LIT_01 failed: %s", exc)
            return {"domain": "molecular biology", "key_genes": [],
                    "search_terms": [query[:100]], "phenotype": ""}

    def _expand_entities(self, parsed: dict) -> list:
        terms = (
            list(parsed.get("key_genes", []))
            + list(parsed.get("key_pathways", []))
            + list(parsed.get("search_terms", []))
        )
        phenotype = parsed.get("phenotype", "")
        if phenotype:
            terms.append(phenotype)
        seen, out = set(), []
        for t in terms:
            t = t.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out[:20]

    def _build_search_queries(self, parsed: dict, entities: list) -> list:
        domain    = parsed.get("domain", "")
        phenotype = parsed.get("phenotype", "")
        custom    = parsed.get("search_terms", [])
        queries   = list(custom[:3])
        if phenotype and domain:
            queries.append(
                f"{phenotype}[Title/Abstract] AND {domain}[Title/Abstract] AND (causal OR mechanism)"
            )
        if entities:
            top = " OR ".join(f'"{e}"' for e in entities[:4])
            queries.append(f"({top}) AND causal[Title/Abstract]")
        if phenotype:
            queries.append(
                f"{phenotype}[Title/Abstract] AND (gene expression OR transcriptomics OR GWAS)"
            )
        seen, deduped = set(), []
        for q in queries:
            if q.strip() and q.strip() not in seen:
                seen.add(q.strip())
                deduped.append(q.strip())
        return deduped[:5]

    def _search_pubmed(self, query: str) -> list:
        try:
            params = {
                "db": "pubmed", "term": query, "retmax": 15,
                "sort": "relevance", "retmode": "json",
                "tool": "CausalPlatform", "email": _EMAIL,
            }
            with httpx.Client(timeout=LIT_TIMEOUT) as c:
                r = c.get(f"{PUBMED_BASE}/esearch.fcgi", params=params)
                r.raise_for_status()
                return r.json().get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            log.debug("PubMed search failed: %s", exc)
            return []

    def _search_europepmc(self, query: str) -> list:
        try:
            params = {
                "query": query, "format": "json",
                "pageSize": 10, "resultType": "core", "sort": "RELEVANCE",
            }
            with httpx.Client(timeout=LIT_TIMEOUT) as c:
                r = c.get(f"{EPMC_BASE}/search", params=params)
                r.raise_for_status()
                items = r.json().get("resultList", {}).get("result", [])
                return [
                    {
                        "pmid":     item.get("pmid"),
                        "doi":      item.get("doi"),
                        "title":    item.get("title", ""),
                        "abstract": item.get("abstractText", ""),
                        "year":     item.get("pubYear"),
                        "journal":  item.get("journalTitle", ""),
                        "source":   "europepmc",
                    }
                    for item in items
                ]
        except Exception as exc:
            log.debug("EuropePMC search failed: %s", exc)
            return []

    def _search_semantic_scholar(self, query: str) -> list:
        try:
            params = {
                "query": query, "limit": 8,
                "fields": "title,year,abstract,externalIds,venue",
            }
            with httpx.Client(timeout=LIT_TIMEOUT) as c:
                r = c.get(f"{S2_BASE}/paper/search", params=params)
                r.raise_for_status()
                return [
                    {
                        "pmid":     item.get("externalIds", {}).get("PubMed"),
                        "doi":      item.get("externalIds", {}).get("DOI"),
                        "title":    item.get("title", ""),
                        "abstract": item.get("abstract", ""),
                        "year":     item.get("year"),
                        "journal":  item.get("venue", ""),
                        "source":   "semanticscholar",
                    }
                    for item in r.json().get("data", [])
                ]
        except Exception as exc:
            log.debug("Semantic Scholar failed: %s", exc)
            return []

    def _deduplicate_and_rank(self, raw_hits: list, LiteraturePaper_cls) -> list:
        seen_pmids: set = set()
        seen_dois:  set = set()
        seen_titles:set = set()
        unique: list[dict] = []
        for h in raw_hits:
            pmid  = str(h.get("pmid") or "").strip()
            doi   = str(h.get("doi")  or "").strip().lower()
            title = str(h.get("title") or "").strip().lower()[:80]
            if pmid  and pmid  in seen_pmids:  continue
            if doi   and doi   in seen_dois:   continue
            if title and title in seen_titles: continue
            if pmid:  seen_pmids.add(pmid)
            if doi:   seen_dois.add(doi)
            if title: seen_titles.add(title)
            unique.append(h)

        def _score(h: dict) -> float:
            has_abstract = 2.0 if h.get("abstract") else 0.0
            year         = int(h.get("year") or 2000)
            causal_bonus = sum(
                1 for kw in ["causal", "gwas", "crispr", "knockdown", "mendelian",
                              "perturbation", "driver", "mechanism"]
                if kw in (h.get("title") or "").lower()
            )
            return has_abstract + (year - 2000) * 0.1 + causal_bonus

        unique.sort(key=_score, reverse=True)
        return [
            LiteraturePaper_cls(
                pmid=h.get("pmid"), doi=h.get("doi"),
                title=h.get("title", ""), abstract=h.get("abstract", ""),
                year=h.get("year"), journal=h.get("journal", ""),
                source=h.get("source", ""),
            )
            for h in unique[:LIT_MAX_PAPERS]
        ]

    def _fetch_abstracts(self, papers: list) -> list:
        need = [p for p in papers if not p.abstract and p.pmid][:10]
        if not need:
            return papers
        pmids = ",".join(p.pmid for p in need if p.pmid)
        try:
            params = {
                "db": "pubmed", "id": pmids,
                "rettype": "abstract", "retmode": "xml",
                "tool": "CausalPlatform", "email": _EMAIL,
            }
            with httpx.Client(timeout=LIT_TIMEOUT) as c:
                r = c.get(f"{PUBMED_BASE}/efetch.fcgi", params=params)
                r.raise_for_status()
                root = ET.fromstring(r.text)
                abstract_map: dict[str, str] = {}
                for article in root.findall(".//PubmedArticle"):
                    pmid_el = article.find(".//PMID")
                    abs_els = article.findall(".//AbstractText")
                    if pmid_el is not None and abs_els:
                        abstract_map[pmid_el.text.strip()] = " ".join(
                            (el.text or "") for el in abs_els if el.text
                        ).strip()
                for p in need:
                    if p.pmid and p.pmid in abstract_map:
                        p.abstract = abstract_map[p.pmid]
        except Exception as exc:
            log.debug("Abstract fetch failed: %s", exc)
        return papers

    def _extract_claims(self, papers: list) -> list:
        with_abstract = [p for p in papers if p.abstract][:LIT_TOP_K]
        if not with_abstract:
            return []
        entries = [
            f"Paper {i+1} (PMID:{p.pmid or 'N/A'}, {p.year or '?'}):\n"
            f"Title: {p.title}\nAbstract: {p.abstract[:800]}"
            for i, p in enumerate(with_abstract)
        ]
        try:
            resp = self._claude.complete("\n\n---\n".join(entries), _SYS_CLAIMS, MAX_TOKENS_CLAIMS)
            return resp.get("claims", []) if isinstance(resp, dict) else []
        except Exception as exc:
            log.debug("LIT_06 claim extraction failed: %s", exc)
            return []

    def _grade_evidence(self, claims: list) -> tuple:
        if not claims:
            return [], [], 0.0
        high_conf = [
            {
                "from":      c["entity_x"],
                "to":        c["entity_y"],
                "relation":  c["relation"],
                "mechanism": f"{c['evidence_type']}: {c.get('quote', '')}",
                "strength":  c["strength"],
                "pmid":      c.get("pmid"),
            }
            for c in claims
            if c.get("evidence_type") in ("genetic", "perturbation")
            and c.get("strength") == "strong"
            and float(c.get("confidence", 0)) >= 0.75
        ]
        pair_rels: dict = {}
        for c in claims:
            key = (c["entity_x"].lower(), c["entity_y"].lower())
            pair_rels.setdefault(key, []).append(c["relation"])
        rule_conflicts = [
            f"{x} -> {y}: conflicting activation vs inhibition"
            for (x, y), rels in pair_rels.items()
            if ("activates" in rels or "drives" in rels) and ("inhibits" in rels)
        ]
        llm_conflicts: list = []
        if rule_conflicts and len(claims) > 5:
            claim_summary = "; ".join(
                f"{c['entity_x']} {c['relation']} {c['entity_y']} ({c['evidence_type']}, {c['strength']})"
                for c in claims[:20]
            )
            try:
                resp = self._claude.complete(claim_summary, _SYS_CONFLICTS, 400)
                if isinstance(resp, dict):
                    llm_conflicts = resp.get("additional_conflicts", [])
            except Exception:
                pass
        all_conflicts = list(set(rule_conflicts + llm_conflicts))[:5]
        conflict_rate = len(all_conflicts) / max(len(claims), 1)
        return high_conf[:10], all_conflicts, conflict_rate

    def _build_planning_brief(
        self, parsed_query: dict, papers: list, claims: list,
        high_conf_edges: list, conflicts: list, conflict_rate: float,
    ) -> dict:
        paper_titles  = "; ".join(f"{p.title[:60]} ({p.year})" for p in papers[:8])
        claim_summary = "; ".join(
            f"{c['entity_x']} {c['relation']} {c['entity_y']} ({c['evidence_type']}, {c['strength']})"
            for c in claims[:15]
        )
        context_str = str({
            k: parsed_query.get(k)
            for k in ["domain", "phenotype", "tissue_or_cell_type", "study_type_filter"]
        })
        try:
            resp = self._claude.complete(
                f"Context: {context_str}\n"
                f"Papers ({len(papers)}): {paper_titles}\n"
                f"Claims ({len(claims)}): {claim_summary}\n"
                f"Conflicts: {'; '.join(conflicts) or 'none'}",
                _SYS_BRIEF, MAX_TOKENS_BRIEF,
            )
            return resp if isinstance(resp, dict) else {}
        except Exception as exc:
            log.debug("LIT_08 brief failed: %s", exc)
            return {}

    @staticmethod
    def _dict_to_claim(c: dict, LitClaim_cls) -> Any:
        return LitClaim_cls(
            entity_x      = c.get("entity_x", ""),
            relation       = c.get("relation", "regulates"),
            entity_y       = c.get("entity_y", ""),
            direction      = c.get("direction", "unknown"),
            evidence_type  = c.get("evidence_type", "association"),
            strength       = c.get("strength", "weak"),
            pmid           = c.get("pmid"),
            quote          = c.get("quote", "")[:120],
            confidence     = float(c.get("confidence", 0.5)),
        )


# =============================================================================
# SECTION 5 — ResultSynthesiser
# =============================================================================

_SYNTHESISER_SYSTEM = """\
You are the Supervisor Agent synthesising the final result of a molecular causal analysis.
Adapt completely to the domain, phenotype, and context that was analyzed.
Never use language from a different molecular domain. Be precise and statistical.

Return ONLY valid JSON (no markdown):
{
  "headline": "one sentence result with specific domain and phenotype",
  "analyzed_context": "what domain/phenotype/condition was analyzed",
  "top_findings": ["3-5 specific findings with entity names, directions, and effect sizes"],
  "tier1_candidates": ["Tier-1 master regulator names"],
  "tier2_candidates": ["Tier-2 secondary driver names"],
  "actionable_targets": [
    {
      "entity": "gene/protein name",
      "action": "inhibit|activate|knockdown|overexpress",
      "druggability": "high|medium|low",
      "existing_drug": "approved drug name or null",
      "rationale": "one sentence specific to this domain and phenotype"
    }
  ],
  "evidence_quality": {
    "streams_present": ["genetic","perturbation","temporal","network","expression"],
    "streams_missing": ["list of missing evidence streams"],
    "overall_confidence": "high|moderate|low",
    "note": "sentence on evidence completeness for this specific analysis"
  },
  "caveats": ["methodological limitations specific to this analysis and domain"],
  "next_experiments": ["2-3 specific follow-up experiments for this domain and phenotype"],
  "missing_data_impact": ["specific data types and how absence affects confidence"]
}"""


class ResultSynthesiser:
    """Calls Claude with run context and returns a structured FinalResult."""

    def __init__(self, claude: ClaudeClient):
        self._claude = claude

    def synthesise(
        self,
        query:          str,
        intent,
        steps_run:      list,
        lit_brief,
        audits:         list,
        artifact_store: Optional[dict] = None,
    ):
        from agent import FinalResult

        artifact_store  = artifact_store or {}
        modules_done    = [s["id"] for s in steps_run if s.get("status") == "done"]
        modules_pending = [s["id"] for s in steps_run if s.get("status") == "pending"]
        artifacts       = [o for s in steps_run if not s.get("skip")
                           for o in s.get("outputs", [])]

        lit_summary  = lit_brief.supervisor_brief if lit_brief else "No literature search performed."
        key_entities = lit_brief.key_entities[:8] if lit_brief else []

        real_artifacts_note = (
            f"Real tool outputs available: {', '.join(artifact_store.keys())}"
            if artifact_store else
            f"Modules pending (not yet implemented): {', '.join(modules_pending)}"
        )

        user_msg = (
            f"Query: {query}\n"
            f"Context: {json.dumps(intent.context)}\n"
            f"Intent: {intent.intent_name}\n"
            f"Modules completed: {', '.join(modules_done)}\n"
            f"Files: {', '.join(a.type_label for a in audits)}\n"
            f"Literature brief: {lit_summary}\n"
            f"Key entities from literature: {', '.join(key_entities)}\n"
            f"{real_artifacts_note}"
        )

        try:
            resp = self._claude.complete(user_msg, _SYNTHESISER_SYSTEM, MAX_TOKENS_RESULT)
            if isinstance(resp, dict) and "raw" not in resp:
                return FinalResult(
                    headline            = resp.get("headline", "Analysis complete."),
                    analyzed_context    = resp.get("analyzed_context", intent.context.get("domain", "")),
                    top_findings        = resp.get("top_findings", []),
                    tier1_candidates    = resp.get("tier1_candidates", []),
                    tier2_candidates    = resp.get("tier2_candidates", []),
                    actionable_targets  = resp.get("actionable_targets", []),
                    evidence_quality    = resp.get("evidence_quality", {}),
                    caveats             = resp.get("caveats", []),
                    next_experiments    = resp.get("next_experiments", []),
                    missing_data_impact = resp.get("missing_data_impact", []),
                    modules_run         = modules_done,
                    artifacts_produced  = list(set(artifacts)),
                )
        except Exception as exc:
            log.warning("ResultSynthesiser failed: %s", exc)

        return FinalResult(
            headline           = "Analysis complete — review logs for details.",
            modules_run        = modules_done,
            artifacts_produced = list(set(artifacts)),
        )


# =============================================================================
# SECTION 6 — StepNarrator
# =============================================================================

_NARRATE_SYSTEM = """\
You are the Supervisor Agent narrating a live molecular analysis pipeline step.
The user is a scientist — be precise about the statistical method and its purpose.
Adapt your language to the molecular domain and phenotype being studied.

Write exactly 3 sentences:
1. Name the specific algorithm or statistical method being applied.
2. What molecular or statistical question it answers in this specific analysis context.
3. What the output feeds into downstream and why it matters for the final result.

Return plain text only. No markdown. No headers. No bullet points."""


class StepNarrator:
    """Generates 3-sentence plain-text narrations for pipeline steps."""

    def __init__(self, claude: ClaudeClient):
        self._claude = claude
        self._local  = _LocalNarrateEngine()

    def narrate(self, step: dict, intent) -> str:
        ctx = intent.context or {}
        user_msg = (
            f"Step: {step['label']} ({step['id']})\n"
            f"Algorithm: {step.get('tool', '')}\n"
            f"Outputs: {', '.join(step.get('outputs', []))}\n"
            f"Domain: {ctx.get('domain', 'molecular biology')}\n"
            f"Phenotype: {ctx.get('phenotype', 'the phenotype of interest')}\n"
            f"Intent: {intent.intent_name}"
        )
        try:
            result = self._claude.complete(user_msg, _NARRATE_SYSTEM, MAX_TOKENS_NARRATE, as_json=False)
            if result and isinstance(result, str) and len(result) > 20:
                return result
        except Exception:
            pass
        return self._local.narrate(step)
