"""
Phase 7: INSPECTION AND LLM ARBITRATION — Module 3
LLMArbitrator (Combined for all modules in each intent)
========================================================
Arbitrates between conflicting evidence using LLM-powered structured reasoning.

Uses the Bio-RAG knowledge base (8.6M vectors) for context and Claude/GPT
for structured arbitration of:
  1. Conflicting edge evidence (statistical vs experimental)
  2. Ambiguous directionality
  3. Disputed tier classifications
  4. Novel causal claims without precedent
  5. Cross-module disagreements

Organization: Ayass Bioscience LLC
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArbitratorConfig:
    confidence_threshold: float = 0.65
    max_conflicts_per_batch: int = 50
    require_minimum_evidence: int = 2
    llm_call: Optional[Callable] = None
    model: str = 'claude-sonnet-4-20250514'
    temperature: float = 0.1
    max_tokens: int = 2000


class LLMArbitrator:
    """Arbitrates conflicting causal evidence using LLM reasoning."""

    def __init__(self, config: Optional[ArbitratorConfig] = None):
        self.config = config or ArbitratorConfig()

    def arbitrate_dag(self, dag: nx.DiGraph,
                      inspection_report: Optional[Dict] = None,
                      gap_report: Optional[Dict] = None) -> Dict:
        """Run arbitration on all conflicts in the DAG."""
        conflicts = self._identify_conflicts(dag, inspection_report, gap_report)

        resolutions = []
        for conflict in conflicts[:self.config.max_conflicts_per_batch]:
            resolution = self._arbitrate_conflict(conflict, dag)
            resolutions.append(resolution)

            self._apply_resolution(resolution, dag)

        return {
            'total_conflicts': len(conflicts),
            'resolved': len(resolutions),
            'resolutions': resolutions,
            'summary': {
                'actions_taken': self._summarize_actions(resolutions),
                'confidence_distribution': self._confidence_distribution(resolutions),
            },
        }

    def arbitrate_edge(self, dag: nx.DiGraph, u: str, v: str) -> Dict:
        """Arbitrate a specific edge."""
        data = dag[u][v] if dag.has_edge(u, v) else {}
        conflict = self._build_edge_conflict(u, v, data, dag)
        return self._arbitrate_conflict(conflict, dag)

    def _identify_conflicts(self, dag: nx.DiGraph,
                             inspection_report: Optional[Dict],
                             gap_report: Optional[Dict]) -> List[Dict]:
        """Identify all conflicts requiring arbitration."""
        conflicts = []

        for u, v, data in dag.edges(data=True):
            flags = data.get('hallucination_flags', [])
            if flags:
                conflicts.append(self._build_edge_conflict(u, v, data, dag,
                                                           reason='hallucination_flagged'))

            conf_score = data.get('confounding_score', 0)
            if conf_score >= 0.4:
                conflicts.append(self._build_edge_conflict(u, v, data, dag,
                                                           reason='confounding_detected'))

            causal_passed = data.get('causal_test_passed', True)
            if not causal_passed:
                conflicts.append(self._build_edge_conflict(u, v, data, dag,
                                                           reason='failed_causality_test'))

            direction_rec = data.get('direction_recommendation', 'confirm')
            if direction_rec == 'uncertain':
                conflicts.append(self._build_edge_conflict(u, v, data, dag,
                                                           reason='ambiguous_directionality'))

        if inspection_report:
            for issue in inspection_report.get('issues', []):
                conflicts.append({
                    'type': 'inspection_issue',
                    'issue': issue,
                    'severity': issue.get('severity', 'warning'),
                })

        conflicts.sort(key=lambda c: {'critical': 0, 'high': 1, 'medium': 2,
                                       'warning': 2, 'low': 3}.get(c.get('severity', 'low'), 3))
        return conflicts

    def _build_edge_conflict(self, u: str, v: str, data: Dict,
                              dag: nx.DiGraph, reason: str = '') -> Dict:
        """Build a structured conflict description for an edge."""
        u_data = dag.nodes.get(u, {})
        v_data = dag.nodes.get(v, {})
        evidence = data.get('evidence', [])
        if isinstance(evidence, set):
            evidence = list(evidence)

        return {
            'type': 'edge_conflict',
            'edge': (u, v),
            'reason': reason,
            'severity': 'high' if reason in ('hallucination_flagged', 'failed_causality_test') else 'medium',
            'context': {
                'source_layer': u_data.get('layer', 'unknown'),
                'target_layer': v_data.get('layer', 'unknown'),
                'confidence': data.get('confidence_score', 0),
                'weight': data.get('weight', 0),
                'evidence': evidence,
                'causal_test_score': data.get('causal_test_score', None),
                'confounding_score': data.get('confounding_score', None),
                'hallucination_flags': data.get('hallucination_flags', []),
            },
        }

    def _arbitrate_conflict(self, conflict: Dict, dag: nx.DiGraph) -> Dict:
        """Arbitrate a single conflict using structured reasoning or LLM."""
        if self.config.llm_call:
            return self._llm_arbitrate(conflict, dag)
        return self._rule_based_arbitrate(conflict, dag)

    def _rule_based_arbitrate(self, conflict: Dict, dag: nx.DiGraph) -> Dict:
        """Rule-based arbitration when no LLM is available."""
        if conflict['type'] == 'edge_conflict':
            ctx = conflict.get('context', {})
            evidence = ctx.get('evidence', [])
            conf = ctx.get('confidence', 0)
            reason = conflict.get('reason', '')

            n_strong = sum(1 for ev in evidence
                           if any(s in str(ev).lower() for s in
                                  ['mendelian_randomization', 'gwas', 'crispr', 'signor']))

            if reason == 'hallucination_flagged':
                if n_strong >= 2:
                    action = 'retain_with_note'
                    rationale = f'Strong evidence ({n_strong} sources) overrides hallucination flag'
                elif n_strong == 1 and conf >= 0.5:
                    action = 'downweight'
                    rationale = 'Single strong evidence source; reduce confidence'
                else:
                    action = 'flag_for_removal'
                    rationale = 'Insufficient strong evidence to support flagged edge'

            elif reason == 'confounding_detected':
                if n_strong >= 2:
                    action = 'retain_with_adjustment'
                    rationale = 'Multiple causal evidence types mitigate confounding concern'
                else:
                    action = 'downweight'
                    rationale = 'Confounding detected without sufficient causal evidence'

            elif reason == 'failed_causality_test':
                if n_strong >= 3:
                    action = 'retain_with_note'
                    rationale = 'Overwhelming evidence despite failed statistical test'
                else:
                    action = 'flag_for_removal'
                    rationale = 'Failed causality test without strong compensating evidence'

            elif reason == 'ambiguous_directionality':
                action = 'flag_for_review'
                rationale = 'Direction uncertain; needs experimental validation'

            else:
                action = 'flag_for_review'
                rationale = 'Unclassified conflict type'

            return {
                'conflict': conflict,
                'action': action,
                'rationale': rationale,
                'confidence': round(min(1.0, n_strong * 0.3 + conf * 0.4), 4),
                'method': 'rule_based',
            }

        return {
            'conflict': conflict,
            'action': 'flag_for_review',
            'rationale': 'Non-edge conflict; manual review recommended',
            'confidence': 0.5,
            'method': 'rule_based',
        }

    def _llm_arbitrate(self, conflict: Dict, dag: nx.DiGraph) -> Dict:
        """LLM-powered arbitration."""
        prompt = self._build_arbitration_prompt(conflict)

        try:
            response = self.config.llm_call(
                prompt=prompt,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            parsed = self._parse_llm_response(response)
            return {
                'conflict': conflict,
                'action': parsed.get('action', 'flag_for_review'),
                'rationale': parsed.get('rationale', 'LLM arbitration'),
                'confidence': parsed.get('confidence', 0.5),
                'method': 'llm',
                'raw_response': response,
            }
        except Exception as e:
            logger.warning(f"LLM arbitration failed: {e}, falling back to rules")
            return self._rule_based_arbitrate(conflict, dag)

    def _build_arbitration_prompt(self, conflict: Dict) -> str:
        """Build structured prompt for LLM arbitration."""
        ctx = conflict.get('context', {})
        return (
            f"You are a causal inference expert arbitrating a conflict in a biological causal DAG.\n\n"
            f"Conflict Type: {conflict.get('reason', 'unknown')}\n"
            f"Edge: {conflict.get('edge', ('?', '?'))}\n"
            f"Source Layer: {ctx.get('source_layer', '?')} -> Target Layer: {ctx.get('target_layer', '?')}\n"
            f"Current Confidence: {ctx.get('confidence', 0)}\n"
            f"Evidence: {ctx.get('evidence', [])}\n"
            f"Hallucination Flags: {ctx.get('hallucination_flags', [])}\n"
            f"Confounding Score: {ctx.get('confounding_score', 'N/A')}\n\n"
            f"Decide: retain, downweight, flag_for_removal, or flag_for_review.\n"
            f"Respond with JSON: {{\"action\": \"...\", \"rationale\": \"...\", \"confidence\": 0.0-1.0}}"
        )

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response into structured resolution."""
        try:
            return json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return {'action': 'flag_for_review', 'rationale': response, 'confidence': 0.5}

    def _apply_resolution(self, resolution: Dict, dag: nx.DiGraph) -> None:
        """Apply arbitration resolution to the DAG."""
        action = resolution.get('action', '')
        conflict = resolution.get('conflict', {})

        if conflict.get('type') != 'edge_conflict':
            return

        edge = conflict.get('edge')
        if not edge or not dag.has_edge(edge[0], edge[1]):
            return

        u, v = edge
        dag[u][v]['arbitration_action'] = action
        dag[u][v]['arbitration_rationale'] = resolution.get('rationale', '')

        if action == 'downweight':
            current = dag[u][v].get('confidence_score', 0.5)
            dag[u][v]['confidence_score'] = round(current * 0.7, 4)

        elif action == 'flag_for_removal':
            dag[u][v]['flagged_for_removal'] = True

    def _summarize_actions(self, resolutions: List[Dict]) -> Dict[str, int]:
        """Summarize actions taken."""
        actions = {}
        for r in resolutions:
            a = r.get('action', 'unknown')
            actions[a] = actions.get(a, 0) + 1
        return actions

    def _confidence_distribution(self, resolutions: List[Dict]) -> Dict:
        """Summarize confidence of resolutions."""
        confs = [r.get('confidence', 0.5) for r in resolutions]
        if not confs:
            return {}
        return {
            'mean': round(float(np.mean(confs)), 4),
            'min': round(min(confs), 4),
            'max': round(max(confs), 4),
        }
