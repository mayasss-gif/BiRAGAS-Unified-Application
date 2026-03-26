"""
Self-correction module for the BiRAGAS Causality Framework.

Provides DAG validation, automatic correction of structural issues (cycles,
orphan nodes, low-confidence edges), and phase-output validation with full
correction logging.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import networkx as nx
except ImportError:
    nx = None

from .config import SelfCorrectionConfig

logger = logging.getLogger(__name__)


@dataclass
class CorrectionRecord:
    """A single correction action applied to the DAG or phase output."""
    phase: str
    module: str
    correction_type: str
    target: str
    action: str
    before_value: Any
    after_value: Any
    reason: str
    confidence: float = 0.0


class SelfCorrector:
    """Validates and auto-corrects DAGs and phase outputs.

    Args:
        config: A SelfCorrectionConfig controlling thresholds and toggles.
        disease_node: The expected disease/root node name in every DAG.
    """

    def __init__(
        self,
        config: Optional[SelfCorrectionConfig] = None,
        disease_node: str = "Disease_Activity",
    ):
        self.config = config or SelfCorrectionConfig()
        self.disease_node = disease_node
        self.corrections: List[CorrectionRecord] = []
        self.error_patterns: Dict[str, Dict[str, int]] = {}
        self._attempt_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # DAG validation
    # ------------------------------------------------------------------

    def validate_dag(
        self, dag: Any, phase: str = "unknown"
    ) -> Tuple[bool, List[str]]:
        """Validate a DAG for structural correctness.

        Checks performed:
            1. Non-empty graph
            2. Acyclicity (must be a DAG)
            3. Connectivity (weakly connected)
            4. Disease node presence
            5. Layer ordering (edges go from lower to higher layers when annotated)
            6. Edge confidence values within [0, 1]

        Args:
            dag: A NetworkX DiGraph (or compatible object).
            phase: The phase name for logging purposes.

        Returns:
            A tuple of (is_valid, list_of_issue_strings).
        """
        issues: List[str] = []

        if nx is None:
            issues.append("networkx not installed; cannot validate DAG")
            return False, issues

        # 1. Empty graph
        if dag is None or dag.number_of_nodes() == 0:
            issues.append("DAG is empty (no nodes)")
            return False, issues

        if dag.number_of_edges() == 0:
            issues.append("DAG has no edges")

        # 2. Acyclicity
        if not nx.is_directed_acyclic_graph(dag):
            cycles = list(nx.simple_cycles(dag))
            cycle_str = "; ".join(
                " -> ".join(c) for c in cycles[:5]
            )
            issues.append(f"DAG contains cycles: [{cycle_str}]")

        # 3. Connectivity
        if dag.number_of_nodes() > 1 and not nx.is_weakly_connected(dag):
            n_components = nx.number_weakly_connected_components(dag)
            issues.append(
                f"DAG is not weakly connected ({n_components} components)"
            )

        # 4. Disease node presence
        if self.disease_node and self.disease_node not in dag.nodes:
            issues.append(
                f"Expected disease node '{self.disease_node}' not found in DAG"
            )

        # 5. Layer ordering
        for u, v, data in dag.edges(data=True):
            u_layer = dag.nodes[u].get("layer") if u in dag.nodes else None
            v_layer = dag.nodes[v].get("layer") if v in dag.nodes else None
            if u_layer is not None and v_layer is not None:
                try:
                    if int(u_layer) > int(v_layer):
                        issues.append(
                            f"Edge {u} -> {v} violates layer ordering "
                            f"(layer {u_layer} -> layer {v_layer})"
                        )
                except (ValueError, TypeError):
                    pass

        # 6. Confidence values
        for u, v, data in dag.edges(data=True):
            conf = data.get("confidence")
            if conf is not None:
                try:
                    conf_f = float(conf)
                    if conf_f < 0.0 or conf_f > 1.0:
                        issues.append(
                            f"Edge {u} -> {v} has out-of-range confidence {conf_f}"
                        )
                except (ValueError, TypeError):
                    issues.append(
                        f"Edge {u} -> {v} has non-numeric confidence '{conf}'"
                    )

        is_valid = len(issues) == 0
        if issues:
            logger.warning(
                "DAG validation failed at phase '%s': %s", phase, "; ".join(issues)
            )
        return is_valid, issues

    # ------------------------------------------------------------------
    # Automatic DAG correction
    # ------------------------------------------------------------------

    def auto_correct_dag(
        self, dag: Any, phase: str = "unknown"
    ) -> Tuple[Any, List[CorrectionRecord]]:
        """Attempt to automatically fix common DAG issues.

        Corrections applied (when enabled in config):
            * Cycle removal by iteratively dropping the weakest edge in each cycle.
            * Removal of edges whose confidence is below the threshold.
            * Removal of orphan nodes (degree-0 after edge pruning).

        Args:
            dag: A NetworkX DiGraph.
            phase: Phase name for correction records.

        Returns:
            A tuple of (corrected_dag, list_of_CorrectionRecords).
        """
        corrections: List[CorrectionRecord] = []

        if nx is None:
            logger.error("networkx not installed; cannot auto-correct DAG")
            return dag, corrections

        if dag is None or dag.number_of_nodes() == 0:
            return dag, corrections

        attempt_key = f"{phase}_auto_correct"
        self._attempt_counts.setdefault(attempt_key, 0)
        if self._attempt_counts[attempt_key] >= self.config.max_correction_attempts:
            logger.warning(
                "Max correction attempts (%d) reached for phase '%s'",
                self.config.max_correction_attempts,
                phase,
            )
            return dag, corrections
        self._attempt_counts[attempt_key] += 1

        # --- Fix cycles ---
        if self.config.auto_fix_cycles:
            max_cycle_iters = dag.number_of_edges()  # safety bound
            iteration = 0
            while not nx.is_directed_acyclic_graph(dag) and iteration < max_cycle_iters:
                iteration += 1
                try:
                    cycle = nx.find_cycle(dag)
                except nx.NetworkXNoCycle:
                    break

                # Find the weakest edge in this cycle
                weakest_edge = None
                weakest_conf = float("inf")
                for u, v in cycle:
                    conf = dag[u][v].get("confidence", 0.5)
                    try:
                        conf_f = float(conf)
                    except (ValueError, TypeError):
                        conf_f = 0.5
                    if conf_f < weakest_conf:
                        weakest_conf = conf_f
                        weakest_edge = (u, v)

                if weakest_edge:
                    u, v = weakest_edge
                    rec = CorrectionRecord(
                        phase=phase,
                        module="self_corrector",
                        correction_type="cycle_removal",
                        target=f"{u} -> {v}",
                        action="remove_edge",
                        before_value=dict(dag[u][v]),
                        after_value=None,
                        reason=f"Removing weakest edge in cycle (confidence={weakest_conf:.3f})",
                        confidence=weakest_conf,
                    )
                    dag.remove_edge(u, v)
                    corrections.append(rec)
                    logger.info("Cycle fix: removed edge %s -> %s (conf=%.3f)", u, v, weakest_conf)

        # --- Fix low-confidence edges ---
        if self.config.auto_fix_low_confidence:
            edges_to_remove = []
            for u, v, data in dag.edges(data=True):
                conf = data.get("confidence")
                if conf is not None:
                    try:
                        conf_f = float(conf)
                    except (ValueError, TypeError):
                        continue
                    if conf_f < self.config.edge_removal_threshold:
                        edges_to_remove.append((u, v, conf_f, dict(data)))

            for u, v, conf_f, edge_data in edges_to_remove:
                rec = CorrectionRecord(
                    phase=phase,
                    module="self_corrector",
                    correction_type="low_confidence_removal",
                    target=f"{u} -> {v}",
                    action="remove_edge",
                    before_value=edge_data,
                    after_value=None,
                    reason=(
                        f"Edge confidence {conf_f:.3f} below threshold "
                        f"{self.config.edge_removal_threshold}"
                    ),
                    confidence=conf_f,
                )
                dag.remove_edge(u, v)
                corrections.append(rec)
                logger.info(
                    "Low-confidence fix: removed edge %s -> %s (conf=%.3f)", u, v, conf_f
                )

        # --- Fix orphan nodes ---
        if self.config.auto_fix_orphans:
            orphans = [
                n
                for n in list(dag.nodes)
                if dag.degree(n) == 0 and n != self.disease_node
            ]
            for node in orphans:
                rec = CorrectionRecord(
                    phase=phase,
                    module="self_corrector",
                    correction_type="orphan_removal",
                    target=node,
                    action="remove_node",
                    before_value=dict(dag.nodes[node]) if node in dag.nodes else None,
                    after_value=None,
                    reason="Orphan node (degree 0) after edge pruning",
                    confidence=0.0,
                )
                dag.remove_node(node)
                corrections.append(rec)
                logger.info("Orphan fix: removed node '%s'", node)

        # Store corrections globally
        if self.config.log_corrections:
            self.corrections.extend(corrections)

        return dag, corrections

    # ------------------------------------------------------------------
    # Phase output validation
    # ------------------------------------------------------------------

    def validate_phase_output(
        self, phase_name: str, result: Any, dag: Any = None
    ) -> Tuple[bool, List[str]]:
        """Validate the output of a specific phase.

        Each phase has bespoke checks in addition to generic ones.

        Args:
            phase_name: One of the recognised phase names.
            result: The output dictionary/object from that phase.
            dag: Optional current DAG for cross-referencing.

        Returns:
            A tuple of (is_valid, list_of_issue_strings).
        """
        issues: List[str] = []

        if result is None:
            issues.append(f"Phase '{phase_name}' returned None")
            return False, issues

        # Generic: result should be a dict (or at least have keys)
        if isinstance(result, dict) and not result:
            issues.append(f"Phase '{phase_name}' returned an empty dictionary")

        # Phase-specific validation
        phase_lower = phase_name.lower()

        if "knowledge" in phase_lower or "phase1" in phase_lower:
            issues.extend(self._validate_knowledge_phase(result))

        elif "generation" in phase_lower or "phase2" in phase_lower:
            issues.extend(self._validate_generation_phase(result))

        elif "causal" in phase_lower or "phase3" in phase_lower:
            issues.extend(self._validate_causal_phase(result, dag))

        elif "counterfactual" in phase_lower or "phase4" in phase_lower:
            issues.extend(self._validate_counterfactual_phase(result))

        elif "validation" in phase_lower or "phase5" in phase_lower:
            issues.extend(self._validate_validation_phase(result))

        elif "stress" in phase_lower or "phase6" in phase_lower:
            issues.extend(self._validate_stress_test_phase(result))

        elif "report" in phase_lower or "phase7" in phase_lower:
            issues.extend(self._validate_reporting_phase(result))

        is_valid = len(issues) == 0
        if not is_valid:
            logger.warning(
                "Phase output validation failed for '%s': %s",
                phase_name,
                "; ".join(issues),
            )
        return is_valid, issues

    # --- Private per-phase validators ---

    @staticmethod
    def _validate_knowledge_phase(result: Any) -> List[str]:
        issues = []
        if isinstance(result, dict):
            if "markers" not in result and "biomarkers" not in result:
                issues.append("Knowledge phase missing 'markers' or 'biomarkers' key")
            if "pathways" not in result and "causal_pathways" not in result:
                issues.append("Knowledge phase missing 'pathways' or 'causal_pathways' key")
        return issues

    @staticmethod
    def _validate_generation_phase(result: Any) -> List[str]:
        issues = []
        if isinstance(result, dict):
            if "dag" not in result and "graph" not in result:
                issues.append("Generation phase missing 'dag' or 'graph' key")
            if "nodes" not in result and "edges" not in result:
                if "dag" not in result:
                    issues.append("Generation phase missing structural graph data")
        return issues

    def _validate_causal_phase(self, result: Any, dag: Any) -> List[str]:
        issues = []
        if isinstance(result, dict):
            if "causal_effects" not in result and "effects" not in result:
                issues.append("Causal phase missing 'causal_effects' or 'effects' key")
        if dag is not None and self.config.dag_validity_check:
            _, dag_issues = self.validate_dag(dag, phase="phase3_causal")
            issues.extend(dag_issues)
        return issues

    @staticmethod
    def _validate_counterfactual_phase(result: Any) -> List[str]:
        issues = []
        if isinstance(result, dict):
            if "counterfactuals" not in result and "scenarios" not in result:
                issues.append(
                    "Counterfactual phase missing 'counterfactuals' or 'scenarios' key"
                )
        return issues

    @staticmethod
    def _validate_validation_phase(result: Any) -> List[str]:
        issues = []
        if isinstance(result, dict):
            if "validation_score" not in result and "score" not in result:
                issues.append("Validation phase missing score information")
        return issues

    @staticmethod
    def _validate_stress_test_phase(result: Any) -> List[str]:
        issues = []
        if isinstance(result, dict):
            if "scenarios_passed" not in result and "results" not in result:
                issues.append("Stress test phase missing results data")
        return issues

    @staticmethod
    def _validate_reporting_phase(result: Any) -> List[str]:
        issues = []
        if isinstance(result, dict):
            if "report" not in result and "summary" not in result:
                issues.append("Reporting phase missing 'report' or 'summary' key")
        return issues

    # ------------------------------------------------------------------
    # Correction summary and error tracking
    # ------------------------------------------------------------------

    def get_correction_summary(self) -> Dict[str, Any]:
        """Return an aggregate summary of all corrections applied.

        Returns:
            A dictionary with totals broken down by correction type and phase.
        """
        summary: Dict[str, Any] = {
            "total_corrections": len(self.corrections),
            "by_type": {},
            "by_phase": {},
            "records": [asdict(r) for r in self.corrections],
        }

        for rec in self.corrections:
            # By type
            summary["by_type"].setdefault(rec.correction_type, 0)
            summary["by_type"][rec.correction_type] += 1
            # By phase
            summary["by_phase"].setdefault(rec.phase, 0)
            summary["by_phase"][rec.phase] += 1

        return summary

    def record_error_pattern(self, phase: str, error_type: str) -> None:
        """Record an error pattern for trend analysis.

        Args:
            phase: The phase where the error occurred.
            error_type: A short descriptor of the error category.
        """
        self.error_patterns.setdefault(phase, {})
        self.error_patterns[phase].setdefault(error_type, 0)
        self.error_patterns[phase][error_type] += 1
        logger.debug(
            "Error pattern recorded: phase=%s, type=%s (count=%d)",
            phase,
            error_type,
            self.error_patterns[phase][error_type],
        )

    def reset(self) -> None:
        """Clear all accumulated corrections and error patterns."""
        self.corrections.clear()
        self.error_patterns.clear()
        self._attempt_counts.clear()
