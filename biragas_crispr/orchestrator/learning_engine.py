"""
Learning engine for the BiRAGAS Causality Framework.

Tracks per-run performance metrics, persists history to JSON, and generates
data-driven recommendations for improving future runs.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import LearningConfig

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    """Comprehensive metrics collected during a single orchestrator run."""
    run_id: str = ""
    disease: str = ""
    timestamp: str = ""

    # Phase durations (phase_name -> seconds)
    durations: Dict[str, float] = field(default_factory=dict)

    # DAG statistics
    dag_nodes: int = 0
    dag_edges: int = 0
    dag_density: float = 0.0
    dag_max_depth: int = 0

    # Tier-1 biomarker count
    tier1_count: int = 0

    # Overall confidence
    confidence_mean: float = 0.0
    confidence_min: float = 0.0
    confidence_max: float = 0.0

    # Hallucination tracking
    hallucinations_detected: int = 0
    hallucination_rate: float = 0.0

    # Self-correction tracking
    corrections_applied: int = 0
    correction_types: Dict[str, int] = field(default_factory=dict)

    # Stress test summary
    stress_test_passed: bool = False
    stress_test_scenarios_passed: int = 0
    stress_test_scenarios_total: int = 0

    # Total wall-clock time
    total_duration: float = 0.0


class LearningEngine:
    """Tracks orchestrator performance across runs and produces recommendations.

    Args:
        config: A LearningConfig instance controlling persistence and adaptation.
    """

    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        self.current_metrics: Optional[RunMetrics] = None
        self.history: List[Dict[str, Any]] = []
        self._phase_start_times: Dict[str, float] = {}
        self._run_start_time: Optional[float] = None

        if self.config.store_history:
            self._load_history()

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, run_id: str, disease: str = "") -> None:
        """Begin tracking a new orchestrator run.

        Args:
            run_id: Unique identifier for this run.
            disease: The disease being analyzed.
        """
        self.current_metrics = RunMetrics(
            run_id=run_id,
            disease=disease,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._run_start_time = time.time()
        self._phase_start_times.clear()
        logger.info("LearningEngine: started run '%s' for disease '%s'", run_id, disease)

    def start_phase(self, phase_name: str) -> None:
        """Record the start time of a phase.

        Args:
            phase_name: The name of the phase being started.
        """
        self._phase_start_times[phase_name] = time.time()
        logger.debug("LearningEngine: phase '%s' started", phase_name)

    def end_phase(self, phase_name: str, success: bool = True) -> float:
        """Record the end time of a phase and compute its duration.

        Args:
            phase_name: The name of the phase that just completed.
            success: Whether the phase completed successfully.

        Returns:
            The elapsed duration in seconds, or 0.0 if start was never recorded.
        """
        start = self._phase_start_times.pop(phase_name, None)
        if start is None:
            logger.warning("LearningEngine: end_phase called for '%s' without start", phase_name)
            return 0.0

        duration = time.time() - start
        if self.current_metrics is not None:
            self.current_metrics.durations[phase_name] = round(duration, 3)
        logger.debug("LearningEngine: phase '%s' completed in %.3fs", phase_name, duration)
        return duration

    def record_dag_metrics(
        self,
        num_nodes: int = 0,
        num_edges: int = 0,
        density: float = 0.0,
        max_depth: int = 0,
        tier1_count: int = 0,
        confidence_mean: float = 0.0,
        confidence_min: float = 0.0,
        confidence_max: float = 0.0,
    ) -> None:
        """Record DAG-level statistics for the current run.

        Args:
            num_nodes: Number of nodes in the DAG.
            num_edges: Number of edges in the DAG.
            density: Graph density.
            max_depth: Longest path length in the DAG.
            tier1_count: Number of tier-1 biomarkers.
            confidence_mean: Mean edge confidence.
            confidence_min: Minimum edge confidence.
            confidence_max: Maximum edge confidence.
        """
        if self.current_metrics is None:
            return
        self.current_metrics.dag_nodes = num_nodes
        self.current_metrics.dag_edges = num_edges
        self.current_metrics.dag_density = round(density, 4)
        self.current_metrics.dag_max_depth = max_depth
        self.current_metrics.tier1_count = tier1_count
        self.current_metrics.confidence_mean = round(confidence_mean, 4)
        self.current_metrics.confidence_min = round(confidence_min, 4)
        self.current_metrics.confidence_max = round(confidence_max, 4)

    def record_corrections(
        self, corrections_applied: int, correction_types: Optional[Dict[str, int]] = None
    ) -> None:
        """Record self-correction statistics.

        Args:
            corrections_applied: Total number of corrections in this run.
            correction_types: Breakdown by correction type.
        """
        if self.current_metrics is None:
            return
        self.current_metrics.corrections_applied = corrections_applied
        if correction_types:
            self.current_metrics.correction_types = dict(correction_types)

    def record_hallucinations(self, detected: int, rate: float = 0.0) -> None:
        """Record hallucination detection statistics.

        Args:
            detected: Number of hallucinations detected.
            rate: Hallucination rate as a fraction.
        """
        if self.current_metrics is None:
            return
        self.current_metrics.hallucinations_detected = detected
        self.current_metrics.hallucination_rate = round(rate, 4)

    def record_stress_test(
        self, passed: bool, scenarios_passed: int = 0, scenarios_total: int = 0
    ) -> None:
        """Record stress test outcome.

        Args:
            passed: Whether the overall stress test passed.
            scenarios_passed: Number of individual scenarios that passed.
            scenarios_total: Total number of scenarios executed.
        """
        if self.current_metrics is None:
            return
        self.current_metrics.stress_test_passed = passed
        self.current_metrics.stress_test_scenarios_passed = scenarios_passed
        self.current_metrics.stress_test_scenarios_total = scenarios_total

    def end_run(self) -> RunMetrics:
        """Finalize the current run and persist metrics to history.

        Returns:
            The completed RunMetrics for this run.
        """
        if self.current_metrics is None:
            logger.warning("LearningEngine: end_run called with no active run")
            return RunMetrics()

        if self._run_start_time is not None:
            self.current_metrics.total_duration = round(
                time.time() - self._run_start_time, 3
            )

        metrics = self.current_metrics
        self.history.append(asdict(metrics))

        if self.config.store_history:
            self._save_history()

        logger.info(
            "LearningEngine: run '%s' completed in %.1fs (%d corrections, %d hallucinations)",
            metrics.run_id,
            metrics.total_duration,
            metrics.corrections_applied,
            metrics.hallucinations_detected,
        )

        self.current_metrics = None
        self._run_start_time = None
        self._phase_start_times.clear()
        return metrics

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def get_recommendations(self) -> List[str]:
        """Analyze run history and produce actionable recommendations.

        Looks for:
            * Phase bottlenecks (phases that take disproportionately long).
            * High correction rates suggesting upstream quality issues.
            * Low mean confidence suggesting weak evidence.
            * High hallucination rates.
            * Stress test failure trends.

        Returns:
            A list of human-readable recommendation strings.
        """
        recommendations: List[str] = []

        if not self.history:
            recommendations.append(
                "No run history available yet. Complete at least one run to receive recommendations."
            )
            return recommendations

        recent = self.history[-min(len(self.history), self.config.min_samples_for_adaptation):]

        # --- Bottleneck detection ---
        phase_totals: Dict[str, List[float]] = {}
        for run in recent:
            for phase, dur in run.get("durations", {}).items():
                phase_totals.setdefault(phase, []).append(dur)

        if phase_totals:
            avg_durations = {
                p: sum(ds) / len(ds) for p, ds in phase_totals.items()
            }
            total_avg = sum(avg_durations.values())
            if total_avg > 0:
                for phase, avg_dur in sorted(avg_durations.items(), key=lambda x: -x[1]):
                    fraction = avg_dur / total_avg
                    if fraction > 0.40:
                        recommendations.append(
                            f"Bottleneck: Phase '{phase}' consumes {fraction:.0%} of total "
                            f"run time (avg {avg_dur:.1f}s). Consider optimizing or parallelizing "
                            f"its modules."
                        )

        # --- High correction rate ---
        correction_counts = [r.get("corrections_applied", 0) for r in recent]
        avg_corrections = sum(correction_counts) / len(correction_counts) if correction_counts else 0
        if avg_corrections > 5:
            recommendations.append(
                f"High correction rate: averaging {avg_corrections:.1f} corrections per run. "
                f"Review upstream phase quality (knowledge retrieval, DAG generation) to "
                f"reduce the need for self-correction."
            )

        # --- Low confidence ---
        conf_means = [r.get("confidence_mean", 0) for r in recent if r.get("confidence_mean", 0) > 0]
        if conf_means:
            overall_mean_conf = sum(conf_means) / len(conf_means)
            if overall_mean_conf < 0.50:
                recommendations.append(
                    f"Low average confidence ({overall_mean_conf:.2f}). Consider enriching "
                    f"the knowledge base, using more specific biomarker evidence, or tuning "
                    f"confidence estimation parameters."
                )

        # --- Hallucination rate ---
        hall_rates = [r.get("hallucination_rate", 0) for r in recent if r.get("hallucination_rate", 0) > 0]
        if hall_rates:
            avg_hall_rate = sum(hall_rates) / len(hall_rates)
            if avg_hall_rate > 0.10:
                recommendations.append(
                    f"Elevated hallucination rate ({avg_hall_rate:.1%}). Consider adding "
                    f"stricter grounding checks or using more authoritative sources."
                )

        # --- Stress test failures ---
        stress_results = [r for r in recent if r.get("stress_test_scenarios_total", 0) > 0]
        if stress_results:
            pass_rates = [
                r["stress_test_scenarios_passed"] / r["stress_test_scenarios_total"]
                for r in stress_results
                if r["stress_test_scenarios_total"] > 0
            ]
            if pass_rates:
                avg_pass_rate = sum(pass_rates) / len(pass_rates)
                if avg_pass_rate < 0.80:
                    recommendations.append(
                        f"Stress test pass rate is low ({avg_pass_rate:.0%}). Review failing "
                        f"scenarios to identify systematic DAG separation or topology issues."
                    )

        if not recommendations:
            recommendations.append(
                "All metrics are within acceptable ranges. No immediate improvements needed."
            )

        return recommendations

    # ------------------------------------------------------------------
    # Performance summary
    # ------------------------------------------------------------------

    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate an aggregate performance summary across all recorded runs.

        Returns:
            A dictionary with run counts, averages, and trend information.
        """
        summary: Dict[str, Any] = {
            "total_runs": len(self.history),
            "avg_duration": 0.0,
            "avg_corrections": 0.0,
            "avg_hallucinations": 0.0,
            "avg_confidence": 0.0,
            "avg_dag_nodes": 0.0,
            "avg_dag_edges": 0.0,
            "stress_test_pass_rate": 0.0,
            "phase_avg_durations": {},
            "recent_runs": [],
        }

        if not self.history:
            return summary

        n = len(self.history)

        summary["avg_duration"] = round(
            sum(r.get("total_duration", 0) for r in self.history) / n, 2
        )
        summary["avg_corrections"] = round(
            sum(r.get("corrections_applied", 0) for r in self.history) / n, 2
        )
        summary["avg_hallucinations"] = round(
            sum(r.get("hallucinations_detected", 0) for r in self.history) / n, 2
        )

        conf_values = [r.get("confidence_mean", 0) for r in self.history if r.get("confidence_mean", 0) > 0]
        if conf_values:
            summary["avg_confidence"] = round(sum(conf_values) / len(conf_values), 4)

        summary["avg_dag_nodes"] = round(
            sum(r.get("dag_nodes", 0) for r in self.history) / n, 1
        )
        summary["avg_dag_edges"] = round(
            sum(r.get("dag_edges", 0) for r in self.history) / n, 1
        )

        # Stress test pass rate
        stress_runs = [
            r for r in self.history if r.get("stress_test_scenarios_total", 0) > 0
        ]
        if stress_runs:
            total_passed = sum(r["stress_test_scenarios_passed"] for r in stress_runs)
            total_scenarios = sum(r["stress_test_scenarios_total"] for r in stress_runs)
            if total_scenarios > 0:
                summary["stress_test_pass_rate"] = round(total_passed / total_scenarios, 4)

        # Per-phase average durations
        phase_sums: Dict[str, List[float]] = {}
        for run in self.history:
            for phase, dur in run.get("durations", {}).items():
                phase_sums.setdefault(phase, []).append(dur)
        summary["phase_avg_durations"] = {
            p: round(sum(ds) / len(ds), 3) for p, ds in phase_sums.items()
        }

        # Last 5 runs as summary snapshots
        summary["recent_runs"] = [
            {
                "run_id": r.get("run_id", ""),
                "disease": r.get("disease", ""),
                "duration": r.get("total_duration", 0),
                "corrections": r.get("corrections_applied", 0),
                "confidence": r.get("confidence_mean", 0),
                "stress_passed": r.get("stress_test_passed", False),
            }
            for r in self.history[-5:]
        ]

        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_history(self) -> None:
        """Persist the run history to the configured JSON file."""
        if not self.config.store_history:
            return

        path = Path(self.config.history_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"runs": self.history, "updated": datetime.now(timezone.utc).isoformat()},
                    f,
                    indent=2,
                    default=str,
                )
            logger.debug("LearningEngine: history saved to %s (%d runs)", path, len(self.history))
        except OSError as exc:
            logger.error("LearningEngine: failed to save history to %s: %s", path, exc)

    def _load_history(self) -> None:
        """Load run history from the configured JSON file if it exists."""
        path = Path(self.config.history_file)
        if not path.exists():
            logger.debug("LearningEngine: no history file at %s; starting fresh", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.history = data.get("runs", [])
            logger.info(
                "LearningEngine: loaded %d historical runs from %s", len(self.history), path
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("LearningEngine: failed to load history from %s: %s", path, exc)
            self.history = []
