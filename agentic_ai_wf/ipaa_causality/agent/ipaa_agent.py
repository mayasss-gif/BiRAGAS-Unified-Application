"""
Production-ready IPAA pipeline orchestrator.
Thread-safe, Celery-safe, LangGraph-safe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from ..config.models import (
    IPAAConfig,
    IPAAPhaseConfig,
    CausalityPhaseConfig,
    AggregationPhaseConfig,
    ReportPhaseConfig,
    IPAAResult,
)


class IPAAAgent:
    """
    Orchestrates IPAA → Causality → Aggregation → Reports.
    No subprocess, no argparse. Pure Python service calls.
    """

    def __init__(self, config: Union[IPAAConfig, dict]):
        if isinstance(config, dict):
            config = IPAAConfig.model_validate(config)
        self.config = config
        self._out_root = Path(config.outdir).resolve()

    def run_full(self) -> IPAAResult:
        """Execute full pipeline: IPAA → Causality → Aggregation → Reports."""
        from ..services import ipaa_service, causality_service, aggregation_service, report_service

        result = IPAAResult(out_root=str(self._out_root))
        phase_results = []

        if not self.config.skip_ipaa:
            r = ipaa_service.run_ipaa(self._ipaa_config())
            result.ipaa = r
            phase_results.append(r)

        if not self.config.skip_causality:
            r = causality_service.run_causality(self._causality_config())
            result.causality = r
            phase_results.append(r)

        if not self.config.skip_aggregator:
            r = aggregation_service.run_aggregation(self._aggregation_config())
            result.aggregation = r
            phase_results.append(r)

        if not self.config.skip_html_report:
            r = report_service.run_reports(self._report_config())
            result.reports = r
            phase_results.append(r)

        failed = [p for p in phase_results if p and not getattr(p, "success", True)]
        result.status = "ok" if not failed else ("partial" if phase_results else "failed")
        if failed:
            for p in failed:
                result.errors.extend(getattr(p, "errors", []) or [])

        return result

    def run_ipaa(self):
        from ..services import ipaa_service
        return ipaa_service.run_ipaa(self._ipaa_config())

    def run_causality(self):
        from ..services import causality_service
        return causality_service.run_causality(self._causality_config())

    def run_aggregation(self):
        from ..services import aggregation_service
        return aggregation_service.run_aggregation(self._aggregation_config())

    def run_reports(self):
        from ..services import report_service
        return report_service.run_reports(self._report_config())

    def _ipaa_config(self) -> IPAAPhaseConfig:
        c = self.config
        return IPAAPhaseConfig(
            outdir=c.outdir,
            items=c.items,
            spec=c.spec,
            counts_default=c.counts_default,
            verbose=c.verbose,
            refresh_omnipath=c.refresh_omnipath,
            workers=c.workers,
            threads_per_cohort=c.threads_per_cohort,
            report_top=c.report_top,
            threads=c.threads,
            gsea_permutations=c.gsea_permutations,
            sig_fdr=c.sig_fdr,
            sig_top_n=c.sig_top_n,
            msigdb_dbver=c.msigdb_dbver,
            run_baseline=c.run_baseline,
            baseline_dir=c.baseline_dir or "",
            auto_select_tissue=c.auto_select_tissue,
            tissue_top_k=c.tissue_top_k,
            skip_engine1=c.skip_engine1,
            engine1_strict=c.engine1_strict,
            engine1_license_mode=c.engine1_license_mode,
            engine1_tf_method=c.engine1_tf_method,
            engine1_tmin=c.engine1_tmin,
            engine1_no_overwrite=c.engine1_no_overwrite,
            engine1_no_regulators_evidence=c.engine1_no_regulators_evidence,
        )

    def _causality_config(self) -> CausalityPhaseConfig:
        c = self.config
        return CausalityPhaseConfig(
            out_root=c.outdir,
            diseases=c.diseases,
            run_all=c.run_all,
            strict=c.strict,
            log_level=c.log_level,
            engine23_script=c.engine23_script,
            no_engine0=c.no_engine0,
            no_engine1=c.no_engine1,
            no_engine2=c.no_engine2,
            no_engine3=c.no_engine3,
            no_omnipath_layer=c.no_omnipath_layer,
            no_refresh_omnipath_cache=c.no_refresh_omnipath_cache,
            no_build_pkn=c.no_build_pkn,
            no_refresh_pkn=c.no_refresh_pkn,
            no_force_engine1=c.no_force_engine1,
            no_force_engine0=c.no_force_engine0,
            signor_edges=c.signor_edges,
            ptm_min_substrates=c.ptm_min_substrates,
            ptm_n_perm=c.ptm_n_perm,
            engine1_method=c.engine1_method,
            engine1_min_size=c.engine1_min_size,
            engine1_max_pathways=c.engine1_max_pathways,
            engine1_max_tfs=c.engine1_max_tfs,
            corr_method=c.corr_method,
            corr_flag_threshold=c.corr_flag_threshold,
            min_markers=c.min_markers,
            pkn_edges=c.pkn_edges,
            max_steps=c.max_steps,
            top_tfs=c.top_tfs,
            confound_penalty_threshold=c.confound_penalty_threshold,
        )

    def _aggregation_config(self) -> AggregationPhaseConfig:
        c = self.config
        diseases = c.aggregator_diseases
        if isinstance(diseases, str) and diseases.strip():
            diseases = [d.strip() for d in diseases.split(",") if d.strip()]
        elif not diseases and c.diseases:
            diseases = c.diseases
        return AggregationPhaseConfig(
            out_root=c.outdir,
            diseases=diseases,
            out_subdir=c.aggregator_out_subdir,
            filter_fdr=0.05,
        )

    def _report_config(self) -> ReportPhaseConfig:
        c = self.config
        return ReportPhaseConfig(
            outdir=c.outdir,
            topn=c.report_topn,
            fdr_cutoff=c.report_fdr_cutoff,
            llm_selector=c.report_llm_selector,
            llm_narrative=c.report_llm_narrative,
            llm_model=c.report_llm_model,
            api_key_env=c.report_api_key_env,
            generate_pdf=c.generate_pdf_report,
        )
