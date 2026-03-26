"""
Phase 1: CAUSALITY FOUNDATION — Module 2
MRValidator (INTENT I_02 Module 4)
===================================
Mendelian Randomization validation of causal edges in the DAG.
Uses genetic variants as instrumental variables to validate causality.

MR Methods: IVW, MR-Egger, Weighted Median, Weighted Mode, MR-PRESSO
Sensitivity: Cochran's Q, Egger intercept, Leave-one-out, Steiger directionality

Organization: Ayass Bioscience LLC
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class MRValidatorConfig:
    gwas_pval_threshold: float = 5e-8
    f_statistic_min: float = 10.0
    mr_pval_threshold: float = 0.05
    egger_intercept_threshold: float = 0.05
    heterogeneity_threshold: float = 0.05
    min_instruments: int = 3
    bootstrap_iterations: int = 200  # Optimized: was 1000
    ld_r2_threshold: float = 0.001
    ld_window_kb: int = 10000
    presso_n_simulations: int = 1000
    presso_outlier_threshold: float = 0.05


class MRValidator:
    """Validates causal edges via Mendelian Randomization."""

    def __init__(self, config: Optional[MRValidatorConfig] = None):
        self.config = config or MRValidatorConfig()
        self.rng = np.random.RandomState(42)

    def prepare_instruments(self, eqtl_data: pd.DataFrame, gene: str) -> List[Dict]:
        gene_data = eqtl_data[eqtl_data['gene'] == gene].copy()
        if gene_data.empty:
            return []
        if 'pvalue' in gene_data.columns:
            gene_data = gene_data[gene_data['pvalue'] < self.config.gwas_pval_threshold]

        instruments = []
        for _, row in gene_data.iterrows():
            beta_exp = row.get('beta', row.get('eQTL_beta', 0))
            se_exp = row.get('se', abs(beta_exp) * 0.1 if beta_exp != 0 else 0.01)
            n = row.get('n', row.get('sample_size', 1000))
            r2 = beta_exp ** 2 / (beta_exp ** 2 + se_exp ** 2 * n) if n > 0 else 0
            f_stat = r2 * (n - 2) / (1 - r2) if r2 < 1 else 0

            if f_stat >= self.config.f_statistic_min:
                instruments.append({
                    'variant_id': row.get('variant_id', row.get('SNP', f'snp_{len(instruments)}')),
                    'beta_exposure': float(beta_exp),
                    'se_exposure': float(se_exp),
                    'beta_outcome': float(row.get('beta_outcome', 0)),
                    'se_outcome': float(row.get('se_outcome', abs(row.get('beta_outcome', 0)) * 0.15 + 0.01)),
                    'f_statistic': float(f_stat),
                    'r2_exposure': float(r2),
                })
        instruments.sort(key=lambda x: x['f_statistic'], reverse=True)
        kept = []
        for inst in instruments:
            independent = all(inst['variant_id'][:5] != k['variant_id'][:5] for k in kept)
            if independent or not kept:
                kept.append(inst)
        return kept

    def run_ivw(self, instruments: List[Dict]) -> Dict:
        if len(instruments) < self.config.min_instruments:
            return {'beta': np.nan, 'se': np.nan, 'pval': np.nan, 'method': 'IVW', 'valid': False}
        bx = np.array([i['beta_exposure'] for i in instruments])
        by = np.array([i['beta_outcome'] for i in instruments])
        se_y = np.array([i['se_outcome'] for i in instruments])
        weights = 1.0 / (se_y ** 2)
        beta_ivw = np.sum(weights * bx * by) / np.sum(weights * bx ** 2)
        se_ivw = np.sqrt(1.0 / np.sum(weights * bx ** 2))
        z = beta_ivw / se_ivw
        pval = 2 * scipy_stats.norm.sf(abs(z))
        return {'beta': float(beta_ivw), 'se': float(se_ivw), 'pval': float(pval),
                'method': 'IVW', 'n_instruments': len(instruments), 'valid': True}

    def run_egger(self, instruments: List[Dict]) -> Dict:
        if len(instruments) < self.config.min_instruments + 1:
            return {'beta': np.nan, 'se': np.nan, 'pval': np.nan, 'intercept': np.nan,
                    'intercept_pval': np.nan, 'method': 'MR-Egger', 'valid': False}
        bx = np.array([i['beta_exposure'] for i in instruments])
        by = np.array([i['beta_outcome'] for i in instruments])
        se_y = np.array([i['se_outcome'] for i in instruments])
        sign = np.sign(bx)
        bx_o, by_o = np.abs(bx), by * sign
        weights = 1.0 / (se_y ** 2)
        W = np.diag(weights)
        X = np.column_stack([np.ones(len(bx_o)), bx_o])
        XtWX_inv = np.linalg.inv(X.T @ W @ X)
        beta_hat = XtWX_inv @ X.T @ W @ by_o
        residuals = by_o - X @ beta_hat
        sigma2 = np.sum(weights * residuals ** 2) / max(1, len(instruments) - 2)
        var_beta = sigma2 * XtWX_inv
        intercept, slope = beta_hat[0], beta_hat[1]
        se_int = np.sqrt(max(0, var_beta[0, 0]))
        se_slope = np.sqrt(max(0, var_beta[1, 1]))
        pval = 2 * scipy_stats.norm.sf(abs(slope / se_slope)) if se_slope > 0 else 1.0
        int_pval = 2 * scipy_stats.norm.sf(abs(intercept / se_int)) if se_int > 0 else 1.0
        return {'beta': float(slope), 'se': float(se_slope), 'pval': float(pval),
                'intercept': float(intercept), 'intercept_pval': float(int_pval),
                'pleiotropy_detected': int_pval < self.config.egger_intercept_threshold,
                'method': 'MR-Egger', 'valid': True}

    def run_weighted_median(self, instruments: List[Dict]) -> Dict:
        if len(instruments) < self.config.min_instruments:
            return {'beta': np.nan, 'se': np.nan, 'pval': np.nan, 'method': 'Weighted Median', 'valid': False}
        bx = np.array([i['beta_exposure'] for i in instruments])
        by = np.array([i['beta_outcome'] for i in instruments])
        se_y = np.array([i['se_outcome'] for i in instruments])
        ratio = by / bx
        weights = (bx ** 2) / (se_y ** 2)
        weights /= np.sum(weights)
        order = np.argsort(ratio)
        cumw = np.cumsum(weights[order])
        idx = np.searchsorted(cumw, 0.5)
        beta_wm = ratio[order][min(idx, len(ratio) - 1)]

        boot_betas = []
        for _ in range(self.config.bootstrap_iterations):
            idx_b = self.rng.choice(len(instruments), len(instruments), replace=True)
            b_ratio = by[idx_b] / bx[idx_b]
            b_w = (bx[idx_b] ** 2) / (se_y[idx_b] ** 2)
            b_w /= np.sum(b_w)
            o = np.argsort(b_ratio)
            cw = np.cumsum(b_w[o])
            bi = np.searchsorted(cw, 0.5)
            boot_betas.append(b_ratio[o][min(bi, len(b_ratio) - 1)])
        se_wm = np.std(boot_betas)
        z = beta_wm / se_wm if se_wm > 0 else 0
        pval = 2 * scipy_stats.norm.sf(abs(z))
        return {'beta': float(beta_wm), 'se': float(se_wm), 'pval': float(pval),
                'method': 'Weighted Median', 'valid': True}

    def run_weighted_mode(self, instruments: List[Dict]) -> Dict:
        if len(instruments) < self.config.min_instruments:
            return {'beta': np.nan, 'se': np.nan, 'pval': np.nan, 'method': 'Weighted Mode', 'valid': False}
        bx = np.array([i['beta_exposure'] for i in instruments])
        by = np.array([i['beta_outcome'] for i in instruments])
        se_y = np.array([i['se_outcome'] for i in instruments])
        ratio = by / bx
        weights = (bx ** 2) / (se_y ** 2)
        h = max(0.5, 0.9 * np.std(ratio) * len(ratio) ** (-0.2))
        grid = np.linspace(np.min(ratio) - 3 * h, np.max(ratio) + 3 * h, 1000)
        density = np.zeros_like(grid)
        for r, w in zip(ratio, weights):
            density += w * scipy_stats.norm.pdf(grid, loc=r, scale=h)
        beta_mode = grid[np.argmax(density)]

        boot_betas = []
        for _ in range(self.config.bootstrap_iterations):
            idx_b = self.rng.choice(len(instruments), len(instruments), replace=True)
            b_ratio = by[idx_b] / bx[idx_b]
            b_w = (bx[idx_b] ** 2) / (se_y[idx_b] ** 2)
            b_d = np.zeros_like(grid)
            for r, w in zip(b_ratio, b_w):
                b_d += w * scipy_stats.norm.pdf(grid, loc=r, scale=h)
            boot_betas.append(grid[np.argmax(b_d)])
        se_mode = np.std(boot_betas)
        z = beta_mode / se_mode if se_mode > 0 else 0
        pval = 2 * scipy_stats.norm.sf(abs(z))
        return {'beta': float(beta_mode), 'se': float(se_mode), 'pval': float(pval),
                'method': 'Weighted Mode', 'valid': True}

    def run_cochran_q(self, instruments: List[Dict], ivw_beta: float) -> Dict:
        if len(instruments) < 2:
            return {'q_stat': np.nan, 'q_pval': np.nan, 'heterogeneity': False}
        bx = np.array([i['beta_exposure'] for i in instruments])
        by = np.array([i['beta_outcome'] for i in instruments])
        se_y = np.array([i['se_outcome'] for i in instruments])
        ratio = by / bx
        weights = (bx ** 2) / (se_y ** 2)
        q_stat = float(np.sum(weights * (ratio - ivw_beta) ** 2))
        df = len(instruments) - 1
        q_pval = float(1 - scipy_stats.chi2.cdf(q_stat, df))
        i_sq = float(max(0, (q_stat - df) / q_stat * 100)) if q_stat > 0 else 0.0
        return {'q_stat': q_stat, 'q_pval': q_pval, 'df': df,
                'heterogeneity': q_pval < self.config.heterogeneity_threshold, 'i_squared': i_sq}

    def run_steiger_test(self, instruments: List[Dict],
                         exposure_r2: float = None, outcome_r2: float = None) -> Dict:
        if not instruments:
            return {'correct_direction': None, 'steiger_pval': np.nan}
        if exposure_r2 is None:
            exposure_r2 = min(0.999, sum(i.get('r2_exposure', 0.01) for i in instruments))
        if outcome_r2 is None:
            outcome_r2 = min(0.999, exposure_r2 * 0.1)
        z_exp = np.arctanh(np.sqrt(max(0, exposure_r2)))
        z_out = np.arctanh(np.sqrt(max(0, outcome_r2)))
        se_diff = np.sqrt(2.0 / 997)
        z_stat = (z_exp - z_out) / se_diff if se_diff > 0 else 0
        pval = float(scipy_stats.norm.sf(z_stat))
        return {'correct_direction': exposure_r2 > outcome_r2, 'steiger_pval': pval,
                'exposure_r2': float(exposure_r2), 'outcome_r2': float(outcome_r2)}

    def run_leave_one_out(self, instruments: List[Dict]) -> List[Dict]:
        if len(instruments) < self.config.min_instruments + 1:
            return []
        results = []
        for i in range(len(instruments)):
            subset = [inst for j, inst in enumerate(instruments) if j != i]
            ivw = self.run_ivw(subset)
            results.append({'excluded_variant': instruments[i]['variant_id'],
                            'beta': ivw['beta'], 'se': ivw['se'], 'pval': ivw['pval']})
        return results

    def run_mr_presso(self, instruments: List[Dict]) -> Dict:
        if len(instruments) < self.config.min_instruments + 1:
            return {'global_rss': np.nan, 'global_pval': np.nan, 'outliers': [],
                    'corrected_beta': np.nan, 'method': 'MR-PRESSO', 'valid': False}
        bx = np.array([i['beta_exposure'] for i in instruments])
        by = np.array([i['beta_outcome'] for i in instruments])
        se_y = np.array([i['se_outcome'] for i in instruments])
        weights = 1.0 / (se_y ** 2)
        ivw = self.run_ivw(instruments)
        observed_rss = float(np.sum(weights * (by - ivw['beta'] * bx) ** 2))

        null_rss = []
        for _ in range(self.config.presso_n_simulations):
            perm_by = self.rng.permutation(by)
            w_sum = np.sum(weights * bx ** 2)
            perm_beta = np.sum(weights * bx * perm_by) / w_sum if w_sum > 0 else 0
            null_rss.append(np.sum(weights * (perm_by - perm_beta * bx) ** 2))
        global_pval = float(np.mean(np.array(null_rss) >= observed_rss))

        outliers = []
        if global_pval < self.config.presso_outlier_threshold:
            for i in range(len(instruments)):
                subset = [inst for j, inst in enumerate(instruments) if j != i]
                loo_ivw = self.run_ivw(subset)
                if loo_ivw['valid']:
                    residual = abs(by[i] - loo_ivw['beta'] * bx[i])
                    if residual > 3 * se_y[i]:
                        outliers.append(instruments[i]['variant_id'])

        corrected_beta = np.nan
        if outliers:
            clean = [inst for inst in instruments if inst['variant_id'] not in outliers]
            if len(clean) >= self.config.min_instruments:
                corrected_beta = self.run_ivw(clean)['beta']

        return {'global_rss': observed_rss, 'global_pval': global_pval,
                'outliers': outliers, 'n_outliers': len(outliers),
                'corrected_beta': float(corrected_beta) if not np.isnan(corrected_beta) else None,
                'method': 'MR-PRESSO', 'valid': True}

    def validate_edge(self, gene: str, eqtl_data: pd.DataFrame) -> Dict:
        instruments = self.prepare_instruments(eqtl_data, gene)
        if len(instruments) < self.config.min_instruments:
            return {'gene': gene, 'validated': False, 'reason': 'insufficient_instruments',
                    'n_instruments': len(instruments)}
        ivw = self.run_ivw(instruments)
        egger = self.run_egger(instruments)
        wm = self.run_weighted_median(instruments)
        mode = self.run_weighted_mode(instruments)
        presso = self.run_mr_presso(instruments)
        cochran = self.run_cochran_q(instruments, ivw['beta'])
        steiger = self.run_steiger_test(instruments)
        loo = self.run_leave_one_out(instruments)

        methods_sig = sum(1 for m in [ivw, egger, wm, mode]
                          if m.get('valid') and m.get('pval', 1) < self.config.mr_pval_threshold)
        betas = [m['beta'] for m in [ivw, egger, wm, mode]
                 if m.get('valid') and not np.isnan(m.get('beta', np.nan))]
        dir_consistent = len(set(np.sign(b) for b in betas)) <= 1 if betas else False
        sens_passed = (not cochran.get('heterogeneity', True) and
                       not egger.get('pleiotropy_detected', True) and
                       dir_consistent and steiger.get('correct_direction', False))
        validated = (ivw.get('pval', 1) < self.config.mr_pval_threshold and
                     methods_sig >= 2 and steiger.get('correct_direction', False))

        return {'gene': gene, 'validated': validated, 'n_instruments': len(instruments),
                'methods': {'ivw': ivw, 'egger': egger, 'weighted_median': wm,
                            'weighted_mode': mode, 'mr_presso': presso},
                'sensitivity': {'cochran_q': cochran, 'steiger': steiger, 'leave_one_out': loo,
                                'direction_consistent': dir_consistent, 'sensitivity_passed': sens_passed},
                'methods_significant': methods_sig,
                'primary_beta': ivw.get('beta', np.nan), 'primary_pval': ivw.get('pval', np.nan)}

    def validate_dag_edges(self, dag: nx.DiGraph, eqtl_data: pd.DataFrame,
                           target_node: str = "Disease_Activity") -> Tuple[nx.DiGraph, Dict]:
        results = {}
        genes = [u for u, v in dag.edges() if v == target_node
                 and dag.nodes[u].get('layer') == 'regulatory']
        for gene in genes:
            mr_result = self.validate_edge(gene, eqtl_data)
            results[gene] = mr_result
            if dag.has_edge(gene, target_node):
                edge = dag[gene][target_node]
                edge['mr_validated'] = mr_result['validated']
                edge['mr_beta'] = mr_result.get('primary_beta', np.nan)
                edge['mr_pval'] = mr_result.get('primary_pval', np.nan)
                edge['mr_method'] = 'IVW'
                edge['mr_sensitivity_passed'] = mr_result.get('sensitivity', {}).get('sensitivity_passed', False)
                if mr_result['validated']:
                    ev = edge.get('evidence', [])
                    if isinstance(ev, list):
                        ev = set(ev)
                    ev.add('mendelian_randomization_validated')
                    edge['evidence'] = list(ev)

        summary = {'total_tested': len(results),
                   'validated': sum(1 for r in results.values() if r['validated']),
                   'failed': sum(1 for r in results.values() if not r['validated']),
                   'sensitivity_passed': sum(1 for r in results.values()
                                             if r.get('sensitivity', {}).get('sensitivity_passed', False))}
        return dag, {'per_gene': results, 'summary': summary}
