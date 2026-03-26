"""
Phase 1: CAUSALITY FOUNDATION — Module 1
DAGBuilder (INTENT I_01 Module 1)
==================================
Constructs a consensus Directed Acyclic Graph from multi-modal causal data.

Architecture (mirrors production pipeline):
  1. Multi-modal data ingestion: GWAS, MR, eQTL, CRISPR, SIGNOR, temporal, transcriptomic, pathway, deconvolution
  2. Strict topological layering: Source(SNP) → Regulatory(Gene) → Program(Pathway) → Trait(Disease)
  3. Patient-level DAG instantiation with z-score filtering
  4. Consensus aggregation with frequency threshold
  5. Advanced constraint arbitration: genetic anchors, temporal, perturbation, cellular veto, SIGNOR

Evidence Weights:
  GWAS=0.90, MR=0.95, CRISPR=0.85, SIGNOR=0.90, TEMPORAL=0.65,
  STATISTICAL=0.35, DATABASE=0.80, EQTL=0.85

Organization: Ayass Bioscience LLC
Platform: BiRAGAS
"""

import logging
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format='[DAGBuilder] %(message)s')
logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, pd.Series):
            return obj.tolist()
        return super().default(obj)


@dataclass
class DAGBuilderConfig:
    disease_name: str = "Disease"
    disease_node: str = "Disease_Activity"
    consensus_threshold: float = 0.20
    correlation_threshold: float = 0.4
    gwas_pval_threshold: float = 5e-8
    mr_pval_threshold: float = 0.05
    granger_q_threshold: float = 0.05
    granger_rescue_q_threshold: float = 0.01
    z_score_threshold_prior: float = 0.75
    z_score_threshold_default: float = 1.0
    program_activity_threshold: float = 0.5
    phenotype_corr_threshold: float = 0.3
    ace_driver_threshold: float = -0.1
    ace_effector_threshold: float = -0.099999999
    ace_min_delta: float = 0.2
    weights: Dict[str, float] = field(default_factory=lambda: {
        'GWAS': 0.90,
        'MR': 0.95,
        'CRISPR': 0.85,
        'SIGNOR': 0.90,
        'TEMPORAL': 0.65,
        'STATISTICAL': 0.35,
        'DATABASE': 0.80,
        'EQTL': 0.85,
    })


class DataLoader:
    """Loads and resolves multi-modal biological data for DAG construction."""

    def __init__(self, config: DAGBuilderConfig):
        self.config = config
        self.data: Dict[str, Any] = {}
        self.synonym_map: Dict[str, str] = {}
        self.ensembl_map: Dict[str, str] = {}

    def _parse_and_resolve(self, raw_string) -> List[str]:
        if pd.isna(raw_string) or raw_string == "":
            return []
        raw_string = str(raw_string)
        cleaned = re.sub(r'(\s+-\s+)|;', ',', raw_string)
        resolved = []
        for g in cleaned.split(','):
            g = g.strip().upper()
            if not g:
                continue
            base_g = g.split('.')[0] if g.startswith('ENSG') else g
            if base_g in self.ensembl_map:
                g = self.ensembl_map[base_g]
            elif g in self.synonym_map:
                g = self.synonym_map[g]
            if g.startswith('ENSG'):
                continue
            resolved.append(g)
        return resolved

    def load_all(self, base_dir: str) -> Dict[str, Any]:
        self._load_transcriptomics(base_dir)
        self._load_pathways(base_dir)
        self._load_context(base_dir)
        self._load_perturbation(base_dir)
        self._load_genetics(base_dir)
        return self.data

    def _load_transcriptomics(self, base_dir: str):
        candidates_path = os.path.join(base_dir, "DEGs_genes_pathway",
                                       f"{self.config.disease_name}_DEGs_prioritized.csv")
        if not os.path.exists(candidates_path):
            candidates_files = [f for f in os.listdir(os.path.join(base_dir, "DEGs_genes_pathway"))
                                if 'prioritized' in f.lower() and f.endswith('.csv')]
            if candidates_files:
                candidates_path = os.path.join(base_dir, "DEGs_genes_pathway", candidates_files[0])

        self.data['candidates_df'] = pd.read_csv(candidates_path)
        self.data['candidates_df']['Gene'] = self.data['candidates_df']['Gene'].str.upper()

        if 'HGNC_Synonyms' in self.data['candidates_df'].columns:
            for _, row in self.data['candidates_df'].iterrows():
                canonical = row['Gene']
                if pd.notna(row.get('HGNC_Synonyms')):
                    syns = str(row['HGNC_Synonyms']).replace(';', ',').split(',')
                    for s in syns:
                        clean_s = s.strip().upper()
                        if clean_s and clean_s != canonical:
                            self.synonym_map[clean_s] = canonical

        deg_genes = set(self.data['candidates_df']['Gene'].unique())

        self.data['literature_scores'] = {}
        lit_cols = [c for c in self.data['candidates_df'].columns
                    if 'pubmed' in c.lower() or 'literature' in c.lower()]
        if lit_cols:
            col = lit_cols[0]
            self.data['candidates_df']['lit_score'] = np.log1p(
                self.data['candidates_df'][col].fillna(0))
            self.data['literature_scores'] = (
                self.data['candidates_df'].set_index('Gene')['lit_score'].to_dict())

        raw_count_files = [f for f in os.listdir(base_dir)
                           if 'raw_count' in f.lower() and f.endswith('.tsv')]
        if raw_count_files:
            self.data['raw_counts'] = pd.read_csv(
                os.path.join(base_dir, raw_count_files[0]), sep='\t')
        else:
            self.data['raw_counts'] = pd.DataFrame(columns=['Gene'])

        meta_path = os.path.join(base_dir, "DEGs_genes_pathway", "prep_meta.csv")
        if os.path.exists(meta_path):
            self.data['metadata'] = pd.read_csv(meta_path)
        else:
            self.data['metadata'] = pd.DataFrame(columns=['sample_id', 'condition'])

        ens_col = next((c for c in self.data['raw_counts'].columns
                        if 'ens' in c.lower()), None)
        if ens_col and 'Gene' in self.data['raw_counts'].columns:
            for _, row in self.data['raw_counts'].iterrows():
                ens_val = str(row[ens_col]).strip().upper()
                sym_val = str(row['Gene']).strip().upper()
                if ens_val.startswith('ENSG') and sym_val != 'NAN':
                    base_ens = ens_val.split('.')[0]
                    self.ensembl_map[ens_val] = sym_val
                    self.ensembl_map[base_ens] = sym_val

        genetic_genes = set()
        self._try_expand_gwas(base_dir, genetic_genes)
        self._try_expand_mr(base_dir, genetic_genes)

        expanded_set = deg_genes.union(genetic_genes)
        available_genes = set(self.data['raw_counts']['Gene'].str.upper().unique()) if 'Gene' in self.data['raw_counts'].columns else expanded_set
        final_list = list(expanded_set.intersection(available_genes)) if available_genes else list(expanded_set)

        self.data['gene_list'] = final_list
        self.data['node_meta'] = self.data['candidates_df'].set_index('Gene').to_dict(orient='index')
        logger.info(f"Loaded {len(self.data['gene_list'])} genes "
                     f"(union of {len(deg_genes)} DEGs + genetics).")

    def _try_expand_gwas(self, base_dir, genetic_genes: set):
        try:
            gwas_files = [f for f in os.listdir(os.path.join(base_dir, "GWAS_data"))
                          if f.endswith('.xlsx') or f.endswith('.csv')]
            for gf in gwas_files:
                path = os.path.join(base_dir, "GWAS_data", gf)
                if gf.endswith('.xlsx'):
                    gwas = pd.read_excel(path)
                else:
                    gwas = pd.read_csv(path)
                if 'P-VALUE' in gwas.columns and 'MAPPED_GENE' in gwas.columns:
                    hits = gwas[gwas['P-VALUE'] < self.config.gwas_pval_threshold]['MAPPED_GENE'].dropna().unique()
                    for g_raw in hits:
                        genetic_genes.update(self._parse_and_resolve(g_raw))
                    break
        except Exception as e:
            logger.warning(f"Could not expand with GWAS: {e}")

    def _try_expand_mr(self, base_dir, genetic_genes: set):
        try:
            mr_path = os.path.join(base_dir, "GWAS_data", "MR_MAIN_RESULTS_ALL_GENES.csv")
            if os.path.exists(mr_path):
                mr = pd.read_csv(mr_path)
                if 'pval' in mr.columns:
                    col = 'gene' if 'gene' in mr.columns else 'exposure'
                    if col in mr.columns:
                        hits = mr[mr['pval'] < self.config.mr_pval_threshold][col].dropna().unique()
                        for g_raw in hits:
                            genetic_genes.update(self._parse_and_resolve(g_raw))
        except Exception as e:
            logger.warning(f"Could not expand with MR: {e}")

    def _load_pathways(self, base_dir: str):
        pathway_files = [f for f in os.listdir(os.path.join(base_dir, "DEGs_genes_pathway"))
                         if 'pathway' in f.lower() and 'enrichment' in f.lower() and f.endswith('.csv')]
        if not pathway_files:
            self.data['pathways_df'] = pd.DataFrame()
            self.data['bioprog_df'] = pd.DataFrame()
            self.data['gene_to_pathway'] = {}
            self.data['gene_to_bioprog'] = {}
            self.data['pathway_ontology'] = {}
            return

        path = pd.read_csv(os.path.join(base_dir, "DEGs_genes_pathway", pathway_files[0]))
        if 'DB_ID' in path.columns:
            self.data['pathways_df'] = path[path['DB_ID'].isin(['REACTOME', 'KEGG', 'WIKIPATHWAY'])]
            self.data['bioprog_df'] = path[path['DB_ID'] == 'GO_BP']
        else:
            self.data['pathways_df'] = path
            self.data['bioprog_df'] = pd.DataFrame()

        self.data['gene_to_pathway'] = self._map_genes_to_programs(self.data['pathways_df'])
        self.data['gene_to_bioprog'] = self._map_genes_to_programs(self.data['bioprog_df'])

        self.data['pathway_ontology'] = {}
        for _, row in path.iterrows():
            p_name = row.get('Pathway', '')
            self.data['pathway_ontology'][p_name] = {
                'Main_Class': row.get('Main_Class', 'Unclassified'),
                'Sub_Class': row.get('Sub_Class', 'Unclassified'),
            }

    def _map_genes_to_programs(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        gene_col = 'Pathway associated genes'
        if gene_col not in df.columns:
            return mapping
        for _, row in df.iterrows():
            if pd.isna(row[gene_col]):
                continue
            resolved = self._parse_and_resolve(row[gene_col])
            for g in resolved:
                if g in self.data['gene_list']:
                    mapping.setdefault(g, []).append(row.get('Pathway', ''))
        return mapping

    def _load_context(self, base_dir: str):
        self.data['cell_map'] = {}
        self.data['cell_types'] = []
        self.data['cell_markers'] = {}
        self.data['peak_times'] = {}
        self.data['granger_edges'] = {}

        deconv_path = os.path.join(base_dir, "deconvolution_data", "signature_matrix.tsv")
        if os.path.exists(deconv_path):
            try:
                sig = pd.read_csv(deconv_path, sep='\t')
                if 'gene' in sig.columns:
                    sig['gene'] = sig['gene'].apply(
                        lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
                    sig = sig.dropna(subset=['gene'])
                    sig = sig[sig['gene'].isin(self.data['gene_list'])]
                if not sig.empty and len(sig.columns) > 2:
                    sig['Dominant'] = sig.iloc[:, 1:].idxmax(axis=1)
                    self.data['cell_map'] = sig.set_index('gene')['Dominant'].to_dict()
                    self.data['cell_types'] = [c for c in sig.columns if c not in ['gene', 'Dominant']]
                    for cell in self.data['cell_types']:
                        markers = sig.nlargest(100, cell)['gene'].tolist()
                        self.data['cell_markers'][cell] = [m for m in markers if m in self.data['gene_list']]
            except Exception as e:
                logger.warning(f"Could not load deconvolution: {e}")

        temp_path = os.path.join(base_dir, "temporal_data", "temporal_gene_fits.tsv")
        if os.path.exists(temp_path):
            try:
                temp = pd.read_csv(temp_path, sep='\t')
                temp['gene_id'] = temp['gene_id'].apply(
                    lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
                temp = temp.dropna(subset=['gene_id'])
                temp = temp[temp['gene_id'].isin(self.data['gene_list'])]
                self.data['peak_times'] = temp.set_index('gene_id')['time_of_peak'].to_dict()
            except Exception as e:
                logger.warning(f"Could not load temporal fits: {e}")

        granger_path = os.path.join(base_dir, "temporal_data", "granger_edges_raw.csv")
        if os.path.exists(granger_path):
            try:
                granger = pd.read_csv(granger_path)
                granger['source'] = granger['source'].apply(
                    lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
                granger['target'] = granger['target'].apply(
                    lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
                granger = granger.dropna(subset=['source', 'target'])
                valid = granger[
                    (granger['source'].isin(self.data['gene_list'])) &
                    (granger['target'].isin(self.data['gene_list']))]
                for _, row in valid.iterrows():
                    key = (row['source'], row['target'])
                    if key not in self.data['granger_edges'] or row['effect_f'] > self.data['granger_edges'][key]['effect_f']:
                        self.data['granger_edges'][key] = {
                            'q_value': row['q_value'], 'p_value': row['p_value'],
                            'effect_f': row['effect_f'], 'lag': row['lag'],
                        }
            except Exception as e:
                logger.warning(f"Could not load Granger edges: {e}")

    def _load_perturbation(self, base_dir: str):
        self.data['perturbation_meta'] = {}
        self.data['druggability'] = {}
        pert_dir = os.path.join(base_dir, "perturbation_data")
        if not os.path.isdir(pert_dir):
            return

        drivers_path = os.path.join(pert_dir, "CausalDrivers_Ranked.csv")
        if os.path.exists(drivers_path):
            try:
                drivers = pd.read_csv(drivers_path)
                for _, row in drivers.iterrows():
                    for g in self._parse_and_resolve(row['gene']):
                        if g in self.data['gene_list']:
                            self.data['perturbation_meta'][g] = {
                                'ACE': row['ACE'],
                                'TherapeuticAlignment': row.get('TherapeuticAlignment', 'Unknown'),
                                'Verdict': row.get('Verdict', 'Unknown'),
                                'BestEssentialityTag': row.get('BestEssentialityTag', 'Unknown'),
                            }
            except Exception as e:
                logger.warning(f"Could not load causal drivers: {e}")

        ess_path = os.path.join(pert_dir, "GeneEssentiality_ByMedian.csv")
        if os.path.exists(ess_path):
            try:
                ess = pd.read_csv(ess_path)
                for _, row in ess.iterrows():
                    for g in self._parse_and_resolve(row['Gene']):
                        if g in self.data['gene_list']:
                            self.data['perturbation_meta'].setdefault(
                                g, {'ACE': 0.0, 'BestEssentialityTag': 'Unknown'})
                            self.data['perturbation_meta'][g]['IsEssential'] = row.get(
                                'IsEssential_byMedianRule', False)
            except Exception:
                pass

        drug_path = os.path.join(pert_dir, "causal_link_table_with_relevance.csv")
        if os.path.exists(drug_path):
            try:
                drugs = pd.read_csv(drug_path)
                drugs['Gene'] = drugs['Gene'].apply(
                    lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
                drugs = drugs.dropna(subset=['Gene'])
                valid = drugs[drugs['Gene'].isin(self.data['gene_list'])]
                if not valid.empty:
                    self.data['druggability'] = valid.groupby('Gene').apply(
                        lambda x: x[['Drug', 'Therapeutic_Relevance']].to_dict('records')).to_dict()
            except Exception:
                self.data['druggability'] = {}

    def _load_genetics(self, base_dir: str):
        gwas_dir = os.path.join(base_dir, "GWAS_data")
        self.data['gwas_roots'] = []
        self.data['gwas_snps'] = []
        self.data['eqtl_edges'] = []
        self.data['eqtl_snps'] = []
        self.data['mr_evidence'] = {}
        self.data['genetic_confidence'] = {}
        self.data['signor_edges'] = pd.DataFrame()
        self.data['lr_pairs'] = set()
        self.data['coloc_proxy'] = set()

        if not os.path.isdir(gwas_dir):
            return

        gwas_files = [f for f in os.listdir(gwas_dir) if f.endswith('.xlsx')]
        for gf in gwas_files:
            try:
                gwas = pd.read_excel(os.path.join(gwas_dir, gf))
                if 'P-VALUE' in gwas.columns and 'MAPPED_GENE' in gwas.columns:
                    sig = gwas[gwas['P-VALUE'] < self.config.gwas_pval_threshold]
                    valid_roots, valid_snps = set(), set()
                    for _, row in sig.iterrows():
                        for g in self._parse_and_resolve(row['MAPPED_GENE']):
                            if g in self.data['gene_list']:
                                valid_roots.add(g)
                                if 'SNPS' in row:
                                    valid_snps.add(row['SNPS'])
                    self.data['gwas_roots'] = list(valid_roots)
                    self.data['gwas_snps'] = list(valid_snps)
                    break
            except Exception as e:
                logger.warning(f"Could not load GWAS: {e}")

        signor_dir = os.path.join(base_dir, "SIGNOR_data")
        if not os.path.isdir(signor_dir):
            signor_dir = os.path.join(base_dir, "signor_data")
        if os.path.isdir(signor_dir):
            signor_files = [f for f in os.listdir(signor_dir)
                            if 'subnetwork' in f.lower() and f.endswith('.tsv')]
            for sf in signor_files:
                try:
                    sig = pd.read_csv(os.path.join(signor_dir, sf), sep='\t')
                    sig['source'] = sig['source'].apply(
                        lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
                    sig['target'] = sig['target'].apply(
                        lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
                    sig = sig.dropna(subset=['source', 'target'])
                    self.data['signor_edges'] = sig[
                        (sig['source'].isin(self.data['gene_list'])) &
                        (sig['target'].isin(self.data['gene_list']))]
                    lr = self.data['signor_edges'][
                        self.data['signor_edges']['MECHANISM'].isin(['binding', 'receptor signaling'])]
                    self.data['lr_pairs'] = set(zip(lr['source'], lr['target']))
                    break
                except Exception:
                    pass

        gene_ev_files = [f for f in os.listdir(gwas_dir) if 'genelevel' in f.lower() and f.endswith('.tsv')]
        for gef in gene_ev_files:
            try:
                gene_ev = pd.read_csv(os.path.join(gwas_dir, gef), sep='\t')
                gene_ev['Gene_Symbol'] = gene_ev['Gene_Symbol'].apply(
                    lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
                gene_ev = gene_ev.dropna(subset=['Gene_Symbol'])
                gene_ev = gene_ev[gene_ev['Gene_Symbol'].isin(self.data['gene_list'])]
                self.data['genetic_confidence'] = gene_ev.set_index(
                    'Gene_Symbol')['Gene_Genetic_Confidence_Score'].to_dict()
                break
            except Exception:
                pass

        var_ev_files = [f for f in os.listdir(gwas_dir) if 'variantlevel' in f.lower() and f.endswith('.tsv')]
        for vef in var_ev_files:
            try:
                var_ev = pd.read_csv(os.path.join(gwas_dir, vef), sep='\t')
                var_ev['Gene_Symbol'] = var_ev['Gene_Symbol'].apply(
                    lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
                var_ev = var_ev.dropna(subset=['Gene_Symbol'])
                var_ev = var_ev[var_ev['Gene_Symbol'].isin(self.data['gene_list'])]
                self.data['eqtl_edges'] = var_ev[['SNP', 'Gene_Symbol', 'eQTL_beta']].to_dict(orient='records')
                self.data['eqtl_snps'] = var_ev['SNP'].unique().tolist()
                break
            except Exception:
                pass

        mr_path = os.path.join(gwas_dir, "MR_MAIN_RESULTS_ALL_GENES.csv")
        if os.path.exists(mr_path):
            try:
                mr_res = pd.read_csv(mr_path)
                col = 'gene' if 'gene' in mr_res.columns else 'exposure'
                mr_res[col] = mr_res[col].apply(
                    lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
                mr_res = mr_res.dropna(subset=[col])
                mr_res = mr_res[mr_res[col].isin(self.data['gene_list'])]
                self.data['mr_evidence'] = mr_res.set_index(col)[
                    ['b', 'pval', 'direction']].to_dict(orient='index')
            except Exception:
                pass

        self.data['coloc_proxy'] = set(self.data['gwas_roots']).intersection(
            {e['Gene_Symbol'] for e in self.data['eqtl_edges']})


class DAGBuilder:
    """Constructs consensus causal DAG from multi-modal biological data."""

    def __init__(self, config: Optional[DAGBuilderConfig] = None):
        self.config = config or DAGBuilderConfig()
        self.W = self.config.weights
        self.data: Dict[str, Any] = {}
        self.global_skeleton: Optional[nx.DiGraph] = None
        self.cohort_expression: Optional[pd.DataFrame] = None
        self.disease_vector: Optional[pd.Series] = None
        self.expr_stats: Dict[str, pd.Series] = {}

    def load_data(self, base_dir: str) -> Dict[str, Any]:
        loader = DataLoader(self.config)
        self.data = loader.load_all(base_dir)
        if not self.data['raw_counts'].empty and 'Gene' in self.data['raw_counts'].columns:
            self.cohort_expression = self._process_expression()
            self.disease_vector = self._process_metadata()
            self.expr_stats = self._calculate_expression_stats()
        return self.data

    def load_data_packet(self, data_packet: Dict[str, Any]):
        self.data = data_packet
        if 'raw_counts' in data_packet and not data_packet['raw_counts'].empty:
            self.cohort_expression = self._process_expression()
            self.disease_vector = self._process_metadata()
            self.expr_stats = self._calculate_expression_stats()

    def _aggregate_confidence(self, current: float, new_evidence: float) -> float:
        current = max(0.0, min(0.99, current))
        new_evidence = max(0.0, min(0.99, new_evidence))
        return min(0.99, 1.0 - ((1.0 - current) * (1.0 - new_evidence)))

    def _confidence_label(self, score: float) -> str:
        if score >= 0.85:
            return "Very High"
        if score >= 0.65:
            return "High"
        if score >= 0.40:
            return "Medium"
        return "Low"

    def _process_expression(self) -> pd.DataFrame:
        raw = self.data['raw_counts']
        valid_genes = [g for g in self.data['gene_list'] if g in raw['Gene'].values]
        expr = raw[raw['Gene'].isin(valid_genes)].set_index('Gene').iloc[:, 3:].astype(float)
        if expr.index.duplicated().any():
            expr = expr.groupby(level=0).mean()
        return np.log1p(expr)

    def _process_metadata(self) -> pd.Series:
        meta = self.data['metadata']
        samples = self.cohort_expression.columns.tolist()
        meta = meta.set_index('sample_id').reindex(samples)
        return meta['condition'].map({'Disease': 1, 'Control': 0}).fillna(0)

    def _calculate_expression_stats(self) -> Dict[str, pd.Series]:
        meta = self.data['metadata'].set_index('sample_id')
        available = self.cohort_expression.columns
        dis_samples = [s for s in available if s in meta.index and meta.loc[s, 'condition'] == 'Disease']
        ctrl_samples = [s for s in available if s in meta.index and meta.loc[s, 'condition'] == 'Control']
        stats = {}
        stats['baseline'] = self.cohort_expression[ctrl_samples].mean(axis=1) if ctrl_samples else pd.Series(0, index=self.cohort_expression.index)
        stats['diseased'] = self.cohort_expression[dis_samples].mean(axis=1) if dis_samples else pd.Series(0, index=self.cohort_expression.index)
        return stats

    def build_consensus_dag(self) -> Tuple[nx.DiGraph, List[nx.DiGraph], Dict]:
        logger.info("Building layered skeleton...")
        self.global_skeleton = self._build_layered_skeleton()

        if self.cohort_expression is not None:
            samples = self.cohort_expression.columns.tolist()
            n_samples = len(samples)
            logger.info(f"Instantiating patient DAGs for {n_samples} patients...")

            # OPTIMIZATION: Pre-compute mean/std ONCE (was recomputed per patient)
            self._expr_mean = self.cohort_expression.mean(axis=1)
            self._expr_std = self.cohort_expression.std(axis=1).replace(0, 1)

            # OPTIMIZATION: Pre-extract regulatory nodes ONCE
            self._reg_nodes = [n for n, d in self.global_skeleton.nodes(data=True)
                               if d.get('layer') == 'regulatory']
            self._prog_nodes = [n for n, d in self.global_skeleton.nodes(data=True)
                                if d.get('layer') == 'program']

            # OPTIMIZATION: Cap patients for large cohorts (diminishing returns after ~50)
            max_patients = min(n_samples, 50)
            if n_samples > max_patients:
                logger.info(f"Sampling {max_patients} of {n_samples} patients (optimization)")
                import random
                samples = random.sample(samples, max_patients)

            patient_dags = []
            for i, sid in enumerate(samples):
                if (i + 1) % 10 == 0:
                    logger.info(f"  Patient DAG {i+1}/{len(samples)}...")
                patient_dags.append(self._instantiate_patient_dag(self.global_skeleton, sid))

            logger.info("Aggregating into consensus DAG...")
            consensus = self._aggregate_dags(patient_dags)
        else:
            logger.info("No expression data — using skeleton as consensus.")
            consensus = self.global_skeleton.copy()
            patient_dags = []

        metrics = {
            'n_nodes': len(consensus.nodes),
            'n_edges': len(consensus.edges),
            'n_gwas_roots': len([n for n, d in consensus.nodes(data=True) if d.get('is_gwas_hit')]),
            'n_snp_nodes': len([n for n, d in consensus.nodes(data=True) if d.get('type') == 'snp']),
            'n_mr_edges': len([u for u, v, d in consensus.edges(data=True)
                               if 'mendelian_randomization_causality' in d.get('evidence', [])]),
            'n_perturbation_drivers': len([u for u, v, d in consensus.edges(data=True)
                                           if 'perturbation_asymmetry' in d.get('evidence', [])]),
        }
        logger.info(f"Consensus DAG: {metrics['n_nodes']} nodes, {metrics['n_edges']} edges")
        return consensus, patient_dags, metrics

    def _build_layered_skeleton(self) -> nx.DiGraph:
        G = nx.DiGraph()
        cfg = self.config

        # Layer 0: SNP source nodes
        gwas_snps = set(self.data.get('gwas_snps', []))
        eqtl_snps = set(self.data.get('eqtl_snps', []))
        for snp in gwas_snps.union(eqtl_snps):
            src_type = 'GWAS_and_eQTL' if snp in gwas_snps and snp in eqtl_snps else (
                'GWAS_Catalog' if snp in gwas_snps else 'eQTL_Study')
            G.add_node(snp, layer='source', type='snp', is_anchor=True, evidence_source=src_type)

        # Layer 1: Gene regulatory nodes
        coloc_set = self.data.get('coloc_proxy', set())
        lit_scores = self.data.get('literature_scores', {})
        for gene in self.data['gene_list']:
            meta = self.data['node_meta'].get(gene, {})
            drugs = self.data['druggability'].get(gene, [])
            gen_conf = self.data.get('genetic_confidence', {}).get(gene, 0)
            pert_meta = self.data.get('perturbation_meta', {}).get(gene, {})
            lit_score = lit_scores.get(gene, 0.0)

            base_expr = self.expr_stats.get('baseline', pd.Series()).get(gene, 0.0)
            dis_expr = self.expr_stats.get('diseased', pd.Series()).get(gene, 0.0)
            if isinstance(base_expr, pd.Series):
                base_expr = base_expr.mean()
            if isinstance(dis_expr, pd.Series):
                dis_expr = dis_expr.mean()
            if isinstance(gen_conf, pd.Series):
                gen_conf = gen_conf.mean()

            ace_score = float(round(pert_meta.get('ACE', 0.0), 4))
            global_essentiality = pert_meta.get('IsEssential', False)
            best_ess_tag = pert_meta.get('BestEssentialityTag', 'Unknown')
            alignment = pert_meta.get('TherapeuticAlignment', 'Unknown')

            ess_tag = best_ess_tag if best_ess_tag and best_ess_tag != 'Unknown' else (
                "Core Essential" if global_essentiality else "Unknown")
            systemic_tox = "High" if global_essentiality else "Low"

            recommendation = "Context-Dependent"
            strategy_type = "Unknown"
            if ace_score <= cfg.ace_driver_threshold:
                if alignment == 'Aggravating':
                    recommendation, strategy_type = "Inhibit", "Antagonist / Inhibitor"
                elif alignment == 'Reversal':
                    recommendation, strategy_type = "Activate", "Agonist / Stabilizer"
            if systemic_tox == "High":
                recommendation += " (WARNING: High systemic toxicity risk)"

            is_gwas = gene in self.data['gwas_roots']
            has_mr = gene in self.data.get('mr_evidence', {})
            evidence_cnt = sum([is_gwas, has_mr, ace_score <= cfg.ace_driver_threshold, lit_score > 2.0])
            tier = "Validated Driver" if (is_gwas or has_mr or ace_score <= cfg.ace_driver_threshold) else "Supported"

            G.add_node(gene, layer='regulatory', type='gene',
                       expression_baseline=float(round(base_expr, 3)),
                       expression_diseased=float(round(dis_expr, 3)),
                       expression_log2fc=float(round(meta.get('Patient_LFC_mean', 0.0), 3)),
                       expression_trend="Up" if dis_expr > base_expr else "Down",
                       is_gwas_hit=is_gwas,
                       genetic_confidence_score=float(gen_conf) if isinstance(gen_conf, (int, float)) else 0.0,
                       perturbation_ace=ace_score,
                       essentiality_tag=ess_tag,
                       systemic_toxicity_risk=systemic_tox,
                       therapeutic_alignment=alignment,
                       therapeutic_recommendation=recommendation,
                       strategy_type=strategy_type,
                       evidence_count=evidence_cnt,
                       causal_tier=tier)

        # Layer 2: Program nodes (pathways, biological programs, cell types)
        if not self.data.get('pathways_df', pd.DataFrame()).empty:
            path_meta = self.data['pathways_df'].drop_duplicates(subset=['Pathway']).set_index('Pathway')
            path_cols = [c for c in ['p_value', 'fdr', 'Regulation', 'Pathway ID'] if c in path_meta.columns]
            if path_cols:
                path_meta = path_meta[path_cols].to_dict('index')
            else:
                path_meta = {}
        else:
            path_meta = {}

        pathway_ontology = self.data.get('pathway_ontology', {})
        for p in set(p for sublist in self.data.get('gene_to_pathway', {}).values() for p in sublist):
            info = path_meta.get(p, {})
            ont = pathway_ontology.get(p, {})
            G.add_node(p, layer='program', type='pathway', source='REACTOME/KEGG',
                       pathway_id=info.get('Pathway ID', 'NA'),
                       p_value=float(info.get('p_value', 1.0)),
                       fdr=float(info.get('fdr', 1.0)),
                       regulation=info.get('Regulation', 'Unknown'),
                       main_class=ont.get('Main_Class', 'Unclassified'),
                       sub_class=ont.get('Sub_Class', 'Unclassified'))

        for p in set(p for sublist in self.data.get('gene_to_bioprog', {}).values() for p in sublist):
            ont = pathway_ontology.get(p, {})
            G.add_node(p, layer='program', type='biological_program', source='GO_BP',
                       main_class=ont.get('Main_Class', 'Unclassified'),
                       sub_class=ont.get('Sub_Class', 'Unclassified'))

        for cell in self.data.get('cell_types', []):
            markers = self.data['cell_markers'].get(cell, [])
            if markers:
                G.add_node(cell, layer='program', type='cell_type', source='Bisque_Deconvolution',
                           marker_count=len(markers),
                           main_class='Cellular Environment', sub_class='Immune Cell Type')

        # Layer 3: Disease trait node
        G.add_node(cfg.disease_node, layer='trait', type='clinical_outcome', measure='Disease_Score')

        # === EDGES ===
        # 1. Transcriptomic co-expression
        if self.cohort_expression is not None and len(self.cohort_expression) > 1:
            corr = self.cohort_expression.T.corr()
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    u, v = corr.columns[i], corr.columns[j]
                    val = abs(corr.iloc[i, j])
                    if val > cfg.correlation_threshold:
                        scaled = (val - cfg.correlation_threshold) / (1.0 - cfg.correlation_threshold)
                        base_prob = min(0.99, scaled * self.W['STATISTICAL'])
                        for src, tgt in [(u, v), (v, u)]:
                            G.add_edge(src, tgt, weight=val, edge_type='regulatory',
                                       mechanism='inferred_regulation',
                                       evidence={'transcriptomic_coexpression'},
                                       confidence_score=round(base_prob, 3),
                                       confidence=self._confidence_label(base_prob))

        # 2. Pathway membership edges
        for gene, progs in self.data.get('gene_to_pathway', {}).items():
            if gene in G.nodes:
                for p in progs:
                    G.add_edge(gene, p, weight=0.9, edge_type='program_membership',
                               mechanism='pathway_annotation',
                               evidence={'reactome_kegg_curation'},
                               confidence_score=self.W['DATABASE'],
                               confidence=self._confidence_label(self.W['DATABASE']))

        for gene, progs in self.data.get('gene_to_bioprog', {}).items():
            if gene in G.nodes:
                for p in progs:
                    score = self.W['DATABASE'] * 0.95
                    G.add_edge(gene, p, weight=0.9, edge_type='program_membership',
                               mechanism='biological_process_annotation',
                               evidence={'gene_ontology_annotation'},
                               confidence_score=round(score, 3),
                               confidence=self._confidence_label(score))

        # 3. Cell marker edges
        for cell, markers in self.data.get('cell_markers', {}).items():
            if cell in G.nodes:
                for m in markers:
                    score = self.W['DATABASE'] * 0.90
                    G.add_edge(m, cell, weight=1.0, edge_type='cell_marker',
                               mechanism='cell_identity_definition',
                               evidence={'transcriptomic_deconvolution_marker'},
                               confidence_score=round(score, 3),
                               confidence=self._confidence_label(score))

        return G

    def _instantiate_patient_dag(self, global_graph: nx.DiGraph, sample_id: str) -> nx.DiGraph:
        # OPTIMIZATION: Use pre-computed mean/std instead of recalculating
        patient_expr = self.cohort_expression[sample_id]
        z_scores = (patient_expr - self._expr_mean) / self._expr_std
        if z_scores.index.duplicated().any():
            z_scores = z_scores.groupby(level=0).mean()

        # OPTIMIZATION: Use pre-extracted node lists instead of re-scanning entire graph
        active_nodes = {self.config.disease_node}

        # Program nodes
        for prog in self._prog_nodes:
            sources = [u for u, _ in global_graph.in_edges(prog)]
            if sources:
                vals = [float(z_scores.get(g, 0)) if not isinstance(z_scores.get(g, 0), pd.Series)
                        else float(z_scores.get(g, 0).mean()) for g in sources]
                score = np.mean(vals)
                if abs(score) > self.config.program_activity_threshold:
                    active_nodes.add(prog)

        # Regulatory nodes — OPTIMIZATION: iterate pre-extracted list, not full graph
        for node in self._reg_nodes:
            z = z_scores.get(node, 0)
            if isinstance(z, pd.Series):
                z = float(z.mean())
            else:
                z = float(z)
            nd = global_graph.nodes[node]
            is_prior = nd.get('is_gwas_hit', False) or (nd.get('perturbation_ace', 0) <= self.config.ace_driver_threshold)
            threshold = self.config.z_score_threshold_prior if is_prior else self.config.z_score_threshold_default
            if abs(z) > threshold:
                active_nodes.add(node)

        # OPTIMIZATION: Don't copy full graph — only create subgraph directly
        return global_graph.subgraph(list(active_nodes)).copy()

    def _aggregate_dags(self, patient_dags: List[nx.DiGraph]) -> nx.DiGraph:
        consensus = nx.DiGraph()
        n_patients = len(patient_dags)
        edge_stats: Dict[Tuple[str, str], Dict] = {}

        for dag in patient_dags:
            for u, v, d in dag.edges(data=True):
                edge = (u, v)
                if edge not in edge_stats:
                    edge_stats[edge] = {
                        'count': 0, 'weight_sum': 0,
                        'mechanism': d.get('mechanism'),
                        'edge_type': d.get('edge_type'),
                        'evidence': set(d.get('evidence', set())),
                        'confidence_score_sum': 0.0,
                    }
                edge_stats[edge]['count'] += 1
                edge_stats[edge]['weight_sum'] += d.get('weight', 0)
                edge_stats[edge]['confidence_score_sum'] += d.get('confidence_score', 0)
                if 'evidence' in d:
                    edge_stats[edge]['evidence'].update(
                        d['evidence'] if isinstance(d['evidence'], set) else set(d['evidence']))

        for (u, v), stats in edge_stats.items():
            freq = stats['count'] / n_patients
            if freq > self.config.consensus_threshold:
                base_prob = stats['confidence_score_sum'] / stats['count']
                freq_bonus = freq * self.W['STATISTICAL'] * 0.5
                final_prob = min(0.99, self._aggregate_confidence(base_prob, freq_bonus))
                consensus.add_edge(u, v, weight=stats['weight_sum'] / stats['count'],
                                   frequency=freq, edge_type=stats['edge_type'],
                                   mechanism=stats['mechanism'], evidence=stats['evidence'],
                                   confidence_score=round(final_prob, 3),
                                   confidence=self._confidence_label(final_prob))

        for node in consensus.nodes:
            if node in self.global_skeleton.nodes:
                consensus.nodes[node].update(self.global_skeleton.nodes[node])

        self._phenotype_arbitration(consensus, patient_dags)
        self._inject_priors(consensus)
        self._apply_advanced_constraints(consensus)

        # Convert sets to lists for serialization
        for u, v, d in consensus.edges(data=True):
            if isinstance(d.get('evidence'), set):
                consensus[u][v]['evidence'] = list(d['evidence'])

        return consensus

    def _phenotype_arbitration(self, consensus: nx.DiGraph, patient_dags: List[nx.DiGraph]):
        trait_node = self.config.disease_node
        if trait_node not in consensus:
            consensus.add_node(trait_node, layer='trait', type='clinical_outcome')
        elif trait_node in self.global_skeleton.nodes:
            consensus.nodes[trait_node].update(self.global_skeleton.nodes[trait_node])

        node_activity_data = []
        disease_data = []
        for dag in patient_dags:
            row = {}
            for node, d in dag.nodes(data=True):
                if d.get('layer') in ['program', 'regulatory']:
                    row[node] = d.get('patient_activity', 0)
            node_activity_data.append(row)
            if trait_node in dag.nodes:
                disease_data.append(dag.nodes[trait_node].get('patient_activity', 0))

        if not disease_data:
            return

        df_activity = pd.DataFrame(node_activity_data).fillna(0)
        vec_disease = np.array(disease_data)

        if len(set(vec_disease)) <= 1:
            return

        for node in df_activity.columns:
            if node not in consensus.nodes:
                continue
            if df_activity[node].std() <= 1e-5:
                continue
            corr_val, _ = pearsonr(df_activity[node].values[:len(vec_disease)], vec_disease)
            if abs(corr_val) <= self.config.phenotype_corr_threshold:
                continue

            prob = min(0.99, abs(corr_val) * self.W['STATISTICAL'])
            nd = consensus.nodes[node]
            layer = nd.get('layer')

            if layer == 'regulatory':
                is_causal = (nd.get('is_gwas_hit', False) or
                             node in self.data.get('mr_evidence', {}) or
                             nd.get('perturbation_ace', 0) <= self.config.ace_driver_threshold)
                if is_causal:
                    consensus.add_edge(node, trait_node, weight=abs(corr_val),
                                       edge_type='phenotype_driver', mechanism='causal_disease_driver',
                                       direction='forward' if corr_val > 0 else 'inverse',
                                       evidence={'transcriptomic_correlation', 'causal_anchor_supported'},
                                       confidence_score=round(prob, 3),
                                       confidence=self._confidence_label(prob))
                else:
                    consensus.add_edge(trait_node, node, weight=abs(corr_val),
                                       edge_type='reactive_biomarker', mechanism='disease_induced_expression',
                                       direction='upregulated_by_disease' if corr_val > 0 else 'downregulated_by_disease',
                                       evidence={'transcriptomic_correlation', 'lacks_causal_anchors'},
                                       confidence_score=round(prob, 3),
                                       confidence=self._confidence_label(prob))
                    if consensus.has_edge(node, trait_node):
                        consensus.remove_edge(node, trait_node)

            elif layer == 'program':
                causal_incoming = any(
                    consensus.nodes[pred].get('causal_tier') == "Validated Driver"
                    for pred in consensus.predecessors(node)
                    if pred in consensus.nodes)
                if causal_incoming:
                    consensus.add_edge(node, trait_node, weight=abs(corr_val),
                                       edge_type='phenotype_driver', mechanism='causal_pathway_module',
                                       direction='forward' if corr_val > 0 else 'inverse',
                                       evidence={'transcriptomic_correlation', 'driven_by_causal_genes'},
                                       confidence_score=round(prob, 3),
                                       confidence=self._confidence_label(prob))
                else:
                    consensus.add_edge(trait_node, node, weight=abs(corr_val),
                                       edge_type='reactive_biomarker', mechanism='disease_associated_pathway',
                                       evidence={'transcriptomic_correlation'},
                                       confidence_score=round(prob, 3),
                                       confidence=self._confidence_label(prob))
                    if consensus.has_edge(node, trait_node):
                        consensus.remove_edge(node, trait_node)

    def _inject_priors(self, consensus: nx.DiGraph):
        trait_node = self.config.disease_node

        # MR evidence injection
        for gene, mr_data in self.data.get('mr_evidence', {}).items():
            if gene not in consensus.nodes:
                continue
            mech = ('genetically_predicted_risk_increase' if mr_data['b'] > 0
                    else 'genetically_predicted_protection')
            evidence = {'mendelian_randomization_causality'}
            if consensus.has_edge(gene, trait_node):
                evidence.update(consensus[gene][trait_node].get('evidence', set()))

            pval = mr_data.get('pval', 1.0)
            pval_scalar = 1.0 if pval < 0.001 else (0.90 if pval < 0.01 else 0.80)
            final_prob = min(0.99, self.W['MR'] * pval_scalar)

            if consensus.has_edge(trait_node, gene):
                consensus.remove_edge(trait_node, gene)

            consensus.add_edge(gene, trait_node, weight=abs(mr_data['b']),
                               edge_type='phenotype_driver', mechanism=mech,
                               evidence=evidence, pval=mr_data['pval'],
                               confidence_score=round(final_prob, 3),
                               confidence=self._confidence_label(final_prob))

        # eQTL edge injection
        for edge in self.data.get('eqtl_edges', []):
            snp, gene = edge['SNP'], edge['Gene_Symbol']
            if gene not in consensus.nodes:
                continue
            if snp not in consensus:
                if snp in self.global_skeleton.nodes:
                    consensus.add_node(snp, **self.global_skeleton.nodes[snp])
                else:
                    consensus.add_node(snp, layer='source', type='snp', is_anchor=True)

            beta_val = abs(edge.get('eQTL_beta', 0))
            base_prob = min(0.95, self.W['EQTL'] * min(1.0, beta_val))
            if gene in self.data.get('coloc_proxy', set()):
                final_prob = min(0.99, self._aggregate_confidence(base_prob, self.W['GWAS']))
            else:
                final_prob = base_prob

            consensus.add_edge(snp, gene, weight=beta_val,
                               edge_type='genetic_regulation', mechanism='eQTL_expression_modulation',
                               evidence={'eqtl_study_support'},
                               confidence_score=round(final_prob, 3),
                               confidence=self._confidence_label(final_prob))

        # SIGNOR injection
        if not self.data.get('signor_edges', pd.DataFrame()).empty:
            for _, row in self.data['signor_edges'].iterrows():
                src, tgt = row['source'], row['target']
                if src == tgt or src not in consensus.nodes or tgt not in consensus.nodes:
                    continue
                if not consensus.has_edge(src, tgt):
                    consensus.add_edge(src, tgt, weight=0.3, edge_type='regulatory',
                                       mechanism=row.get('MECHANISM', 'physical_interaction'),
                                       evidence={'signor_physical_interaction'},
                                       confidence_score=0.1,
                                       confidence=self._confidence_label(0.1))
                else:
                    ev = consensus[src][tgt].get('evidence', set())
                    if isinstance(ev, list):
                        ev = set(ev)
                    ev.add('signor_physical_interaction')
                    consensus[src][tgt]['evidence'] = ev

        # Granger edge injection
        granger = self.data.get('granger_edges', {})
        for (u, v), metrics in granger.items():
            if metrics['q_value'] < self.config.granger_q_threshold:
                if u in consensus.nodes and v in consensus.nodes:
                    if not consensus.has_edge(u, v):
                        base_prob = min(0.99, self.W['STATISTICAL'] * min(1.0, metrics['effect_f'] / 40.0))
                        consensus.add_edge(u, v, weight=0.3, edge_type='regulatory',
                                           mechanism='inferred_regulation',
                                           evidence={'var_granger_predictivity'},
                                           confidence_score=round(base_prob, 3),
                                           confidence=self._confidence_label(base_prob))
                    else:
                        ev = consensus[u][v].get('evidence', set())
                        if isinstance(ev, list):
                            ev = set(ev)
                        ev.add('var_granger_predictivity')
                        consensus[u][v]['evidence'] = ev

            # One-node anchor rescue for highly significant edges
            if metrics['q_value'] < self.config.granger_rescue_q_threshold:
                if (u in consensus.nodes) ^ (v in consensus.nodes):
                    if u not in consensus.nodes and u in self.global_skeleton.nodes:
                        consensus.add_node(u, **self.global_skeleton.nodes[u])
                    if v not in consensus.nodes and v in self.global_skeleton.nodes:
                        consensus.add_node(v, **self.global_skeleton.nodes[v])
                    if u in consensus.nodes and v in consensus.nodes and not consensus.has_edge(u, v):
                        base_prob = min(0.99, self.W['STATISTICAL'] * min(1.0, metrics['effect_f'] / 40.0))
                        consensus.add_edge(u, v, weight=0.3, edge_type='regulatory',
                                           mechanism='inferred_regulation',
                                           evidence={'var_granger_predictivity', 'one_node_anchor_rescue'},
                                           confidence_score=round(base_prob, 3),
                                           confidence=self._confidence_label(base_prob))

    def _apply_advanced_constraints(self, G: nx.DiGraph):
        # 1. Genetic anchor flow
        roots = set(self.data.get('gwas_roots', []))
        for u, v in list(G.edges):
            if not G.has_edge(u, v):
                continue
            if G[u][v].get('edge_type') != 'regulatory':
                continue
            if u in roots and v not in roots:
                ev = G[u][v].get('evidence', set())
                if isinstance(ev, list):
                    ev = set(ev)
                ev.add('gwas_anchor_downstream_flow')
                G[u][v]['evidence'] = ev
                new_prob = self._aggregate_confidence(G[u][v]['confidence_score'], self.W['GWAS'] * 0.8)
                G[u][v]['confidence_score'] = round(new_prob, 3)
                G[u][v]['confidence'] = self._confidence_label(new_prob)
                if G.has_edge(v, u):
                    pen = max(0.1, G[v][u]['confidence_score'] * (1.0 - self.W['GWAS'] * 0.5))
                    G[v][u]['confidence_score'] = round(pen, 3)
                    G[v][u]['confidence'] = self._confidence_label(pen)
                    G[v][u]['mechanism'] = 'potential_feedback_loop'

        # 2. Hybrid temporal arbitration
        times = self.data.get('peak_times', {})
        granger = self.data.get('granger_edges', {})
        for u, v in list(G.edges):
            if not G.has_edge(u, v) or G[u][v].get('edge_type') != 'regulatory':
                continue
            has_granger = (u, v) in granger and granger[(u, v)]['q_value'] < self.config.granger_q_threshold
            has_impulse = u in times and v in times and times[u] < (times[v] - 0.05)

            ev = G[u][v].get('evidence', set())
            if isinstance(ev, list):
                ev = set(ev)

            if has_granger and has_impulse:
                if G.has_edge(v, u):
                    G.remove_edge(v, u)
                ev.add('spatiotemporal_causality_verified')
                f_scalar = min(1.5, granger[(u, v)]['effect_f'] / 30.0)
                synergy = self.W['TEMPORAL'] * f_scalar
                new_prob = self._aggregate_confidence(G[u][v]['confidence_score'], synergy)
                G[u][v]['confidence_score'] = round(new_prob, 3)
                G[u][v]['confidence'] = self._confidence_label(new_prob)
            elif has_granger:
                ev.add('var_granger_predictivity')
                f_scalar = min(1.0, granger[(u, v)]['effect_f'] / 40.0)
                new_prob = self._aggregate_confidence(G[u][v]['confidence_score'], self.W['TEMPORAL'] * f_scalar)
                G[u][v]['confidence_score'] = round(new_prob, 3)
                G[u][v]['confidence'] = self._confidence_label(new_prob)
                if G.has_edge(v, u):
                    pen = max(0.1, G[v][u]['confidence_score'] * (1.0 - self.W['TEMPORAL'] * 0.5))
                    G[v][u]['confidence_score'] = round(pen, 3)
                    G[v][u]['confidence'] = self._confidence_label(pen)
            elif has_impulse:
                if G.has_edge(v, u):
                    G.remove_edge(v, u)
                ev.add('pseudotime_progression')
                new_prob = self._aggregate_confidence(G[u][v]['confidence_score'], self.W['TEMPORAL'] * 0.5)
                G[u][v]['confidence_score'] = round(new_prob, 3)
                G[u][v]['confidence'] = self._confidence_label(new_prob)

            G[u][v]['evidence'] = ev

        # 3. Perturbation asymmetry
        cfg = self.config
        for u, v in list(G.edges):
            if not G.has_edge(u, v) or G[u][v].get('edge_type') != 'regulatory':
                continue
            ace_u = G.nodes[u].get('perturbation_ace', 0) if u in G.nodes else 0
            ace_v = G.nodes[v].get('perturbation_ace', 0) if v in G.nodes else 0
            delta = abs(ace_u - ace_v)

            if ace_u <= cfg.ace_driver_threshold and ace_v >= cfg.ace_effector_threshold and delta >= cfg.ace_min_delta:
                if G.has_edge(v, u):
                    G.remove_edge(v, u)
                ev = G[u][v].get('evidence', set())
                if isinstance(ev, list):
                    ev = set(ev)
                ev.add('perturbation_asymmetry')
                G[u][v]['evidence'] = ev
                G[u][v]['mechanism'] = 'fitness_driver_regulation'
                dynamic_w = self.W['CRISPR'] * min(1.0, abs(ace_u))
                new_prob = self._aggregate_confidence(G[u][v]['confidence_score'], dynamic_w)
                G[u][v]['confidence_score'] = round(new_prob, 3)
                G[u][v]['confidence'] = self._confidence_label(new_prob)
            elif ace_v <= cfg.ace_driver_threshold and ace_u >= cfg.ace_effector_threshold and delta >= cfg.ace_min_delta:
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
                if G.has_edge(v, u):
                    ev = G[v][u].get('evidence', set())
                    if isinstance(ev, list):
                        ev = set(ev)
                    ev.add('perturbation_asymmetry')
                    G[v][u]['evidence'] = ev
                    G[v][u]['mechanism'] = 'fitness_driver_regulation'
                    dynamic_w = self.W['CRISPR'] * min(1.0, abs(ace_v))
                    new_prob = self._aggregate_confidence(G[v][u]['confidence_score'], dynamic_w)
                    G[v][u]['confidence_score'] = round(new_prob, 3)
                    G[v][u]['confidence'] = self._confidence_label(new_prob)

        # 4. Cellular context veto
        cell_map = self.data.get('cell_map', {})
        lr_pairs = self.data.get('lr_pairs', set())
        for u, v in list(G.edges):
            if not G.has_edge(u, v) or G[u][v].get('edge_type') != 'regulatory':
                continue
            c_u = cell_map.get(u, 'unk')
            c_v = cell_map.get(v, 'unk')
            if c_u != 'unk' and c_v != 'unk' and c_u != c_v:
                if (u, v) in lr_pairs:
                    G[u][v]['edge_type'] = 'paracrine_signaling'
                    G[u][v]['mechanism'] = 'ligand_receptor_binding'
                    ev = G[u][v].get('evidence', set())
                    if isinstance(ev, list):
                        ev = set(ev)
                    ev.add('intercellular_communication')
                    G[u][v]['evidence'] = ev
                    G[u][v]['weight'] = G[u][v].get('weight', 0) + 0.2
                    new_prob = self._aggregate_confidence(G[u][v]['confidence_score'], self.W['DATABASE'] * 0.5)
                    G[u][v]['confidence_score'] = round(new_prob, 3)
                    G[u][v]['confidence'] = self._confidence_label(new_prob)
                else:
                    G.remove_edge(u, v)
                    if G.has_edge(v, u):
                        G.remove_edge(v, u)

        # 5. SIGNOR arbitration bonus
        if not self.data.get('signor_edges', pd.DataFrame()).empty:
            for _, row in self.data['signor_edges'].iterrows():
                src, tgt = row['source'], row['target']
                if src == tgt:
                    continue
                if G.has_edge(src, tgt):
                    ev = G[src][tgt].get('evidence', set())
                    if isinstance(ev, list):
                        ev = set(ev)
                    ev.add('signor_physical_interaction')
                    G[src][tgt]['evidence'] = ev
                    G[src][tgt]['mechanism'] = row.get('MECHANISM', G[src][tgt].get('mechanism', ''))
                    new_prob = self._aggregate_confidence(G[src][tgt]['confidence_score'], self.W['SIGNOR'])
                    G[src][tgt]['confidence_score'] = round(new_prob, 3)
                    G[src][tgt]['confidence'] = self._confidence_label(new_prob)
                    if G.has_edge(tgt, src):
                        G.remove_edge(tgt, src)

    def save_outputs(self, consensus_dag: nx.DiGraph, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        output_json = nx.node_link_data(consensus_dag)
        output_json['meta'] = {
            'Disease': self.config.disease_name,
            'version': '2.0_Causal_DAG',
            'module': 'DAGBuilder',
        }
        with open(os.path.join(output_dir, 'consensus_causal_dag.json'), 'w') as f:
            json.dump(output_json, f, indent=2, cls=NpEncoder)

        nodes_df = pd.DataFrame([dict(id=n, **d) for n, d in consensus_dag.nodes(data=True)])
        nodes_df.to_csv(os.path.join(output_dir, 'consensus_dag_nodes.csv'), index=False)

        edges_list = []
        for u, v, d in consensus_dag.edges(data=True):
            edge_dict = {'source': u, 'target': v}
            for key, val in d.items():
                edge_dict[key] = '|'.join(val) if isinstance(val, (list, set)) else val
            edges_list.append(edge_dict)
        edges_df = pd.DataFrame(edges_list)
        edges_df.to_csv(os.path.join(output_dir, 'consensus_dag_edges.csv'), index=False)

        logger.info(f"Saved outputs to {output_dir}")
