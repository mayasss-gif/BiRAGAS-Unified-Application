import networkx as nx
import numpy as np
import pandas as pd
import logging
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format='[DAGEngine] %(message)s')

class AdvancedDAGEngine:
    ACE_DRIVER_THRESHOLD = -0.1
    ACE_EFFECTOR_THRESHOLD = -0.099999999
    ACE_MIN_DELTA = 0.2

    def __init__(self, data_packet, config):
        self.config = config
        self.data = data_packet
        self.trait_node = f"{config.disease_name.replace(' ', '_')}_Disease_Activity"

        self.W = {
            'GWAS': min(0.90, config.WEIGHTS.get('GWAS', 0.90)),
            'MR': min(0.95, config.WEIGHTS.get('MR', 0.95)),
            'CRISPR': min(0.85, config.WEIGHTS.get('CRISPR', 0.85)),
            'SIGNOR': min(0.90, config.WEIGHTS.get('SIGNOR', 0.90)),
            'TEMPORAL': min(0.65, config.WEIGHTS.get('TEMPORAL', 0.65)),
            'STATISTICAL': min(0.50, config.WEIGHTS.get('STATISTICAL', 0.35)),
            'DATABASE': 0.80,
            'EQTL': 0.85,
        }

        self.cohort_expression = self._process_expression()
        self.disease_vector = self._process_metadata()
        self.expr_stats = self._calculate_expression_stats()
        self.global_skeleton = None

    def _aggregate_confidence(self, current_score, new_evidence_weight):
        """Calculates probabilistic union. Capped strictly at 0.99."""
        current_score = max(0.0, min(0.99, current_score))
        new_evidence_weight = max(0.0, min(0.99, new_evidence_weight))
        agg_score = 1.0 - ((1.0 - current_score) * (1.0 - new_evidence_weight))
        return min(0.99, agg_score) 

    def _calculate_confidence_label(self, score):
        if score >= 0.85: return "Very High"
        elif score >= 0.65: return "High"
        elif score >= 0.40: return "Medium"
        else: return "Low"
    # ---------------------------------------------

    def _process_expression(self):
        raw = self.data['raw_counts']
        valid_genes = [g for g in self.data['gene_list'] if g in raw['Gene'].values]
        expr = raw[raw['Gene'].isin(valid_genes)].set_index('Gene').iloc[:, 3:].astype(float)
        
        # Handle duplicate gene names (isoforms/transcripts) by taking their mean
        if expr.index.duplicated().any():
            expr = expr.groupby(level=0).mean()
            
        return np.log1p(expr)

    def _process_metadata(self):
        meta = self.data['metadata']
        samples = self.cohort_expression.columns.tolist()
        meta = meta.set_index('sample_id').reindex(samples)
        return meta['condition'].map({'Disease': 1, 'Control': 0}).fillna(0)

    def _calculate_expression_stats(self):
        meta = self.data['metadata'].set_index('sample_id')
        available = self.cohort_expression.columns 
        dis_samples = [s for s in available if meta.loc[s, 'condition'] == 'Disease']
        ctrl_samples = [s for s in available if meta.loc[s, 'condition'] == 'Control']
        
        stats = {}
        if ctrl_samples: stats['baseline'] = self.cohort_expression[ctrl_samples].mean(axis=1)
        else: stats['baseline'] = pd.Series(0, index=self.cohort_expression.index)
            
        if dis_samples: stats['diseased'] = self.cohort_expression[dis_samples].mean(axis=1)
        else: stats['diseased'] = pd.Series(0, index=self.cohort_expression.index)
        return stats

    def build_consensus_pipeline(self):
        self.global_skeleton = self._build_layered_skeleton()
        patient_dags = []
        samples = self.cohort_expression.columns.tolist()
        logging.info(f"Instantiating personalized DAGs for {len(samples)} patients...")
        
        for sample_id in samples:
            p_dag = self._instantiate_patient_dag(self.global_skeleton, sample_id)
            patient_dags.append(p_dag)
            
        final_dag = self._aggregate_dags(patient_dags)
        return final_dag, patient_dags

    def _build_layered_skeleton(self):
        G = nx.DiGraph()
        
        gwas_snps = set(self.data.get('gwas_snps', []))
        eqtl_snps = set(self.data.get('eqtl_snps', []))
        all_snps = gwas_snps.union(eqtl_snps)
        
        for snp in all_snps:
            source_type = 'GWAS_Catalog' if snp in gwas_snps else 'eQTL_Study'
            if snp in gwas_snps and snp in eqtl_snps: source_type = 'GWAS_and_eQTL'
            G.add_node(snp, layer='source', type='snp', is_anchor=True, evidence_source=source_type)

        coloc_proxy_set = self.data.get('coloc_proxy', set()) 
        lit_scores = self.data.get('literature_scores', {}) 

        for gene in self.data['gene_list']:
            meta = self.data['node_meta'].get(gene, {})
            drugs = self.data['druggability'].get(gene, [])
            gen_conf = self.data.get('genetic_confidence', {}).get(gene, 0)
            pert_meta = self.data.get('perturbation_meta', {}).get(gene, {})
            lit_score = lit_scores.get(gene, 0.0)
            
            base_expr = self.expr_stats['baseline'].get(gene, 0.0)
            dis_expr = self.expr_stats['diseased'].get(gene, 0.0)
            if isinstance(base_expr, pd.Series): base_expr = base_expr.mean()
            if isinstance(dis_expr, pd.Series): dis_expr = dis_expr.mean()
            if isinstance(gen_conf, pd.Series): gen_conf = gen_conf.mean()
            
            ace_score = float(round(pert_meta.get('ACE', 0.0), 4))
            global_essentiality = pert_meta.get('IsEssential', False)
            best_ess_tag = pert_meta.get('BestEssentialityTag', 'Unknown')
            druggable = len(drugs) > 0

            essentialityTag = best_ess_tag if (best_ess_tag and best_ess_tag != 'Unknown') else ("Core Essential" if global_essentiality else "Unknown")
            systemic_tox = "High" if global_essentiality else "Low"
            alignment = pert_meta.get('TherapeuticAlignment', 'Unknown')
            recommendation = "Context-Dependent"
            strategy_type = "Unknown"
            
            if ace_score <= self.ACE_DRIVER_THRESHOLD:
                if alignment == 'Aggravating':
                    recommendation = "Inhibit"
                    strategy_type = "Antagonist / Inhibitor"
                elif alignment == 'Reversal':
                    recommendation = "Activate"
                    strategy_type = "Agonist / Stabilizer"
            
            if systemic_tox == "High": recommendation += " (WARNING: High systemic toxicity risk)"

            is_gwas = (gene in self.data['gwas_roots'])
            has_mr = (gene in self.data.get('mr_evidence', {}))
            has_coloc = (gene in coloc_proxy_set)
            
            evidence_cnt = sum([is_gwas, has_mr, (ace_score <= self.ACE_DRIVER_THRESHOLD), (lit_score > 2.0)])
            
            # tier = "Tier 3 (Candidate)"
            # if is_gwas and (has_mr or has_coloc): tier = "Tier 1 (Validated Driver)"
            # elif is_gwas or has_mr or (ace_score <= self.ACE_DRIVER_THRESHOLD): tier = "Tier 2 (Supported)"
            tier = "Supported"
            if is_gwas or has_mr or (ace_score <= self.ACE_DRIVER_THRESHOLD): tier = "Validated Driver"

            G.add_node(gene, layer='regulatory', type='gene',
                       expression_baseline=float(round(base_expr, 3)),
                       expression_diseased=float(round(dis_expr, 3)),
                       expression_log2fc=float(round(meta.get('Patient_LFC_mean', 0.0), 3)),
                       expression_trend="Up" if dis_expr > base_expr else "Down",
                       is_gwas_hit=is_gwas,
                       genetic_confidence_score=float(gen_conf) if isinstance(gen_conf, (int, float)) else 0.0,
                       perturbation_ace=ace_score,
                       essentiality_tag=essentialityTag,
                       systemic_toxicity_risk=systemic_tox,
                       therapeutic_alignment=alignment,
                       therapeutic_recommendation=recommendation,
                       strategy_type=strategy_type,
                       #druggable=druggable, drug_count=len(drugs),
                       #approved_drugs=drugs[:5], 
                       evidence_count=evidence_cnt,
                       #literature_score=float(round(lit_score, 2)), 
                       causal_tier=tier
            )

        # ==========================================
        # === LAYER 2: PROGRAMS (WITH ONTOLOGY) ====
        # ==========================================
        path_meta = self.data['pathways_df'].drop_duplicates(subset=['Pathway']).set_index('Pathway')[['p_value', 'fdr', 'Regulation', 'Pathway ID']].to_dict('index')
        bioprog_meta = self.data['bioprog_df'].drop_duplicates(subset=['Pathway']).set_index('Pathway')[['p_value', 'fdr', 'Regulation', 'Pathway ID']].to_dict('index')
        pathway_ontology = self.data.get('pathway_ontology', {})

        for p in set([p for sublist in self.data['gene_to_pathway'].values() for p in sublist]):
            info = path_meta.get(p, {})
            ontology = pathway_ontology.get(p, {})
            G.add_node(p, layer='program', type='pathway', source='REACTOME/KEGG', 
                       pathway_id=info.get('Pathway ID', 'NA'), p_value=float(info.get('p_value', 1.0)),
                       fdr=float(info.get('fdr', 1.0)), regulation=info.get('Regulation', 'Unknown'),
                       main_class=ontology.get('Main_Class', 'Unclassified'), sub_class=ontology.get('Sub_Class', 'Unclassified'))
            
        for p in set([p for sublist in self.data['gene_to_bioprog'].values() for p in sublist]):
            info = bioprog_meta.get(p, {})
            ontology = pathway_ontology.get(p, {})
            G.add_node(p, layer='program', type='biological_program', source='GO_BP', 
                       pathway_id=info.get('Pathway ID', 'NA'), p_value=float(info.get('p_value', 1.0)),
                       fdr=float(info.get('fdr', 1.0)), regulation=info.get('Regulation', 'Unknown'),
                       main_class=ontology.get('Main_Class', 'Unclassified'), sub_class=ontology.get('Sub_Class', 'Unclassified'))
            
        for cell in self.data['cell_types']:
            markers = self.data['cell_markers'].get(cell, [])
            if len(markers) > 0:
                G.add_node(cell, layer='program', type='cell_type', source='Bisque_Deconvolution', 
                           marker_count=len(markers), main_class='Cellular Environment', sub_class='Immune Cell Type')

        G.add_node(self.trait_node, layer='trait', type='clinical_outcome', measure='Disease_Activity_Proxy')

        # ==========================================
        # === EDGES: PROBABILISTIC BASE SCORING ====
        # ==========================================
        
        # 1. Transcriptomic Correlation (Statistical) - Now using SETS for evidence
        corr = self.cohort_expression.T.corr()
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                u, v = corr.columns[i], corr.columns[j]
                if u == v: continue
                val = abs(corr.iloc[i,j])
                if val > self.config.CORRELATION_THRESHOLD:
                    scaled_val = ((val - self.config.CORRELATION_THRESHOLD) / (1.0 - self.config.CORRELATION_THRESHOLD))
                    base_prob = min(0.99, scaled_val * self.W['STATISTICAL'])
                    G.add_edge(u, v, weight=val, edge_type='regulatory', mechanism='inferred_regulation', 
                               evidence={'transcriptomic_coexpression'}, confidence_score=round(base_prob, 3), confidence=self._calculate_confidence_label(base_prob))
                    G.add_edge(v, u, weight=val, edge_type='regulatory', mechanism='inferred_regulation', 
                               evidence={'transcriptomic_coexpression'}, confidence_score=round(base_prob, 3), confidence=self._calculate_confidence_label(base_prob))

        for gene, progs in self.data['gene_to_pathway'].items():
            if gene in G.nodes:
                for p in progs: 
                    G.add_edge(gene, p, weight=0.9, edge_type='program_membership', mechanism='pathway_annotation', 
                               evidence={'reactome_kegg_curation'}, confidence_score=self.W['DATABASE'], confidence=self._calculate_confidence_label(self.W['DATABASE']))
        
        for gene, progs in self.data['gene_to_bioprog'].items():
            if gene in G.nodes:
                for p in progs: 
                    score = self.W['DATABASE'] * 0.95 
                    G.add_edge(gene, p, weight=0.9, edge_type='program_membership', mechanism='biological_process_annotation', 
                               evidence={'gene_ontology_annotation'}, confidence_score=round(score, 3), confidence=self._calculate_confidence_label(score))
        
        for cell, markers in self.data['cell_markers'].items():
            if cell in G.nodes:
                for m in markers: 
                    score = self.W['DATABASE'] * 0.90
                    G.add_edge(m, cell, weight=1.0, edge_type='cell_marker', mechanism='cell_identity_definition', 
                               evidence={'transcriptomic_deconvolution_marker'}, confidence_score=round(score, 3), confidence=self._calculate_confidence_label(score))

        return G

    def _instantiate_patient_dag(self, global_graph, sample_id):
        p_dag = global_graph.copy()
        patient_expr = self.cohort_expression[sample_id]
        mean = self.cohort_expression.mean(axis=1)
        std = self.cohort_expression.std(axis=1)
        z_scores = (patient_expr - mean) / std
        if z_scores.index.duplicated().any(): z_scores = z_scores.groupby(level=0).mean()

        program_nodes = [n for n, d in p_dag.nodes(data=True) if d.get('layer') == 'program']
        program_scores = {}
        for prog in program_nodes:
            sources = [u for u, v in p_dag.in_edges(prog)]
            if sources:
                source_z_values = []
                for g in sources:
                    z = z_scores.get(g, 0)
                    if isinstance(z, pd.Series): z = z.mean()
                    source_z_values.append(z)
                score = np.mean(source_z_values)
                program_scores[prog] = score
                p_dag.nodes[prog]['patient_activity'] = score
            else:
                p_dag.nodes[prog]['patient_activity'] = 0.0

        trait_val = self.disease_vector.get(sample_id, 0)
        p_dag.nodes[self.trait_node]['patient_activity'] = trait_val
        active_nodes = set([self.trait_node])
        
        for node in [n for n,d in p_dag.nodes(data=True) if d.get('layer') == 'regulatory']:
            z = z_scores.get(node, 0)
            if isinstance(z, pd.Series): z = z.mean()

            node_data = p_dag.nodes[node]
            is_prior = node_data.get('is_gwas_hit', False) or (node_data.get('perturbation_ace', 0) <= self.ACE_DRIVER_THRESHOLD)

            survival_threshold = 0.75 if is_prior else 1.0

            if abs(z) > survival_threshold:
                active_nodes.add(node)
                p_dag.nodes[node]['patient_activity'] = z
        
        for node in program_scores:
            if abs(program_scores[node]) > 0.5: active_nodes.add(node)
        return p_dag.subgraph(list(active_nodes)).copy()

    def _aggregate_dags(self, patient_dags):
        consensus = nx.DiGraph()
        n_patients = len(patient_dags)
        edge_stats = {}
        
        for dag in patient_dags:
            for u, v, d in dag.edges(data=True):
                edge = (u, v)
                if edge not in edge_stats:
                    edge_stats[edge] = {'count': 0, 'weight_sum': 0, 
                                      'mechanism': d.get('mechanism'),
                                      'edge_type': d.get('edge_type'),
                                      'evidence': set(d.get('evidence', set())),
                                      'confidence_score_sum': 0.0} 
                edge_stats[edge]['count'] += 1
                edge_stats[edge]['weight_sum'] += d.get('weight', 0)
                edge_stats[edge]['confidence_score_sum'] += d.get('confidence_score', 0)
                if 'evidence' in d: edge_stats[edge]['evidence'].update(d['evidence'])

        for (u, v), stats in edge_stats.items():
            freq = stats['count'] / n_patients
            if freq > self.config.CONSENSUS_THRESHOLD:
                base_prob = stats['confidence_score_sum'] / stats['count']
                freq_bonus = freq * self.W['STATISTICAL'] * 0.5
                final_prob = min(0.99, self._aggregate_confidence(base_prob, freq_bonus))
                
                consensus.add_edge(u, v, 
                                   weight=stats['weight_sum']/stats['count'],
                                   frequency=freq,
                                   edge_type=stats['edge_type'],
                                   mechanism=stats['mechanism'],
                                   evidence=stats['evidence'], # Kept as SET
                                   confidence_score=round(final_prob, 3),
                                   confidence=self._calculate_confidence_label(final_prob)
                )

        for node in consensus.nodes:
            if node in self.global_skeleton.nodes:
                consensus.nodes[node].update(self.global_skeleton.nodes[node])

        # === PHENOTYPE ARBITRATION ===
        node_activity_data = []
        disease_data = []
        for dag in patient_dags:
            row = {}
            for node, d in dag.nodes(data=True):
                if d.get('layer') in ['program', 'regulatory']: 
                    row[node] = d.get('patient_activity', 0)
            node_activity_data.append(row)
            disease_data.append(dag.nodes[self.trait_node]['patient_activity'])
            
        df_activity = pd.DataFrame(node_activity_data).fillna(0)
        vec_disease = np.array(disease_data)
        trait_node = self.trait_node
        
        if trait_node not in consensus:
             consensus.add_node(trait_node, layer='trait', type='clinical_outcome', measure='Disease_Activity', description='Disease Activity')
        elif trait_node in self.global_skeleton.nodes:
             consensus.nodes[trait_node].update(self.global_skeleton.nodes[trait_node])
        
        if len(set(vec_disease)) > 1:
            for node in df_activity.columns:
                if node in consensus.nodes:
                    if df_activity[node].std() > 1e-5: 
                        corr, _ = pearsonr(df_activity[node], vec_disease)
                        if abs(corr) > 0.3:
                            prob = min(0.99, abs(corr) * self.W['STATISTICAL'])
                            node_data = consensus.nodes[node]
                            layer = node_data.get('layer')
                            
                            if layer == 'regulatory':
                                is_gwas = node_data.get('is_gwas_hit', False)
                                has_mr = (node in self.data.get('mr_evidence', {}))
                                ace_score = node_data.get('perturbation_ace', 0.0)
                                
                                if is_gwas or has_mr or (ace_score <= self.ACE_DRIVER_THRESHOLD):
                                    consensus.add_edge(node, trait_node,
                                                       weight=abs(corr), edge_type='phenotype_driver', mechanism='causal_disease_driver',
                                                       direction='forward' if corr > 0 else 'inverse', evidence={'transcriptomic_correlation', 'causal_anchor_supported'},
                                                       confidence_score=round(prob, 3), confidence=self._calculate_confidence_label(prob))
                                else:
                                    consensus.add_edge(trait_node, node,
                                                       weight=abs(corr), edge_type='reactive_biomarker', mechanism='disease_induced_expression',
                                                       direction='upregulated_by_disease' if corr > 0 else 'downregulated_by_disease', evidence={'transcriptomic_correlation', 'lacks_causal_anchors'},
                                                       confidence_score=round(prob, 3), confidence=self._calculate_confidence_label(prob))
                                    if consensus.has_edge(node, trait_node): consensus.remove_edge(node, trait_node)
                            
                            elif layer == 'program':
                                causal_incoming = False
                                for predecessor in consensus.predecessors(node):
                                    pred_data = consensus.nodes[predecessor]
                                    # if pred_data.get('causal_tier') in ["Tier 1 (Validated Driver)", "Tier 2 (Supported)"]:
                                    if pred_data.get('causal_tier') in ["Validated Driver"]:
                                        causal_incoming = True
                                        break
                                
                                if causal_incoming:
                                    consensus.add_edge(node, trait_node,
                                                       weight=abs(corr), edge_type='phenotype_driver', mechanism='causal_pathway_module',
                                                       direction='forward' if corr > 0 else 'inverse', evidence={'transcriptomic_correlation', 'driven_by_causal_genes'},
                                                       confidence_score=round(prob, 3), confidence=self._calculate_confidence_label(prob))
                                else:
                                    consensus.add_edge(trait_node, node,
                                                       weight=abs(corr), edge_type='reactive_biomarker', mechanism='disease_associated_pathway',
                                                       direction='upregulated_by_disease' if corr > 0 else 'downregulated_by_disease', evidence={'transcriptomic_correlation'},
                                                       confidence_score=round(prob, 3), confidence=self._calculate_confidence_label(prob))
                                    if consensus.has_edge(node, trait_node): consensus.remove_edge(node, trait_node)

        # === DIRECT INJECTION OF PRIORS (Bypasses Patient Dropout) ===
        for gene, mr_data in self.data.get('mr_evidence', {}).items():
            if gene in consensus.nodes:
                mech = 'genetically_predicted_risk_increase' if mr_data['b'] > 0 else 'genetically_predicted_protection'
                evidence = {'mendelian_randomization_causality'}
                if consensus.has_edge(gene, trait_node):
                    evidence.update(consensus[gene][trait_node].get('evidence', set()))
                
                pval = mr_data.get('pval', 1.0)
                pval_scalar = 1.0 if pval < 0.001 else (0.90 if pval < 0.01 else 0.80)
                final_prob = min(0.99, self.W['MR'] * pval_scalar)
                
                if consensus.has_edge(trait_node, gene): consensus.remove_edge(trait_node, gene)
                
                consensus.add_edge(gene, trait_node, weight=abs(mr_data['b']), edge_type='phenotype_driver', mechanism=mech,
                                   evidence=evidence, pval=mr_data['pval'], confidence_score=round(final_prob, 3), confidence=self._calculate_confidence_label(final_prob))

        for edge in self.data.get('eqtl_edges', []):
            snp, gene = edge['SNP'], edge['Gene_Symbol']
            if gene in consensus.nodes:
                if snp not in consensus:
                    if snp in self.global_skeleton.nodes: consensus.add_node(snp, **self.global_skeleton.nodes[snp])
                    else: consensus.add_node(snp, layer='source', type='snp', is_anchor=True, evidence_source='GWAS_Catalog')
                
                beta_val = abs(edge.get('eQTL_beta', 0))
                coloc_proxy_set = self.data.get('coloc_proxy', set())
                base_prob = min(0.95, self.W['EQTL'] * min(1.0, beta_val))
                if gene in coloc_proxy_set: final_prob = min(0.99, self._aggregate_confidence(base_prob, self.W['GWAS']))
                else: final_prob = base_prob
                
                consensus.add_edge(snp, gene, weight=beta_val, edge_type='genetic_regulation', mechanism='eQTL_expression_modulation',
                                   evidence={'eqtl_study_support'}, confidence_score=round(final_prob, 3), confidence=self._calculate_confidence_label(final_prob))

        # Inject SIGNOR
        if 'signor_edges' in self.data and not self.data['signor_edges'].empty:
            for _, row in self.data['signor_edges'].iterrows():
                src, tgt = row['source'], row['target']
                if src == tgt: continue
                if src in consensus.nodes and tgt in consensus.nodes:
                    if not consensus.has_edge(src, tgt):
                        consensus.add_edge(src, tgt, weight=0.3, edge_type='regulatory', mechanism=row['MECHANISM'], 
                                           evidence={'signor_physical_interaction'}, confidence_score=0.1, confidence=self._calculate_confidence_label(0.1))
                    else:
                        consensus[src][tgt]['evidence'].add('signor_physical_interaction')
                        
        granger = self.data.get('granger_edges', {})
        for (u, v), metrics in granger.items():
            
            # STANDARD INJECTION: q < 0.05 and BOTH nodes survived patient filtering
            if metrics['q_value'] < 0.05:
                if u in consensus.nodes and v in consensus.nodes:
                    if not consensus.has_edge(u, v):
                        base_prob = min(0.99, self.W['STATISTICAL'] * min(1.0, metrics['effect_f'] / 40.0))
                        consensus.add_edge(u, v, weight=0.3, edge_type='regulatory', mechanism='inferred_regulation', 
                                           evidence={'var_granger_predictivity'}, confidence_score=round(base_prob, 3), confidence=self._calculate_confidence_label(base_prob))
                    else:
                        consensus[u][v]['evidence'].add('var_granger_predictivity')
                        
            # ONE-NODE ANCHOR RESCUE: q < 0.01 (Highly Significant) and at least ONE node survived
            if metrics['q_value'] < 0.01:
                # XOR condition: one is in the graph, the other is missing
                if (u in consensus.nodes) ^ (v in consensus.nodes):
                    
                    # Fetch the missing node from the global skeleton and inject it
                    if u not in consensus.nodes and u in self.global_skeleton.nodes:
                        consensus.add_node(u, **self.global_skeleton.nodes[u])
                        consensus.nodes[u]['patient_activity'] = 0.0 # Rescued node
                        
                    if v not in consensus.nodes and v in self.global_skeleton.nodes:
                        consensus.add_node(v, **self.global_skeleton.nodes[v])
                        consensus.nodes[v]['patient_activity'] = 0.0 # Rescued node
                        
                    # Now that both exist, inject the highly significant edge
                    if not consensus.has_edge(u, v):
                        base_prob = min(0.99, self.W['STATISTICAL'] * min(1.0, metrics['effect_f'] / 40.0))
                        consensus.add_edge(u, v, weight=0.3, edge_type='regulatory', mechanism='inferred_regulation', 
                                           evidence={'var_granger_predictivity', 'one_node_anchor_rescue'}, 
                                           confidence_score=round(base_prob, 3), confidence=self._calculate_confidence_label(base_prob))

        # NOW ARBITRATE THE FINAL CONSENSUS GRAPH
        self._apply_advanced_constraints(consensus)

        # Convert SETS to LISTS for clean JSON serialization
        for u, v, d in consensus.edges(data=True):
            if isinstance(d.get('evidence'), set):
                consensus[u][v]['evidence'] = list(d['evidence'])

        return consensus

    def _apply_advanced_constraints(self, G):
        # 1. Genetic Anchors
        roots = set(self.data['gwas_roots'])
        for u, v in list(G.edges):
            if not G.has_edge(u, v): continue
            if G[u][v].get('edge_type') == 'regulatory':
                if u in roots and v not in roots:
                    G[u][v]['evidence'].add('gwas_anchor_downstream_flow')
                    new_prob = self._aggregate_confidence(G[u][v]['confidence_score'], self.W['GWAS'] * 0.8) 
                    G[u][v]['confidence_score'] = round(new_prob, 3)
                    G[u][v]['confidence'] = self._calculate_confidence_label(new_prob)
                    
                    if G.has_edge(v, u): 
                        G[v][u]['weight'] *= 0.5 
                        G[v][u]['mechanism'] = 'potential_feedback_loop'
                        penalized_prob = max(0.1, G[v][u]['confidence_score'] * (1.0 - (self.W['GWAS'] * 0.5)))
                        G[v][u]['confidence_score'] = round(penalized_prob, 3)
                        G[v][u]['confidence'] = self._calculate_confidence_label(penalized_prob)

        # 2. Hybrid Temporal Arbitration
        times = self.data.get('peak_times', {})
        granger = self.data.get('granger_edges', {})
        
        for u, v in list(G.edges):
            if not G.has_edge(u, v): continue
            if G[u][v].get('edge_type') != 'regulatory': continue
            
            has_granger = (u, v) in granger and granger[(u, v)]['q_value'] < 0.05
            has_impulse = (u in times and v in times and times[u] < (times[v] - 0.05))
            
            if has_granger and has_impulse:
                if G.has_edge(v, u): G.remove_edge(v, u) 
                G[u][v]['evidence'].add('spatiotemporal_causality_verified')
                f_stat_scalar = min(1.5, granger[(u, v)]['effect_f'] / 30.0) 
                synergy_weight = self.W['TEMPORAL'] * f_stat_scalar
                new_prob = self._aggregate_confidence(G[u][v]['confidence_score'], synergy_weight)
                G[u][v]['confidence_score'] = round(new_prob, 3)
                G[u][v]['confidence'] = self._calculate_confidence_label(new_prob)
                
            elif has_granger:
                G[u][v]['evidence'].add('var_granger_predictivity')
                f_stat_scalar = min(1.0, granger[(u, v)]['effect_f'] / 40.0)
                granger_weight = self.W['TEMPORAL'] * f_stat_scalar
                new_prob = self._aggregate_confidence(G[u][v]['confidence_score'], granger_weight)
                G[u][v]['confidence_score'] = round(new_prob, 3)
                G[u][v]['confidence'] = self._calculate_confidence_label(new_prob)
                
                if G.has_edge(v, u):
                    penalized_prob = max(0.1, G[v][u]['confidence_score'] * (1.0 - (self.W['TEMPORAL'] * 0.5)))
                    G[v][u]['confidence_score'] = round(penalized_prob, 3)
                    G[v][u]['confidence'] = self._calculate_confidence_label(penalized_prob)

            elif has_impulse:
                if G.has_edge(v, u): G.remove_edge(v, u)
                G[u][v]['evidence'].add('pseudotime_progression')
                new_prob = self._aggregate_confidence(G[u][v]['confidence_score'], self.W['TEMPORAL'] * 0.5)
                G[u][v]['confidence_score'] = round(new_prob, 3)
                G[u][v]['confidence'] = self._calculate_confidence_label(new_prob)

        # 3. Perturbation
        for u, v in list(G.edges):
            if not G.has_edge(u, v): continue
            if G[u][v].get('edge_type') != 'regulatory': continue
            
            ace_u = G.nodes[u].get('perturbation_ace', 0)
            ace_v = G.nodes[v].get('perturbation_ace', 0)
            delta = abs(ace_u - ace_v)
            
            if (ace_u <= self.ACE_DRIVER_THRESHOLD) and (ace_v >= self.ACE_EFFECTOR_THRESHOLD) and (delta >= self.ACE_MIN_DELTA):
                if G.has_edge(v, u): G.remove_edge(v, u)
                G[u][v]['evidence'].add('perturbation_asymmetry')
                G[u][v]['mechanism'] = 'fitness_driver_regulation'
                dynamic_crispr_weight = self.W['CRISPR'] * min(1.0, abs(ace_u)) 
                new_prob = self._aggregate_confidence(G[u][v]['confidence_score'], dynamic_crispr_weight)
                G[u][v]['confidence_score'] = round(new_prob, 3)
                G[u][v]['confidence'] = self._calculate_confidence_label(new_prob)

            elif (ace_v <= self.ACE_DRIVER_THRESHOLD) and (ace_u >= self.ACE_EFFECTOR_THRESHOLD) and (delta >= self.ACE_MIN_DELTA):
                if G.has_edge(u, v): G.remove_edge(u, v)
                if G.has_edge(v, u):
                    G[v][u]['evidence'].add('perturbation_asymmetry')
                    G[v][u]['mechanism'] = 'fitness_driver_regulation'
                    dynamic_crispr_weight = self.W['CRISPR'] * min(1.0, abs(ace_v)) 
                    new_prob = self._aggregate_confidence(G[v][u]['confidence_score'], dynamic_crispr_weight)
                    G[v][u]['confidence_score'] = round(new_prob, 3)
                    G[v][u]['confidence'] = self._calculate_confidence_label(new_prob)

        # 4. Cellular Context
        cell_map = self.data['cell_map']
        lr_pairs = self.data['lr_pairs']
        for u, v in list(G.edges):
            if not G.has_edge(u, v): continue
            if G[u][v].get('edge_type') != 'regulatory': continue
            c_u = cell_map.get(u, 'unk')
            c_v = cell_map.get(v, 'unk')
            if c_u != 'unk' and c_v != 'unk' and c_u != c_v:
                if (u, v) in lr_pairs:
                    G[u][v]['edge_type'] = 'paracrine_signaling'
                    G[u][v]['mechanism'] = "ligand_receptor_binding"
                    G[u][v]['evidence'].add('intercellular_communication')
                    G[u][v]['weight'] += 0.2
                    new_prob = self._aggregate_confidence(G[u][v]['confidence_score'], self.W['DATABASE'] * 0.5)
                    G[u][v]['confidence_score'] = round(new_prob, 3)
                    G[u][v]['confidence'] = self._calculate_confidence_label(new_prob)
                else:
                    if G.has_edge(u, v): G.remove_edge(u, v)
                    if G.has_edge(v, u): G.remove_edge(v, u)

        # 5. SIGNOR Arbitration (Applying bonuses to existing edges)
        if 'signor_edges' in self.data and not self.data['signor_edges'].empty:
            for _, row in self.data['signor_edges'].iterrows():
                src, tgt = row['source'], row['target']
                if src == tgt: continue
                if G.has_edge(src, tgt):
                    G[src][tgt]['evidence'].add('signor_physical_interaction')
                    G[src][tgt]['mechanism'] = row['MECHANISM']
                    new_prob = self._aggregate_confidence(G[src][tgt]['confidence_score'], self.W['SIGNOR'])
                    G[src][tgt]['confidence_score'] = round(new_prob, 3)
                    G[src][tgt]['confidence'] = self._calculate_confidence_label(new_prob)
                    if G.has_edge(tgt, src): G.remove_edge(tgt, src)
