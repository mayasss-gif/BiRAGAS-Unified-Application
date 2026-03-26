import pandas as pd
import numpy as np
import os
import logging
import re

logging.basicConfig(level=logging.INFO, format='[DataLoader] %(message)s')

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data = {}
        self.synonym_map = {}
        self.ensembl_map = {} 

    def load_all(self):
        self._load_transcriptomics() 
        self._load_pathways()        
        self._load_context()         
        self._load_perturbation()    
        self._load_genetics()        
        return self.data

    def _parse_and_resolve(self, raw_string):
        if pd.isna(raw_string) or raw_string == "":
            return []
            
        raw_string = str(raw_string)
        cleaned_string = re.sub(r'(\s+-\s+)|;', ',', raw_string)
        
        resolved_genes = []
        for g in cleaned_string.split(','):
            g = g.strip().upper()
            if not g: continue
                
            base_g = g.split('.')[0] if g.startswith('ENSG') else g
                
            if base_g in self.ensembl_map:
                g = self.ensembl_map[base_g]
            elif g in self.synonym_map:
                g = self.synonym_map[g]
            
            if g.startswith('ENSG'):
                continue
                
            resolved_genes.append(g)
            
        return resolved_genes

    def _load_transcriptomics(self):
        self.data['candidates_df'] = pd.read_csv(self.config.DEGS_PRIORITY)
        self.data['candidates_df']['Gene'] = self.data['candidates_df']['Gene'].str.upper()
        
        if 'HGNC_Synonyms' in self.data['candidates_df'].columns:
            for _, row in self.data['candidates_df'].iterrows():
                canonical = row['Gene']
                if pd.notna(row['HGNC_Synonyms']):
                    syns = str(row['HGNC_Synonyms']).replace(';', ',').split(',')
                    for s in syns:
                        clean_s = s.strip().upper()
                        if clean_s and clean_s != canonical:
                            self.synonym_map[clean_s] = canonical

        deg_genes = set(self.data['candidates_df']['Gene'].unique())
        
        self.data['literature_scores'] = {}
        lit_cols = [c for c in self.data['candidates_df'].columns if 'pubmed' in c.lower() or 'literature' in c.lower()]
        if lit_cols:
            col = lit_cols[0]
            self.data['candidates_df']['lit_score'] = np.log1p(self.data['candidates_df'][col].fillna(0))
            self.data['literature_scores'] = self.data['candidates_df'].set_index('Gene')['lit_score'].to_dict()

        self.data['raw_counts'] = pd.read_csv(self.config.RAW_COUNTS, sep='\t')
        self.data['metadata'] = pd.read_csv(self.config.METADATA)
        
        ens_col = next((c for c in self.data['raw_counts'].columns if 'ens' in c.lower()), None)
        if ens_col and 'Gene' in self.data['raw_counts'].columns:
            for _, row in self.data['raw_counts'].iterrows():
                ens_val = str(row[ens_col]).strip().upper()
                sym_val = str(row['Gene']).strip().upper()
                
                if ens_val.startswith('ENSG') and sym_val != 'NAN':
                    base_ens = ens_val.split('.')[0]
                    self.ensembl_map[ens_val] = sym_val
                    self.ensembl_map[base_ens] = sym_val
                

        genetic_genes = set()
        
        try:
            gwas = pd.read_excel(self.config.GWAS_ASSOC)
            if 'P-VALUE' in gwas.columns and 'MAPPED_GENE' in gwas.columns:
                gwas_hits = gwas[gwas['P-VALUE'] < 5e-8]['MAPPED_GENE'].dropna().unique()
                for g_raw in gwas_hits:
                    genetic_genes.update(self._parse_and_resolve(g_raw))
        except Exception as e:
            logging.warning(f"Could not expand list with GWAS: {e}")

        try:
            mr = pd.read_csv(self.config.MR_RESULTS)
            if 'pval' in mr.columns:
                col = 'gene' if 'gene' in mr.columns else 'exposure'
                if col in mr.columns:
                    mr_hits = mr[mr['pval'] < 0.05][col].dropna().unique()
                    for g_raw in mr_hits:
                        genetic_genes.update(self._parse_and_resolve(g_raw))
        except Exception as e:
            logging.warning(f"Could not expand list with MR: {e}")

        expanded_set = deg_genes.union(genetic_genes)  
        expanded_df = pd.DataFrame(list(expanded_set), columns=['Gene'])
        expanded_path = os.path.join(os.path.dirname(self.config.RAW_COUNTS), "expanded_gene_set.csv")
        expanded_df.to_csv(expanded_path, index=False)
        logging.info(f"Saved expanded gene set to: {expanded_path} ({len(expanded_df)} genes)")
    
        available_genes = set(self.data['raw_counts']['Gene'].str.upper().unique())
        final_list = list(expanded_set.intersection(available_genes))

        self.data['gene_list'] = final_list
        self.data['node_meta'] = self.data['candidates_df'].set_index('Gene').to_dict(orient='index')
        
        logging.info(f"Loaded {len(self.data['gene_list'])} Genes (Union of {len(deg_genes)} DEGs + Cleaned Genetics).")

    def _load_pathways(self):
        path = pd.read_csv(self.config.PATHWAYS)
        
        self.data['pathways_df'] = path[path['DB_ID'].isin(['REACTOME', 'KEGG', 'WIKIPATHWAY'])]
        self.data['bioprog_df'] = path[path['DB_ID'] == 'GO_BP']
        
        self.data['gene_to_pathway'] = self._map_genes_to_programs(self.data['pathways_df'])
        self.data['gene_to_bioprog'] = self._map_genes_to_programs(self.data['bioprog_df'])
        
        self.data['pathway_ontology'] = {}
        for _, row in path.iterrows():
            p_name = row['Pathway']
            self.data['pathway_ontology'][p_name] = {
                'Main_Class': row.get('Main_Class', 'Unclassified'),
                'Sub_Class': row.get('Sub_Class', 'Unclassified')
            }

    def _map_genes_to_programs(self, df):
        mapping = {}
        for _, row in df.iterrows():
            if 'Pathway associated genes' not in row or pd.isna(row['Pathway associated genes']): 
                continue
            
            genes_raw = str(row['Pathway associated genes'])
            resolved_genes = self._parse_and_resolve(genes_raw)
            
            for g in resolved_genes:
                if g in self.data['gene_list']:
                    if g not in mapping: mapping[g] = []
                    mapping[g].append(row['Pathway'])
        return mapping

    def _load_context(self):
        try:
            sig = pd.read_csv(self.config.DECONVOLUTION, sep='\t')
            if 'gene' in sig.columns:
                sig['gene'] = sig['gene'].apply(lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
                sig = sig.dropna(subset=['gene'])
                sig = sig[sig['gene'].isin(self.data['gene_list'])]
            
            if not sig.empty:
                sig['Dominant'] = sig.iloc[:, 1:].idxmax(axis=1)
                self.data['cell_map'] = sig.set_index('gene')['Dominant'].to_dict()
                
                self.data['cell_types'] = [c for c in sig.columns if c not in ['gene', 'Dominant']]
                self.data['cell_markers'] = {}
                for cell in self.data['cell_types']:
                    markers = sig.nlargest(100, cell)['gene'].tolist()
                    valid_markers = [m for m in markers if m in self.data['gene_list']]
                    self.data['cell_markers'][cell] = valid_markers
            else:
                self.data['cell_map'], self.data['cell_types'], self.data['cell_markers'] = {}, [], {}
        except: 
            self.data['cell_map'], self.data['cell_types'], self.data['cell_markers'] = {}, [], {}

        self.data['peak_times'] = {}
        self.data['granger_edges'] = {}
        
        try:
            temp = pd.read_csv(self.config.TEMPORAL_FITS, sep='\t')
            temp['gene_id'] = temp['gene_id'].apply(lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
            temp = temp.dropna(subset=['gene_id'])
            temp = temp[temp['gene_id'].isin(self.data['gene_list'])]
            self.data['peak_times'] = temp.set_index('gene_id')['time_of_peak'].to_dict()
        except Exception as e: 
            logging.warning(f"Could not load Temporal Fits: {e}")

        try:
            granger = pd.read_csv(self.config.GRANGER_EDGES)
            granger['source'] = granger['source'].apply(lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
            granger['target'] = granger['target'].apply(lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
            granger = granger.dropna(subset=['source', 'target'])
            
            valid_granger = granger[(granger['source'].isin(self.data['gene_list'])) & 
                                    (granger['target'].isin(self.data['gene_list']))]
            
            for _, row in valid_granger.iterrows():
                src = row['source']
                tgt = row['target']
                
                if (src, tgt) not in self.data['granger_edges'] or row['effect_f'] > self.data['granger_edges'][(src, tgt)]['effect_f']:
                    self.data['granger_edges'][(src, tgt)] = {
                        'q_value': row['q_value'],
                        'p_value': row['p_value'],
                        'effect_f': row['effect_f'],
                        'lag': row['lag']
                    }
        except Exception as e:
            logging.warning(f"Could not load Granger Edges: {e}")

    def _load_perturbation(self):
        self.data['perturbation_meta'] = {}
        
        try:
            drivers = pd.read_csv(self.config.CAUSAL_DRIVERS)
            for _, row in drivers.iterrows():
                resolved_list = self._parse_and_resolve(row['gene'])
                for g in resolved_list:
                    if g in self.data['gene_list']:
                        self.data['perturbation_meta'][g] = {
                            'ACE': row['ACE'],
                            'TherapeuticAlignment': row['TherapeuticAlignment'],
                            'Verdict': row['Verdict'],
                            'BestEssentialityTag': row.get('BestEssentialityTag', 'Unknown')
                        }
        except Exception as e: logging.warning(f"File Load Error: {e}")

        try:
            ess = pd.read_csv(self.config.ESSENTIALITY)
            for _, row in ess.iterrows():
                resolved_list = self._parse_and_resolve(row['Gene'])
                for g in resolved_list:
                    if g in self.data['gene_list']:
                        if g not in self.data['perturbation_meta']: 
                            self.data['perturbation_meta'][g] = {'ACE': 0.0, 'BestEssentialityTag': 'Unknown'}
                        self.data['perturbation_meta'][g]['IsEssential'] = row['IsEssential_byMedianRule']
        except: pass

        try:
            drugs = pd.read_csv(self.config.DRUG_LINKS)
            drugs['Gene'] = drugs['Gene'].apply(lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
            drugs = drugs.dropna(subset=['Gene'])
            
            valid_drugs = drugs[drugs['Gene'].isin(self.data['gene_list'])]
            if not valid_drugs.empty:
                drug_map = valid_drugs.groupby('Gene').apply(lambda x: x[['Drug', 'Therapeutic_Relevance']].to_dict('records')).to_dict()
                self.data['druggability'] = drug_map
            else:
                self.data['druggability'] = {}
        except: self.data['druggability'] = {}

        try:
            crispr = pd.read_csv(self.config.PERTURBATION)
            raw_impact = crispr.groupby('Gene')['GuideLFC'].median().to_dict()
            for g_raw, impact in raw_impact.items():
                resolved_list = self._parse_and_resolve(g_raw)
                for g in resolved_list:
                    if g in self.data['gene_list']:
                        if g not in self.data['perturbation_meta']:
                            self.data['perturbation_meta'][g] = {'ACE': impact, 'BestEssentialityTag': 'Unknown'}
        except: pass

    def _load_genetics(self):
        try:
            gwas = pd.read_excel(self.config.GWAS_ASSOC)
            sig_gwas = gwas[gwas['P-VALUE'] < 5e-8].copy()
            
            valid_roots = set()
            valid_snps = set()
            
            for _, row in sig_gwas.iterrows():
                resolved_genes = self._parse_and_resolve(row['MAPPED_GENE'])
                for g in resolved_genes:
                    if g in self.data['gene_list']:
                        valid_roots.add(g)
                        valid_snps.add(row['SNPS'])
            
            self.data['gwas_roots'] = list(valid_roots)
            self.data['gwas_snps'] = list(valid_snps)
        except Exception as e:
            self.data['gwas_roots'] = []
            self.data['gwas_snps'] = []

        try:
            sig = pd.read_csv(self.config.SIGNOR_EDGES, sep='\t')
            sig['source'] = sig['source'].apply(lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
            sig['target'] = sig['target'].apply(lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
            sig = sig.dropna(subset=['source', 'target'])
            
            self.data['signor_edges'] = sig[
                (sig['source'].isin(self.data['gene_list'])) & 
                (sig['target'].isin(self.data['gene_list']))
            ]
            lr = self.data['signor_edges'][self.data['signor_edges']['MECHANISM'].isin(['binding', 'receptor signaling'])]
            self.data['lr_pairs'] = set(zip(lr['source'], lr['target']))
        except:
            self.data['signor_edges'] = pd.DataFrame()
            self.data['lr_pairs'] = set()

        try:
            gene_ev = pd.read_csv(self.config.GENE_EVIDENCE, sep='\t')
            gene_ev['Gene_Symbol'] = gene_ev['Gene_Symbol'].apply(lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
            gene_ev = gene_ev.dropna(subset=['Gene_Symbol'])
            gene_ev = gene_ev[gene_ev['Gene_Symbol'].isin(self.data['gene_list'])]
            self.data['genetic_confidence'] = gene_ev.set_index('Gene_Symbol')['Gene_Genetic_Confidence_Score'].to_dict()
        except: self.data['genetic_confidence'] = {}

        try:
            var_ev = pd.read_csv(self.config.VARIANT_EVIDENCE, sep='\t')
            var_ev['Gene_Symbol'] = var_ev['Gene_Symbol'].apply(lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
            var_ev = var_ev.dropna(subset=['Gene_Symbol'])
            var_ev = var_ev[var_ev['Gene_Symbol'].isin(self.data['gene_list'])]
            
            self.data['eqtl_edges'] = var_ev[['SNP', 'Gene_Symbol', 'eQTL_beta']].to_dict(orient='records')
            self.data['eqtl_snps'] = var_ev['SNP'].unique().tolist()
        except: 
            self.data['eqtl_edges'] = []
            self.data['eqtl_snps'] = []

        try:
            mr_res = pd.read_csv(self.config.MR_RESULTS)
            col = 'gene' if 'gene' in mr_res.columns else 'exposure'
            mr_res[col] = mr_res[col].apply(lambda x: self._parse_and_resolve(x)[0] if self._parse_and_resolve(x) else None)
            mr_res = mr_res.dropna(subset=[col])
            mr_res = mr_res[mr_res[col].isin(self.data['gene_list'])]
            
            self.data['mr_evidence'] = mr_res.set_index(col)[['b', 'pval', 'direction']].to_dict(orient='index')
        except: self.data['mr_evidence'] = {}

        self.data['coloc_proxy'] = set(self.data['gwas_roots']).intersection(
            set([e['Gene_Symbol'] for e in self.data['eqtl_edges']])
        )
