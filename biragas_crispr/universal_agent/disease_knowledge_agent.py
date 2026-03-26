"""
BiRAGAS Disease Knowledge Agent
=================================
Universal disease intelligence system that queries public biomedical databases
to retrieve molecular, genetic, pathway, and clinical data for ANY disease.

Supported Data Sources:
    - OpenTargets Platform API (diseases, targets, evidence)
    - GWAS Catalog REST API (genetic associations)
    - KEGG REST API (pathways)
    - Reactome Content Service (pathway enrichment)
    - STRING DB API (protein interactions)
    - Ensembl REST API (gene/variant data)
"""

import json
import logging
import os
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote, urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger("biragas.disease_knowledge")


class APICache:
    """File-based cache for API responses."""
    def __init__(self, cache_dir=".biragas_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(self, url):
        return hashlib.md5(url.encode()).hexdigest()

    def get(self, url):
        path = os.path.join(self.cache_dir, f"{self._key(url)}.json")
        if os.path.exists(path):
            age_hours = (time.time() - os.path.getmtime(path)) / 3600
            if age_hours < 72:
                try:
                    with open(path) as f:
                        self.hits += 1
                        return json.load(f)
                except Exception:
                    pass
        self.misses += 1
        return None

    def set(self, url, data):
        path = os.path.join(self.cache_dir, f"{self._key(url)}.json")
        try:
            with open(path, 'w') as f:
                json.dump(data, f, default=str)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")


class RobustHTTPClient:
    """HTTP client with retry logic, rate limiting, and error handling."""
    def __init__(self, cache=None):
        self.cache = cache or APICache()
        self.rate_limit_delay = 0.3
        self._last_request_time = 0

    def fetch_json(self, url, headers=None, max_retries=3):
        cached = self.cache.get(url)
        if cached is not None:
            return cached

        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

        for attempt in range(1, max_retries + 1):
            try:
                req = Request(url)
                req.add_header('Accept', 'application/json')
                req.add_header('User-Agent', 'BiRAGAS/2.0 (Ayass Bioscience)')
                if headers:
                    for k, v in headers.items():
                        req.add_header(k, v)

                self._last_request_time = time.time()
                with urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode())
                    self.cache.set(url, data)
                    return data
            except HTTPError as e:
                if e.code == 429:
                    time.sleep(min(2 ** attempt, 30))
                elif e.code == 404:
                    return None
                else:
                    if attempt == max_retries:
                        return None
                    time.sleep(1)
            except Exception as e:
                if attempt == max_retries:
                    return None
                time.sleep(1)
        return None


class DiseaseResolver:
    """Resolves disease names to standardized ontology identifiers."""

    DISEASE_TAXONOMY = {
        "sle": {"efo": "EFO_0002690", "category": "autoimmune_systemic"},
        "systemic lupus erythematosus": {"efo": "EFO_0002690", "category": "autoimmune_systemic"},
        "rheumatoid arthritis": {"efo": "EFO_0000685", "category": "autoimmune_articular"},
        "ra": {"efo": "EFO_0000685", "category": "autoimmune_articular"},
        "type 1 diabetes": {"efo": "EFO_0001359", "category": "autoimmune_endocrine"},
        "type 2 diabetes": {"efo": "EFO_0001360", "category": "metabolic_endocrine"},
        "multiple sclerosis": {"efo": "EFO_0003885", "category": "autoimmune_neurological"},
        "hashimoto's thyroiditis": {"efo": "EFO_0005556", "category": "autoimmune_endocrine"},
        "graves' disease": {"efo": "EFO_0004237", "category": "autoimmune_endocrine"},
        "psoriasis": {"efo": "EFO_0000676", "category": "autoimmune_skin"},
        "crohn's disease": {"efo": "EFO_0000384", "category": "autoimmune_gi"},
        "ulcerative colitis": {"efo": "EFO_0000729", "category": "autoimmune_gi"},
        "ankylosing spondylitis": {"efo": "EFO_0003898", "category": "autoimmune_articular"},
        "sjogren's syndrome": {"efo": "EFO_0009809", "category": "autoimmune_systemic"},
        "antiphospholipid syndrome": {"efo": "EFO_1001462", "category": "autoimmune_vascular"},
        "myasthenia gravis": {"efo": "EFO_0001366", "category": "autoimmune_neurological"},
        "primary biliary cholangitis": {"efo": "EFO_0004230", "category": "autoimmune_hepatic"},
        "celiac disease": {"efo": "EFO_0001060", "category": "autoimmune_gi"},
        "vitiligo": {"efo": "EFO_0004208", "category": "autoimmune_skin"},
        "iga nephropathy": {"efo": "EFO_0004194", "category": "autoimmune_renal"},
        "lupus nephritis": {"efo": "EFO_0004250", "category": "autoimmune_renal"},
        "pancreatic cancer": {"efo": "EFO_0002618", "category": "cancer_gi"},
        "melanoma": {"efo": "EFO_0000389", "category": "cancer_skin"},
        "breast cancer": {"efo": "EFO_0000305", "category": "cancer_breast"},
        "lung cancer": {"efo": "EFO_0001071", "category": "cancer_lung"},
        "colorectal cancer": {"efo": "EFO_0000365", "category": "cancer_gi"},
        "prostate cancer": {"efo": "EFO_0001663", "category": "cancer_urologic"},
        "ovarian cancer": {"efo": "EFO_0001075", "category": "cancer_gynecologic"},
        "glioblastoma": {"efo": "EFO_0000519", "category": "cancer_brain"},
        "hepatocellular carcinoma": {"efo": "EFO_0000182", "category": "cancer_liver"},
        "renal cell carcinoma": {"efo": "EFO_0000681", "category": "cancer_urologic"},
        "acute myeloid leukemia": {"efo": "EFO_0000222", "category": "cancer_hematologic"},
        "obesity": {"efo": "EFO_0001073", "category": "metabolic_endocrine"},
        "nafld": {"efo": "EFO_0004886", "category": "metabolic_hepatic"},
        "gout": {"efo": "EFO_0004267", "category": "metabolic_articular"},
        "coronary artery disease": {"efo": "EFO_0001645", "category": "cardiovascular"},
        "heart failure": {"efo": "EFO_0003144", "category": "cardiovascular"},
        "atrial fibrillation": {"efo": "EFO_0000275", "category": "cardiovascular"},
        "hypertension": {"efo": "EFO_0000537", "category": "cardiovascular"},
        "myocardial infarction": {"efo": "EFO_0000612", "category": "cardiovascular"},
        "stroke": {"efo": "EFO_0000712", "category": "cerebrovascular"},
        "alzheimer's disease": {"efo": "EFO_0000249", "category": "neurodegenerative"},
        "parkinson's disease": {"efo": "EFO_0002508", "category": "neurodegenerative"},
        "epilepsy": {"efo": "EFO_0000474", "category": "neurological"},
        "als": {"efo": "EFO_0000253", "category": "neurodegenerative"},
        "chronic kidney disease": {"efo": "EFO_0003884", "category": "renal_metabolic"},
        "fsgs": {"efo": "EFO_0009301", "category": "renal_glomerular"},
        "copd": {"efo": "EFO_0000341", "category": "pulmonary"},
        "asthma": {"efo": "EFO_0000270", "category": "pulmonary_allergic"},
        "idiopathic pulmonary fibrosis": {"efo": "EFO_0000768", "category": "pulmonary_fibrotic"},
        "covid-19": {"efo": "MONDO_0100096", "category": "infectious_viral"},
        "tuberculosis": {"efo": "EFO_0007445", "category": "infectious_bacterial"},
        "hiv": {"efo": "EFO_0000764", "category": "infectious_viral"},
        "hepatitis b": {"efo": "EFO_0004197", "category": "infectious_viral"},
        "hepatitis c": {"efo": "EFO_0004196", "category": "infectious_viral"},
        "sickle cell disease": {"efo": "EFO_0006831", "category": "hematologic_genetic"},
        "factor v leiden": {"efo": "EFO_0004352", "category": "hematologic_thrombophilia"},
        "allergic rhinitis": {"efo": "EFO_0003785", "category": "allergic"},
        "atopic dermatitis": {"efo": "EFO_0000274", "category": "allergic_skin"},
        "cystic fibrosis": {"efo": "EFO_0000508", "category": "genetic_pulmonary"},
        "huntington's disease": {"efo": "EFO_0000532", "category": "neurodegenerative_genetic"},
    }

    CATEGORY_HIERARCHY = {
        "autoimmune_systemic": {"parent": "autoimmune", "immune_axis": "Th1/Th17/IFN", "organ": "multi-organ"},
        "autoimmune_articular": {"parent": "autoimmune", "immune_axis": "Th1/Th17", "organ": "joint"},
        "autoimmune_endocrine": {"parent": "autoimmune", "immune_axis": "organ-specific", "organ": "endocrine"},
        "autoimmune_neurological": {"parent": "autoimmune", "immune_axis": "Th1/Th17", "organ": "nervous_system"},
        "autoimmune_skin": {"parent": "autoimmune", "immune_axis": "Th17/Th22", "organ": "skin"},
        "autoimmune_gi": {"parent": "autoimmune", "immune_axis": "Th1/Th17", "organ": "gi"},
        "autoimmune_vascular": {"parent": "autoimmune", "immune_axis": "complement/coagulation", "organ": "vascular"},
        "autoimmune_hepatic": {"parent": "autoimmune", "immune_axis": "organ-specific", "organ": "liver"},
        "autoimmune_renal": {"parent": "autoimmune", "immune_axis": "complement/IC", "organ": "kidney"},
        "cancer_gi": {"parent": "cancer", "immune_axis": "immune_evasion", "organ": "gi"},
        "cancer_skin": {"parent": "cancer", "immune_axis": "immune_evasion", "organ": "skin"},
        "cancer_breast": {"parent": "cancer", "immune_axis": "immune_evasion", "organ": "breast"},
        "cancer_lung": {"parent": "cancer", "immune_axis": "immune_evasion", "organ": "lung"},
        "cancer_urologic": {"parent": "cancer", "immune_axis": "immune_evasion", "organ": "urologic"},
        "cancer_gynecologic": {"parent": "cancer", "immune_axis": "immune_evasion", "organ": "gynecologic"},
        "cancer_brain": {"parent": "cancer", "immune_axis": "immune_evasion", "organ": "brain"},
        "cancer_liver": {"parent": "cancer", "immune_axis": "immune_evasion", "organ": "liver"},
        "cancer_hematologic": {"parent": "cancer", "immune_axis": "immune_evasion", "organ": "blood"},
        "metabolic_endocrine": {"parent": "metabolic", "immune_axis": "metabolic_inflammation", "organ": "endocrine"},
        "metabolic_hepatic": {"parent": "metabolic", "immune_axis": "metabolic_inflammation", "organ": "liver"},
        "metabolic_articular": {"parent": "metabolic", "immune_axis": "crystal_inflammation", "organ": "joint"},
        "cardiovascular": {"parent": "cardiovascular", "immune_axis": "vascular_injury", "organ": "heart"},
        "cerebrovascular": {"parent": "cardiovascular", "immune_axis": "vascular_injury", "organ": "brain"},
        "neurodegenerative": {"parent": "neurological", "immune_axis": "neuroinflammation", "organ": "brain"},
        "neurological": {"parent": "neurological", "immune_axis": "neuroimmune", "organ": "nervous_system"},
        "neurodegenerative_genetic": {"parent": "neurological", "immune_axis": "neuroinflammation", "organ": "brain"},
        "renal_metabolic": {"parent": "renal", "immune_axis": "metabolic", "organ": "kidney"},
        "renal_glomerular": {"parent": "renal", "immune_axis": "immune/complement", "organ": "kidney"},
        "pulmonary": {"parent": "pulmonary", "immune_axis": "Th1/Th17", "organ": "lung"},
        "pulmonary_allergic": {"parent": "pulmonary", "immune_axis": "Th2/IgE", "organ": "lung"},
        "pulmonary_fibrotic": {"parent": "pulmonary", "immune_axis": "fibrotic", "organ": "lung"},
        "infectious_viral": {"parent": "infectious", "immune_axis": "antiviral/IFN", "organ": "varies"},
        "infectious_bacterial": {"parent": "infectious", "immune_axis": "Th1/innate", "organ": "varies"},
        "hematologic_genetic": {"parent": "hematologic", "immune_axis": "none", "organ": "blood"},
        "hematologic_thrombophilia": {"parent": "hematologic", "immune_axis": "coagulation", "organ": "vascular"},
        "allergic": {"parent": "allergic", "immune_axis": "Th2/IgE", "organ": "respiratory"},
        "allergic_skin": {"parent": "allergic", "immune_axis": "Th2/IgE", "organ": "skin"},
        "genetic_pulmonary": {"parent": "genetic", "immune_axis": "none", "organ": "lung"},
        "genetic_connective": {"parent": "genetic", "immune_axis": "none", "organ": "connective_tissue"},
    }

    def __init__(self, http_client=None):
        self.http = http_client or RobustHTTPClient()

    def resolve(self, disease_name):
        key = disease_name.lower().strip()
        if key in self.DISEASE_TAXONOMY:
            result = dict(self.DISEASE_TAXONOMY[key])
            result['name'] = disease_name
            result['resolved_from'] = 'local_taxonomy'
            return result
        for db_name, data in self.DISEASE_TAXONOMY.items():
            if key in db_name or db_name in key:
                result = dict(data)
                result['name'] = disease_name
                result['matched_to'] = db_name
                result['resolved_from'] = 'fuzzy_match'
                return result
        return {'name': disease_name, 'efo': None, 'category': 'unknown', 'resolved_from': 'unresolved'}

    def get_category_info(self, category):
        return self.CATEGORY_HIERARCHY.get(category, {"parent": "unknown", "immune_axis": "unknown", "organ": "unknown"})

    def get_all_diseases(self):
        return list(self.DISEASE_TAXONOMY.keys())

    def get_diseases_by_category(self, category):
        return [n for n, d in self.DISEASE_TAXONOMY.items() if d.get('category') == category]

    def get_diseases_by_parent(self, parent):
        cats = [c for c, i in self.CATEGORY_HIERARCHY.items() if i.get('parent') == parent]
        diseases = []
        for c in cats:
            diseases.extend(self.get_diseases_by_category(c))
        return diseases


class NegativeControlGenerator:
    """Generates negative control (differential diagnosis) pairs for any disease."""

    def __init__(self, resolver):
        self.resolver = resolver

    def generate_negatives(self, disease_name, max_pairs=10):
        disease_info = self.resolver.resolve(disease_name)
        category = disease_info.get('category', 'unknown')
        cat_info = self.resolver.CATEGORY_HIERARCHY.get(category, {})
        pairs = []

        # Strategy 1: Same organ, different mechanism
        organ = cat_info.get('organ', '')
        if organ:
            for cat, cd in self.resolver.CATEGORY_HIERARCHY.items():
                if cd.get('organ') == organ and cat != category:
                    for d in self.resolver.get_diseases_by_category(cat)[:2]:
                        pairs.append({'disease_b': d.replace('_', ' ').title(), 'relationship_type': 'same_organ_different_mechanism', 'causal_distinction': f"Same organ ({organ}) different mechanism", 'expected_dag_overlap': 'low'})

        # Strategy 2: Same immune axis, different organ
        axis = cat_info.get('immune_axis', '')
        if axis and axis != 'none':
            for cat, cd in self.resolver.CATEGORY_HIERARCHY.items():
                if cd.get('immune_axis') == axis and cd.get('organ') != organ:
                    for d in self.resolver.get_diseases_by_category(cat)[:1]:
                        pairs.append({'disease_b': d.replace('_', ' ').title(), 'relationship_type': 'same_axis_different_organ', 'causal_distinction': f"Shared axis ({axis}) different organs", 'expected_dag_overlap': 'moderate_upstream'})

        # Strategy 3: Opposite polarity
        opposites = {'Th1/Th17': 'Th2/IgE', 'Th1/Th17/IFN': 'Th2/IgE', 'Th2/IgE': 'Th1/Th17', 'metabolic_inflammation': 'Th1/Th17', 'immune_evasion': 'Th1/Th17'}
        opp = opposites.get(axis)
        if opp:
            for cat, cd in self.resolver.CATEGORY_HIERARCHY.items():
                if cd.get('immune_axis') == opp:
                    for d in self.resolver.get_diseases_by_category(cat)[:2]:
                        pairs.append({'disease_b': d.replace('_', ' ').title(), 'relationship_type': 'opposite_polarity', 'causal_distinction': f"Opposite: {axis} vs {opp}", 'expected_dag_overlap': 'none'})

        # Strategy 4: Cancer vs autoimmune
        parent = cat_info.get('parent', '')
        if parent == 'cancer':
            for d in self.resolver.get_diseases_by_parent('autoimmune')[:2]:
                pairs.append({'disease_b': d.replace('_', ' ').title(), 'relationship_type': 'cancer_vs_autoimmune', 'causal_distinction': 'Immune evasion vs hyperactivation', 'expected_dag_overlap': 'none'})
        elif parent == 'autoimmune':
            for d in self.resolver.get_diseases_by_parent('cancer')[:2]:
                pairs.append({'disease_b': d.replace('_', ' ').title(), 'relationship_type': 'autoimmune_vs_cancer', 'causal_distinction': 'Immune hyperactivation vs evasion', 'expected_dag_overlap': 'none'})

        # Strategy 5: Metabolic vs immune
        if parent == 'metabolic':
            for d in self.resolver.get_diseases_by_parent('autoimmune')[:2]:
                pairs.append({'disease_b': d.replace('_', ' ').title(), 'relationship_type': 'metabolic_vs_immune', 'causal_distinction': 'Metabolic vs autoimmune inflammation', 'expected_dag_overlap': 'low'})

        # Deduplicate
        seen = set()
        unique = []
        for p in pairs:
            k = p['disease_b']
            if k not in seen and k.lower() != disease_name.lower():
                seen.add(k)
                unique.append(p)
        return unique[:max_pairs]


class GeneticFetcher:
    """Fetches GWAS and OpenTargets data."""
    def __init__(self, http_client):
        self.http = http_client

    def fetch_gwas_hits(self, disease_efo, max_results=100):
        if not disease_efo:
            return []
        url = f"https://www.ebi.ac.uk/gwas/rest/api/efoTraits/{disease_efo}/associations?projection=associationByEfoTrait"
        data = self.http.fetch_json(url)
        if not data:
            return []
        hits = []
        for assoc in data.get('_embedded', {}).get('associations', [])[:max_results]:
            for locus in assoc.get('loci', []):
                for gene in locus.get('authorReportedGenes', []):
                    hits.append({
                        'gene': gene.get('geneName', ''),
                        'snp': assoc.get('snps', [{}])[0].get('rsId', '') if assoc.get('snps') else '',
                        'p_value': assoc.get('pvalue', 1.0),
                        'odds_ratio': assoc.get('orPerCopyNum'),
                    })
        return [h for h in hits if h['gene']]

    def fetch_opentargets_associations(self, disease_efo, max_results=50):
        if not disease_efo:
            return []
        url = f"https://api.platform.opentargets.org/api/v4/disease/{disease_efo}/associations/targets?size={max_results}"
        data = self.http.fetch_json(url)
        if not data or 'data' not in data:
            return []
        return [{'gene': r.get('target', {}).get('approvedSymbol', ''), 'score': r.get('score', 0)} for r in data.get('data', [])]


class PathwayFetcher:
    """Fetches pathway data from Reactome."""
    def __init__(self, http_client):
        self.http = http_client

    def fetch_reactome_pathways(self, gene_list):
        pathways = []
        for gene in gene_list[:15]:
            url = f"https://reactome.org/ContentService/search/query?query={gene}&types=Pathway&cluster=true"
            data = self.http.fetch_json(url)
            if data and 'results' in data:
                for r in data['results'][:2]:
                    for e in r.get('entries', [])[:2]:
                        pathways.append({'gene': gene, 'pathway': e.get('name', ''), 'pathway_id': e.get('stId', ''), 'source': 'REACTOME'})
        return pathways


class InteractionFetcher:
    """Fetches protein interactions from STRING."""
    def __init__(self, http_client):
        self.http = http_client

    def fetch_string_interactions(self, gene_list, min_score=700):
        if not gene_list:
            return []
        proteins = "%0d".join(gene_list[:50])
        url = f"https://string-db.org/api/json/network?identifiers={proteins}&species=9606&required_score={min_score}"
        data = self.http.fetch_json(url)
        if not data or not isinstance(data, list):
            return []
        return [{'source': i.get('preferredName_A', ''), 'target': i.get('preferredName_B', ''), 'score': i.get('score', 0)} for i in data]


class DiseaseKnowledgeAgent:
    """
    Universal disease intelligence agent.
    Given ANY disease name, gathers genetic, pathway, and interaction data.
    """
    def __init__(self, cache_dir=".biragas_cache"):
        self.cache = APICache(cache_dir)
        self.http = RobustHTTPClient(self.cache)
        self.resolver = DiseaseResolver(self.http)
        self.negative_gen = NegativeControlGenerator(self.resolver)
        self.genetic_fetcher = GeneticFetcher(self.http)
        self.pathway_fetcher = PathwayFetcher(self.http)
        self.interaction_fetcher = InteractionFetcher(self.http)
        logger.info("DiseaseKnowledgeAgent initialized")

    def gather_disease_data(self, disease_name):
        logger.info(f"Gathering data for: {disease_name}")
        disease_info = self.resolver.resolve(disease_name)
        efo_id = disease_info.get('efo', '')
        logger.info(f"  Resolved: {disease_info.get('resolved_from')} -> {efo_id}")

        gwas_hits = self.genetic_fetcher.fetch_gwas_hits(efo_id)
        ot_assocs = self.genetic_fetcher.fetch_opentargets_associations(efo_id)
        all_genes = list(set([h['gene'] for h in gwas_hits if h.get('gene')] + [a['gene'] for a in ot_assocs if a.get('gene')]))
        logger.info(f"  GWAS: {len(gwas_hits)}, OpenTargets: {len(ot_assocs)}, Genes: {len(all_genes)}")

        pathways = self.pathway_fetcher.fetch_reactome_pathways(all_genes[:15])
        interactions = self.interaction_fetcher.fetch_string_interactions(all_genes[:30])
        logger.info(f"  Pathways: {len(pathways)}, Interactions: {len(interactions)}")

        return {
            'disease_name': disease_name, 'disease_info': disease_info,
            'gwas_hits': gwas_hits, 'opentargets_associations': ot_assocs,
            'all_genes': all_genes, 'reactome_pathways': pathways,
            'string_interactions': interactions, 'timestamp': time.time(),
        }

    def generate_stress_test_scenarios(self, disease_name, max_scenarios=10):
        negatives = self.negative_gen.generate_negatives(disease_name, max_pairs=max_scenarios)
        return [{'scenario_id': i, 'disease_a': disease_name, 'disease_b': n['disease_b'],
                 'relationship_type': n['relationship_type'], 'causal_distinction': n['causal_distinction']}
                for i, n in enumerate(negatives, 1)]

    def get_supported_diseases(self):
        return self.resolver.get_all_diseases()

    def get_disease_categories(self):
        categories = {}
        for disease, info in self.resolver.DISEASE_TAXONOMY.items():
            cat = info.get('category', 'unknown')
            parent = self.resolver.CATEGORY_HIERARCHY.get(cat, {}).get('parent', 'other')
            categories.setdefault(parent, []).append(disease)
        return categories
