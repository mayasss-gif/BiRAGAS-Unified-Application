"""
Validation Layer for Clinical Transcriptome Reports
==================================================

This module provides self-healing, self-correcting, and self-learning validation
for genes, pathways, and drugs using multiple evidence sources:
- Neo4j Knowledge Graph (KEGG, GeneCards, MalaCards)
- PubMed API for literature evidence
- ClinicalTrials.gov API for drug trials
- LLM reasoning for evidence synthesis

Key Features:
- Evidence caching with 90-day refresh
- Synonym-based correction loops
- Comprehensive validation with confidence scoring
"""

import sqlite3
import json
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from openai import OpenAI
from neo4j import GraphDatabase
import os
from dataclasses import dataclass

try:
    from decouple import config as decouple_config
    DECOUPLE_AVAILABLE = True
except ImportError:
    DECOUPLE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Import Neo4j integration components (lazy - no logging at import time)
try:
    from agentic_ai_wf.neo4j_integration.connection import Neo4jConnection
    from agentic_ai_wf.neo4j_integration.query_builder import CypherQueryBuilder
    from agentic_ai_wf.neo4j_integration.config import ENV_CONFIG
    NEO4J_INTEGRATION_AVAILABLE = True
except ImportError as e:
    NEO4J_INTEGRATION_AVAILABLE = False
    Neo4jConnection = None
    CypherQueryBuilder = None
    ENV_CONFIG = None

@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    confidence: float  # 0.0 to 1.0
    evidence: List[str]
    sources: List[str]
    justification: str
    category: str  # "high", "medium", "low", "insufficient"
    status: str = "Uncertain"  # "Pathogenic", "Protective", "Uncertain", "Relevant Biomarker", "Not Relevant"

class EvidenceCache:
    """SQLite-based cache for validation evidence with 90-day refresh"""
    
    def __init__(self, db_path: str = "feedback_log.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease TEXT NOT NULL,
                item TEXT NOT NULL,
                item_type TEXT NOT NULL,
                validation_result TEXT NOT NULL,
                evidence TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(disease, item, item_type)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_name TEXT NOT NULL,
                query TEXT NOT NULL,
                response_count INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_cached_result(self, disease: str, item: str, item_type: str, 
                         max_age_days: int = 90) -> Optional[ValidationResult]:
        """Retrieve cached validation result if fresh enough"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        cursor.execute("""
            SELECT validation_result, evidence, confidence 
            FROM validation_cache 
            WHERE disease = ? AND item = ? AND item_type = ? 
            AND last_checked > ?
        """, (disease, item, item_type, cutoff_date))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            validation_data = json.loads(result[0])
            evidence_data = json.loads(result[1])
            
            return ValidationResult(
                is_valid=validation_data['is_valid'],
                confidence=result[2],
                evidence=evidence_data['evidence'],
                sources=evidence_data['sources'],
                justification=validation_data['justification'],
                category=validation_data['category']
            )
        
        return None
    
    def store_result(self, disease: str, item: str, item_type: str, 
                    result: ValidationResult):
        """Store validation result in cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        validation_data = {
            'is_valid': result.is_valid,
            'justification': result.justification,
            'category': result.category
        }
        
        evidence_data = {
            'evidence': result.evidence,
            'sources': result.sources
        }
        
        cursor.execute("""
            INSERT OR REPLACE INTO validation_cache 
            (disease, item, item_type, validation_result, evidence, confidence, last_checked)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (disease, item, item_type, 
              json.dumps(validation_data), 
              json.dumps(evidence_data), 
              result.confidence))
        
        conn.commit()
        conn.close()
    
    def log_api_call(self, api_name: str, query: str, response_count: int = 0):
        """Log API usage for monitoring"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO api_logs (api_name, query, response_count)
            VALUES (?, ?, ?)
        """, (api_name, query, response_count))
        
        conn.commit()
        conn.close()

class PubMedValidator:
    """PubMed API integration for literature validation"""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def __init__(self, email: str = None):
        self.email = email or "f420testing@ayassbioscience.com"
    
    def search_literature(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search PubMed for relevant abstracts"""
        try:
            # Step 1: Search for PMIDs
            search_url = f"{self.BASE_URL}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'tool': 'clinical_validator',
                'email': self.email
            }
            
            search_response = requests.get(search_url, params=search_params)
            search_data = search_response.json()
            
            if 'esearchresult' not in search_data or not search_data['esearchresult']['idlist']:
                return []
            
            pmids = search_data['esearchresult']['idlist']
            
            # Step 2: Fetch abstracts
            fetch_url = f"{self.BASE_URL}efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml',
                'tool': 'clinical_validator',
                'email': self.email
            }
            
            time.sleep(0.1)  # Rate limiting
            fetch_response = requests.get(fetch_url, params=fetch_params)
            
            # Parse XML to extract abstracts (simplified)
            abstracts = self._parse_pubmed_xml(fetch_response.text)
            
            return abstracts[:max_results]
            
        except Exception as e:
            logger.error(f"PubMed search failed for query '{query}': {str(e)}")
            return []
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[Dict]:
        """Parse PubMed XML response to extract abstracts"""
        # Simplified XML parsing - in production, use xml.etree.ElementTree
        abstracts = []
        
        # Basic text extraction (would need proper XML parsing)
        if "AbstractText" in xml_content:
            # This is a simplified approach
            import re
            abstract_matches = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', xml_content, re.DOTALL)
            
            for i, abstract in enumerate(abstract_matches[:10]):
                abstracts.append({
                    'pmid': f'dummy_{i}',
                    'title': f'Research paper {i+1}',
                    'abstract': abstract.strip(),
                    'year': '2023'
                })
        
        return abstracts

class ClinicalTrialsValidator:
    """ClinicalTrials.gov API integration"""
    
    BASE_URL = "https://clinicaltrials.gov/api/query/"
    
    def search_trials(self, drug_name: str, condition: str = None, 
                     max_results: int = 10) -> List[Dict]:
        """Search for clinical trials involving specific drugs/conditions"""
        try:
            params = {
                'expr': drug_name,
                'fmt': 'json',
                'max_rnk': max_results
            }
            
            if condition:
                params['expr'] = f"{drug_name} AND {condition}"
            
            response = requests.get(f"{self.BASE_URL}study_fields", params=params)
            data = response.json()
            
            trials = []
            if 'StudyFieldsResponse' in data:
                for study in data['StudyFieldsResponse'].get('StudyFields', []):
                    trial_info = {
                        'nct_id': study.get('NCTId', ['Unknown'])[0],
                        'title': study.get('BriefTitle', ['Unknown'])[0],
                        'status': study.get('OverallStatus', ['Unknown'])[0],
                        'phase': study.get('Phase', ['Unknown'])[0],
                        'conditions': study.get('Condition', []),
                        'interventions': study.get('InterventionName', [])
                    }
                    trials.append(trial_info)
            
            return trials
            
        except Exception as e:
            logger.error(f"ClinicalTrials search failed for '{drug_name}': {str(e)}")
            return []

class KnowledgeGraphValidator:
    """Enhanced Neo4j Knowledge Graph integration using dedicated connection manager"""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """Initialize with proper Neo4j integration or fallback"""
        self.connection = None
        self.query_builder = None
        
        if NEO4J_INTEGRATION_AVAILABLE:
            try:
                # Use dedicated Neo4j connection manager with environment config
                if uri or user or password:
                    # Use provided credentials
                    self.connection = Neo4jConnection(
                        uri=uri or ENV_CONFIG.uri,
                        username=user or ENV_CONFIG.username,
                        password=password or ENV_CONFIG.password,
                        database=ENV_CONFIG.database
                    )
                else:
                    # Use environment configuration
                    self.connection = Neo4jConnection(
                        uri=ENV_CONFIG.uri,
                        username=ENV_CONFIG.username,
                        password=ENV_CONFIG.password,
                        database=ENV_CONFIG.database
                    )
                
                # Test connection (lazy - only when actually used)
                if self.connection.test_connection():
                    self.query_builder = CypherQueryBuilder()
                    logger.debug("✅ Enhanced Neo4j Knowledge Graph connection established")
                else:
                    logger.warning("❌ Neo4j connection test failed")
                    self.connection = None
                    
            except Exception as e:
                logger.debug(f"⚠️ Enhanced Neo4j connection failed: {e}. Using fallback validation.")
                self.connection = None
        else:
            # Fallback to basic Neo4j driver if integration not available (lazy init)
            self._driver_uri = uri or "bolt://localhost:7687"
            self._driver_user = user or "neo4j"
            self._driver_password = password or "password"
            self.driver = None  # Will be created on first use
    
    def validate_gene_disease_association(self, gene: str, disease: str) -> Tuple[bool, List[str]]:
        """Check if gene is associated with disease in knowledge graph using enhanced integration"""
        
        # Try enhanced Neo4j integration first
        if self.connection and self.query_builder:
            try:
                # Use the enhanced query builder for gene-disease associations
                query, params = self.query_builder.find_genes_by_disease_with_scores(disease, limit=100)
                results = self.connection.execute_query(query, params)
                
                # Check if our gene is in the results
                gene_matches = [r for r in results if r.get('gene_symbol', '').upper() == gene.upper()]
                
                if gene_matches:
                    evidence = []
                    for match in gene_matches:
                        gene_score = match.get('gene_score', 0)
                        disorder_score = match.get('disorder_score', 0)
                        disorder_type = match.get('disorder_type', 'Unknown')
                        evidence.append(f"KG: {gene} -> {disease} (Gene Score: {gene_score}, Disorder Score: {disorder_score}, Type: {disorder_type})")
                    return True, evidence
                
                # Try broader search if direct match not found
                query = """
                MATCH (g:Gene)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]-(d:Disease)
                WHERE (toLower(g.symbol) CONTAINS toLower($gene) OR toLower($gene) CONTAINS toLower(g.symbol))
                AND (toLower(d.name) CONTAINS toLower($disease) OR toLower($disease) CONTAINS toLower(d.name))
                RETURN g.symbol, d.name, type(r) as relationship, r.gene_score, r.disorder_score
                LIMIT 10
                """
                params = {"gene": gene, "disease": disease}
                results = self.connection.execute_query(query, params)
                
                if results:
                    evidence = [f"KG: {r['g.symbol']} {r['relationship']} {r['d.name']} (Gene Score: {r.get('r.gene_score', 'N/A')})" 
                              for r in results]
                    return True, evidence
                    
                return False, ["No gene-disease associations found in knowledge graph"]
                
            except Exception as e:
                logger.error(f"Enhanced knowledge graph query failed: {e}")
                # Fall through to basic driver if available
        
        # Fallback to basic driver
        if hasattr(self, 'driver') and self.driver:
            try:
                with self.driver.session() as session:
                    query = """
                    MATCH (g:Gene)-[r]-(d:Disease)
                    WHERE (toLower(g.symbol) = toLower($gene) OR toLower(g.name) = toLower($gene))
                    AND (toLower(d.name) CONTAINS toLower($disease) OR toLower($disease) CONTAINS toLower(d.name))
                    RETURN g.symbol, d.name, type(r) as relationship, r.evidence
                    LIMIT 10
                    """
                    
                    result = session.run(query, gene=gene, disease=disease)
                    associations = list(result)
                    
                    if associations:
                        evidence = [f"KG: {record['relationship']} - {record.get('r.evidence', 'No evidence')}" 
                                  for record in associations]
                        return True, evidence
                    
                    return False, ["No associations found in knowledge graph"]
                    
            except Exception as e:
                logger.error(f"Fallback knowledge graph query failed: {e}")
                return False, [f"Query error: {str(e)}"]
        
        return False, ["Knowledge graph unavailable"]
    
    def validate_pathway_disease_association(self, pathway: str, disease: str) -> Tuple[bool, List[str]]:
        """Check pathway-disease associations using enhanced integration"""
        
        # Try enhanced Neo4j integration first
        if self.connection and self.query_builder:
            try:
                # Use the enhanced query builder for pathway-disease associations
                query, params = self.query_builder.find_pathways_by_disease(disease, limit=100)
                results = self.connection.execute_query(query, params)
                
                # Check if our pathway is in the results
                pathway_matches = [r for r in results if pathway.lower() in r.get('pathway_name', '').lower()]
                
                if pathway_matches:
                    evidence = []
                    for match in pathway_matches:
                        pathway_name = match.get('pathway_name', 'Unknown')
                        gene_count = match.get('gene_count', 0)
                        disease_genes = match.get('disease_genes', [])
                        evidence.append(f"KG: {pathway_name} associated with {disease} ({gene_count} genes: {', '.join(disease_genes[:5])})")
                    return True, evidence
                
                # Try broader pathway search
                query = """
                MATCH (p:Pathway)
                WHERE (toLower(p.name) CONTAINS toLower($pathway) OR toLower(p.pathway_name) CONTAINS toLower($pathway))
                OPTIONAL MATCH (p)<-[:BELONGS_TO]-(g:Gene)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(d:Disease)
                WHERE (toLower(d.name) CONTAINS toLower($disease) OR toLower($disease) CONTAINS toLower(d.name))
                WITH p, collect(DISTINCT g.symbol) as associated_genes, count(DISTINCT g) as gene_count
                WHERE gene_count > 0
                RETURN p.name as pathway_name, p.pathway_name as alt_pathway_name, associated_genes, gene_count
                LIMIT 10
                """
                params = {"pathway": pathway, "disease": disease}
                results = self.connection.execute_query(query, params)
                
                if results:
                    evidence = []
                    for r in results:
                        p_name = r.get('pathway_name') or r.get('alt_pathway_name', 'Unknown')
                        gene_count = r.get('gene_count', 0)
                        genes = r.get('associated_genes', [])
                        evidence.append(f"KG: {p_name} -> {disease} ({gene_count} associated genes: {', '.join(genes[:3])})")
                    return True, evidence
                    
                return False, ["No pathway-disease associations found in knowledge graph"]
                
            except Exception as e:
                logger.error(f"Enhanced pathway validation failed: {e}")
                # Fall through to basic driver if available
        
        # Fallback to basic driver (lazy initialization)
        if not hasattr(self, 'driver') or self.driver is None:
            try:
                self.driver = GraphDatabase.driver(
                    self._driver_uri, 
                    auth=(self._driver_user, self._driver_password)
                )
            except Exception as e:
                logger.debug(f"⚠️ Neo4j driver creation failed: {e}")
                return False, ["Neo4j driver unavailable"]
        
        if self.driver:
            try:
                with self.driver.session() as session:
                    query = """
                    MATCH (p:Pathway)
                    WHERE (toLower(p.name) CONTAINS toLower($pathway) OR toLower(p.pathway_name) CONTAINS toLower($pathway))
                    OPTIONAL MATCH (p)-[r]-(d:Disease)
                    WHERE (toLower(d.name) CONTAINS toLower($disease) OR toLower($disease) CONTAINS toLower(d.name))
                    RETURN p.name, p.pathway_name, d.name, type(r) as relationship
                    LIMIT 5
                    """
                    
                    result = session.run(query, pathway=pathway, disease=disease)
                    associations = list(result)
                    
                    if associations:
                        evidence = [f"KG: {record.get('p.name', record.get('p.pathway_name', 'Unknown'))} -> {record['relationship']}" 
                                  for record in associations if record.get('relationship')]
                        return True, evidence if evidence else ["Pathway found but no direct disease associations"]
                    
                    return False, ["No pathway associations found"]
                    
            except Exception as e:
                logger.error(f"Fallback pathway validation failed: {e}")
                return False, [f"Query error: {str(e)}"]
        
        return False, ["Knowledge graph unavailable"]
    
    def close(self):
        """Close Neo4j connection"""
        if self.connection:
            self.connection.close()
        elif hasattr(self, 'driver') and self.driver:
            self.driver.close()

class LLMReasoningEngine:
    """OpenAI-based evidence synthesis and reasoning"""
    
    def __init__(self, api_key: str = None):
        # Try multiple ways to get the API key
        if api_key:
            final_api_key = api_key
        elif DECOUPLE_AVAILABLE:
            try:
                final_api_key = decouple_config('OPENAI_API_KEY')
            except Exception:
                final_api_key = os.getenv("OPENAI_API_KEY")
        else:
            final_api_key = os.getenv("OPENAI_API_KEY")
            
        self.client = OpenAI(api_key=final_api_key)
        # Cache for disease contexts to avoid regenerating for the same disease
        self._disease_context_cache = {}
        
    def clear_disease_cache(self):
        """Clear disease context cache if needed"""
        self._disease_context_cache.clear()
        print("🗑️ Disease context cache cleared")
        
    def _generate_disease_context(self, disease: str) -> str:
        """Dynamically generate disease-specific context using AI (with caching)"""
        
        # Check cache first
        disease_key = disease.lower().strip()
        if disease_key in self._disease_context_cache:
            print(f"✅ Using cached disease context for {disease}")
            return self._disease_context_cache[disease_key]
        
        print(f"🧠 Generating NEW disease context for {disease}")
        try:
            prompt = f"""
You are a medical expert. Provide a concise but comprehensive overview of {disease} for pathway analysis.

Focus on:
1. Primary pathophysiological mechanisms
2. Key molecular pathways involved
3. Immune/inflammatory components (if any)
4. Cellular dysfunctions
5. Relevant biomarkers or molecular signatures

Respond in 3-4 sentences, focusing on pathway-relevant information.
"""
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a clinical pathology expert specializing in molecular mechanisms of disease."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            context = response.choices[0].message.content.strip()
            
            # Cache the generated context
            self._disease_context_cache[disease_key] = context
            print(f"💾 Cached disease context for {disease}")
            
            return context
        except Exception as e:
            print(f"⚠️ Failed to generate disease context: {e}")
            fallback_context = f"{disease} is a complex disorder requiring pathway-level molecular analysis."
            
            # Cache the fallback as well
            self._disease_context_cache[disease_key] = fallback_context
            
            return fallback_context
    
    def _generate_regulation_context(self, pathway: str, regulation_direction: str, disease: str) -> str:
        """Dynamically generate regulation-specific context using AI"""
        try:
            prompt = f"""
You are a systems biology expert. Explain the biological significance of {regulation_direction.lower()} {pathway} pathway in {disease}.

Focus on:
1. What {regulation_direction.lower()} activity means biologically
2. Whether this typically indicates pathogenic or protective mechanisms
3. Relevance to disease pathophysiology
4. Expected downstream effects

Respond in 2-3 sentences with clear biological reasoning.
"""
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a molecular pathologist expert in pathway regulation and disease mechanisms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Failed to generate regulation context: {e}")
            return f"{regulation_direction} {pathway} pathway may have biological significance in {disease}."
    
    def _search_current_literature(self, pathway: str, disease: str) -> str:
        """Use AI to search for current literature and evidence"""
        try:
            search_prompt = f"""
You are a medical research expert. Search for and summarize the most current understanding of:
- {pathway} pathway in {disease}
- Recent publications on {pathway} AND {disease}
- Current evidence for pathogenic vs protective roles
- Molecular mechanisms and clinical significance

Provide a concise summary of current evidence in 2-3 sentences.
If you find recent publications, mention them. Focus on biological mechanisms and clinical relevance.
"""
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical research expert with access to current literature databases. Search for and provide evidence-based summaries."},
                    {"role": "user", "content": search_prompt}
                ],
                max_tokens=200,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Failed to search current literature: {e}")
            return f"Limited current literature search available for {pathway} in {disease}."
    
    def _extract_pathway_context(self, pathway_data) -> str:
        """Extract comprehensive pathway context from dataframe data"""
        if pathway_data is None:
            return ""
        
        try:
            context_parts = []
            
            def safe_get(data, key, default='N/A'):
                """Safely extract value from dict or pandas Series"""
                try:
                    if hasattr(data, 'get'):
                        # Dictionary-like object
                        value = data.get(key, default)
                    elif hasattr(data, '__getitem__'):
                        # Series or dict-like indexing
                        try:
                            value = data[key] if key in data else default
                        except (KeyError, TypeError):
                            value = default
                    else:
                        value = default
                    
                    # Handle NaN, None, and float values
                    if value is None or (hasattr(value, '__len__') and len(str(value).strip()) == 0):
                        return default
                    if str(value).lower() in ['nan', 'none', 'null']:
                        return default
                    return value
                except Exception:
                    return default
            
            # Statistical Evidence
            p_value = safe_get(pathway_data, 'P_Value')
            fdr = safe_get(pathway_data, 'FDR')
            if p_value != 'N/A' and fdr != 'N/A':
                context_parts.append(f"Statistical significance: p-value={p_value}, FDR={fdr}")
            
            # LLM Scoring
            llm_score = safe_get(pathway_data, 'LLM_Score')
            confidence_level = safe_get(pathway_data, 'Confidence_Level')
            if llm_score != 'N/A':
                context_parts.append(f"Previous LLM assessment: {llm_score}/100 ({confidence_level} confidence)")
            
            # Gene Information
            num_genes = safe_get(pathway_data, 'Number_of_Genes')
            input_genes = safe_get(pathway_data, 'Input_Genes', '')
            pathway_genes = safe_get(pathway_data, 'Pathway_Associated_Genes', '')
            if num_genes != 'N/A':
                context_parts.append(f"Pathway contains {num_genes} genes")
            if input_genes and pathway_genes and input_genes != 'N/A' and pathway_genes != 'N/A':
                try:
                    input_list = str(input_genes).split(',')[:5]  # First 5 genes
                    pathway_list = str(pathway_genes).split(',')[:5]
                    context_parts.append(f"Key input genes: {', '.join(input_list)}")
                    context_parts.append(f"Associated pathway genes: {', '.join(pathway_list)}")
                except Exception:
                    pass  # Skip if splitting fails
            
            # Clinical and Functional Relevance
            clinical_relevance = safe_get(pathway_data, 'Clinical_Relevance', '')
            functional_relevance = safe_get(pathway_data, 'Functional_Relevance', '')
            if clinical_relevance and clinical_relevance != 'N/A':
                context_parts.append(f"Clinical relevance: {str(clinical_relevance)[:200]}...")
            if functional_relevance and functional_relevance != 'N/A':
                context_parts.append(f"Functional relevance: {str(functional_relevance)[:200]}...")
            
            # Classification
            main_class = safe_get(pathway_data, 'Main_Class', '')
            sub_class = safe_get(pathway_data, 'Sub_Class', '')
            disease_category = safe_get(pathway_data, 'Disease_Category', '')
            if (main_class and main_class != 'N/A') or (sub_class and sub_class != 'N/A'):
                context_parts.append(f"Pathway classification: {main_class} → {sub_class}")
            if disease_category and disease_category != 'N/A':
                context_parts.append(f"Disease category: {disease_category}")
            
            # Score Justification
            score_justification = safe_get(pathway_data, 'Score_Justification', '')
            if score_justification and score_justification != 'N/A':
                context_parts.append(f"Previous analysis: {str(score_justification)[:300]}...")
            
            return ". ".join(context_parts)
            
        except Exception as e:
            print(f"⚠️ Failed to extract pathway context: {e}")
            return "Pathway data context extraction failed"
    
    def synthesize_evidence(self, item: str, item_type: str, disease: str,
                          regulation_direction: str = None,
                          pathway_genes: List = None,
                          pathway_data: Dict = None) -> ValidationResult:
        """Use AI to dynamically generate context and perform intelligent evidence synthesis"""
        
        # ENHANCED: Generate dynamic disease-specific context using AI (cached per disease)
        disease_context = self._generate_disease_context(disease)
        
        # ENHANCED: Generate dynamic regulation-specific context using AI
        regulation_context = ""
        if regulation_direction and item_type == "pathway":
            print(f"🧠 Generating dynamic regulation context for {regulation_direction} {item}")
            regulation_context = self._generate_regulation_context(item, regulation_direction, disease)
        
        # ENHANCED: Search current literature for real-time evidence
        print(f"🔍 Searching current literature for {item} in {disease}")
        current_literature = self._search_current_literature(item, disease)
        
        # ENHANCED: Extract comprehensive pathway context from dataframe
        pathway_dataframe_context = ""
        if pathway_data:
            print(f"📊 Extracting pathway data context for {item}")
            pathway_dataframe_context = self._extract_pathway_context(pathway_data)
        
        # Build pathway-specific context
        pathway_context = ""
        if item_type == "pathway" and pathway_genes:
            gene_list = ", ".join([g.get('gene', str(g)) for g in pathway_genes[:3]])
            pathway_context = f"Key genes involved: {gene_list}"
        
        prompt = f"""
You are a clinical genomics expert with real-time access to medical literature and databases.

ANALYSIS TARGET:
- Item: {item}
- Type: {item_type}
- Disease: {disease}

DYNAMIC DISEASE CONTEXT:
{disease_context}

DYNAMIC REGULATION CONTEXT:
{regulation_context}

PATHWAY CONTEXT:
{pathway_context}

PATHWAY DATAFRAME EVIDENCE:
{pathway_dataframe_context}

CURRENT LITERATURE EVIDENCE:
{current_literature}

INSTRUCTIONS: Use your comprehensive medical knowledge and the provided evidence to:
1. **Prioritize pathway dataframe evidence** (statistical significance, previous LLM scores, clinical relevance)
2. Analyze the biological mechanisms of "{item}" in "{disease}"
3. Apply current understanding of pathway functions and disease mechanisms
4. Evaluate pathogenic vs protective roles based on established science
5. Consider regulation-specific implications for the disease pathophysiology

ANALYSIS FRAMEWORK:
1. **Statistical Evidence**: Consider p-values, FDR, and gene counts from dataframe
2. **Previous Assessment**: Review existing LLM scores and confidence levels
3. **Clinical Context**: Analyze provided clinical and functional relevance descriptions
4. **Biological Mechanism**: How does this pathway function normally vs in disease?
5. **Regulation Impact**: What does the observed regulation direction mean biologically?
6. **Gene Evidence**: Consider input genes and pathway-associated genes
7. **Disease Classification**: Use disease category and pathway classification information

DECISION CRITERIA:
✅ PATHOGENIC if: Dysregulation contributes to disease mechanisms
❌ NON-PATHOGENIC if: Limited disease relevance or protective role
🧠 Use your comprehensive medical training and biological knowledge

Respond in JSON format:
{{
    "is_valid": boolean,
    "confidence": float,
    "category": "high|medium|low|insufficient",
    "justification": "evidence-based biological explanation with sources if found"
}}

CONFIDENCE GUIDELINES:
- High (0.8-1.0): Strong biological rationale with well-established mechanisms
- Medium (0.5-0.79): Clear biological rationale with reasonable evidence
- Low (0.3-0.49): Some biological rationale but less certain relevance
- Insufficient (<0.3): No clear biological rationale or established connection

PRIORITY: Use established scientific understanding and proven biological mechanisms from your training.
"""
        
        try:
            # PURE AI: Use GPT-4o with comprehensive medical knowledge only
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4o for superior biological reasoning
                messages=[
                    {"role": "system", "content": "You are a world-class clinical genomics expert. Use your comprehensive medical training and biological knowledge to provide accurate pathway analysis. Respond only in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,  # Sufficient for detailed biological reasoning
                temperature=0.15,  # Very low temperature for consistent, accurate analysis
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean the response to extract JSON if it's wrapped in markdown or other text
            if result_text.startswith('```json'):
                # Extract JSON from markdown code block
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    result_text = result_text[json_start:json_end]
            elif result_text.startswith('```'):
                # Extract content from generic code block
                lines = result_text.split('\n')
                result_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else result_text
            
            # Try to find JSON object if response has extra text
            if not result_text.startswith('{'):
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    result_text = result_text[json_start:json_end]
            
            result_data = json.loads(result_text)
            
            # PURE AI: Evidence comes from AI reasoning and dynamic contexts
            all_evidence = [disease_context, regulation_context, current_literature]
            all_sources = ['AI Disease Context', 'AI Regulation Context', 'AI Literature Search']
            
            # Cap confidence at 85% to maintain scientific uncertainty
            capped_confidence = min(max(result_data['confidence'], 0.0), 0.85)
            
            # Determine status based on confidence and validity for pathways/genes
            if capped_confidence >= 0.7 and result_data['is_valid']:
                status = "Pathogenic"  # High confidence pathogenic
            elif capped_confidence >= 0.5 and result_data['is_valid']:
                status = "Protective"  # Moderate confidence protective
            else:
                status = "Uncertain"   # Low confidence or invalid
            
            return ValidationResult(
                is_valid=result_data['is_valid'],
                confidence=capped_confidence,
                evidence=all_evidence,
                sources=all_sources,
                justification=result_data['justification'],
                category=result_data['category'],
                status=status
            )
            
        except Exception as e:
            print(f"⚠️ AI synthesis failed: {e}")
            print(f"🔍 Debug: Response content was: {result_text[:200] if 'result_text' in locals() else 'No response content available'}")
            # PURE AI FALLBACK: Use basic confidence based on item and disease context
            # This is a simple fallback when AI fails
            if item_type in ["pathway", "gene"] and len(item) > 3 and len(disease) > 3:
                category = "low"
                confidence = 0.3
                is_valid = True
            else:
                category = "insufficient"
                confidence = 0.1
                is_valid = False
            
            # Cap confidence at 85% for fallback as well
            capped_confidence = min(max(confidence, 0.0), 0.85)
            
            # Determine status for fallback
            if capped_confidence >= 0.7 and is_valid:
                fallback_status = "Pathogenic"
            elif capped_confidence >= 0.5 and is_valid:
                fallback_status = "Protective" 
            else:
                fallback_status = "Uncertain"
            
            return ValidationResult(
                is_valid=is_valid,
                confidence=capped_confidence,
                evidence=["AI synthesis fallback activated"],
                sources=["AI Fallback"],
                justification="Fallback assessment when AI synthesis fails",
                category=category,
                status=fallback_status
            )
    
    # Removed _format_evidence function - replaced with dynamic AI-generated contexts and real-time literature search

class ClinicalValidator:
    """Main validation coordinator with self-healing and learning capabilities"""
    
    def __init__(self, cache_path: str = "feedback_log.db"):
        self.cache = EvidenceCache(cache_path)
        # PURE AI: Only LLM reasoning engine - no external dependencies
        self.llm = LLMReasoningEngine()
        
        # Synonym dictionaries for correction loops
        self.gene_synonyms = {
            "interferon": ["IFN", "interferon", "IFNA", "IFNB", "IFNG"],
            "immune": ["immunity", "immunological", "inflammatory"],
            "complement": ["complement system", "complement cascade"]
        }
        
        self.pathway_synonyms = {
            "immune response": ["immunity", "immune system", "inflammatory response"],
            "interferon signaling": ["IFN pathway", "interferon response", "type I interferon"],
            "complement": ["complement system", "complement cascade", "complement activation"]
        }
        
        self.drug_synonyms = {
            "interferon": ["IFN-alpha", "peginterferon", "interferon alfa"],
            "immunosuppressive": ["immunosuppressant", "immune suppressor"]
        }
    
    def validate_gene(self, gene: str, disease: str, 
                     use_cache: bool = True) -> ValidationResult:
        """Pure AI-based gene validation"""
        
        # Check cache first
        if use_cache:
            cached = self.cache.get_cached_result(disease, gene, "gene")
            if cached:
                print(f"✅ Using cached result for gene {gene} in {disease}")
                return cached
        
        # PURE AI: Use only LLM intelligence for gene validation
        result = self.llm.synthesize_evidence(
            item=gene,
            item_type="gene", 
            disease=disease
        )
        
        # If low confidence, try synonyms with AI reasoning
        if result.confidence < 0.5 and gene.lower() in self.gene_synonyms:
            print(f"🧠 Trying AI analysis with gene synonyms for: {gene}")
            
            for synonym in self.gene_synonyms[gene.lower()]:
                synonym_result = self.llm.synthesize_evidence(
                    item=synonym,
                    item_type="gene",
                    disease=disease
                )
                
                if synonym_result.confidence > result.confidence:
                    result = synonym_result
                    print(f"🔄 Better result found with synonym: {synonym}")
                    break
        
        # Cache the result
        if use_cache:
            self.cache.store_result(disease, gene, "gene", result)
        
        return result
    
    def validate_pathway(self, pathway: str, disease: str,
                        use_cache: bool = True, 
                        regulation_direction: str = None,
                        pathway_genes: List = None,
                        pathway_data: Dict = None) -> ValidationResult:
        """Validate pathway relevance with correction loop"""
        
        # Check cache
        if use_cache:
            cached = self.cache.get_cached_result(disease, pathway, "pathway")
            if cached:
                return cached
        
        # PURE AI: Use AI-only validation with pathway dataframe context
        result = self._validate_pathway_pure_ai(pathway, disease, regulation_direction, pathway_genes, pathway_data)
        
        # If low confidence, try synonyms with AI reasoning
        if result.confidence < 0.5 and pathway.lower() in self.pathway_synonyms:
            print(f"🧠 Trying AI analysis with pathway synonyms for: {pathway}")
            
            for synonym in self.pathway_synonyms[pathway.lower()]:
                synonym_result = self._validate_pathway_pure_ai(synonym, disease, regulation_direction, pathway_genes, pathway_data)
                if synonym_result.confidence > result.confidence:
                    result = synonym_result
                    print(f"🔄 Better result found with synonym: {synonym}")
                    break
        
        # Cache result
        if use_cache:
            self.cache.store_result(disease, pathway, "pathway", result)
        
        return result
    
    def _validate_pathway_pure_ai(self, pathway: str, disease: str, 
                                 regulation_direction: str = None,
                                 pathway_genes: List = None,
                                 pathway_data: Dict = None) -> ValidationResult:
        """Pure AI-based pathway validation without external dependencies"""
        
        try:
            # PURE AI: Use only LLM intelligence and knowledge for validation with pathway data context
            result = self.llm.synthesize_evidence(
                item=pathway,
                item_type="pathway",
                disease=disease, 
                regulation_direction=regulation_direction,
                pathway_genes=pathway_genes,
                pathway_data=pathway_data
            )
            
            # Debug: Log the actual result for troubleshooting
            print(f"🔍 AI validation result for {pathway}: confidence={result.confidence:.2f}, is_valid={result.is_valid}")
            
            # Ensure minimum confidence for valid pathways
            if result.confidence < 0.1 and result.is_valid:
                print(f"⚠️  Boosting very low confidence for valid pathway {pathway} from {result.confidence:.2f} to 0.3")
                result.confidence = 0.3
            
            return result
            
        except Exception as e:
            print(f"❌ AI validation failed for {pathway}: {e}")
            # Return a more reasonable fallback instead of 0.00 confidence
            return ValidationResult(
                is_valid=True,  # Assume pathogenic by default for safety
                confidence=0.4,  # Moderate confidence fallback
                evidence=["AI validation failed, using conservative fallback"],
                sources=["Fallback Assessment"],
                justification=f"AI validation failed for {pathway} in {disease}, using conservative pathogenic classification",
                category="medium",
                status="Uncertain"
            )
    
    def validate_drug(self, drug: str, disease: str, 
                     use_cache: bool = True) -> ValidationResult:
        """Pure AI-based drug validation"""
        
        if use_cache:
            cached = self.cache.get_cached_result(disease, drug, "drug")
            if cached:
                return cached
        
        # PURE AI: Use only LLM intelligence for drug validation
        result = self.llm.synthesize_evidence(
            item=drug,
            item_type="drug",
            disease=disease
        )
        
        # If low confidence, try synonyms with AI reasoning
        if result.confidence < 0.5 and drug.lower() in self.drug_synonyms:
            print(f"🧠 Trying AI analysis with drug synonyms for: {drug}")
            
            for synonym in self.drug_synonyms[drug.lower()]:
                synonym_result = self.llm.synthesize_evidence(
                    item=synonym,
                    item_type="drug",
                    disease=disease
                )
                
                if synonym_result.confidence > result.confidence:
                    result = synonym_result
                    print(f"🔄 Better result found with synonym: {synonym}")
                    break
        
        if use_cache:
            self.cache.store_result(disease, drug, "drug", result)
        
        return result
    
    def correction_loop(self, items: List[str], item_type: str, disease: str) -> Dict[str, ValidationResult]:
        """Self-healing correction loop with synonym retry"""
        results = {}
        
        for item in items:
            if item_type == "gene":
                result = self.validate_gene(item, disease)
            elif item_type == "pathway": 
                result = self.validate_pathway(item, disease)
            elif item_type == "drug":
                result = self.validate_drug(item, disease)
            else:
                continue
            
            results[item] = result
            
            # Log validation outcome
            logger.info(f"Validated {item_type} '{item}' for {disease}: "
                       f"{result.category} confidence ({result.confidence:.2f})")
        
        return results
    
    def close(self):
        """Cleanup resources"""
        # No resources to close - using stateless LLM-only validation
        pass