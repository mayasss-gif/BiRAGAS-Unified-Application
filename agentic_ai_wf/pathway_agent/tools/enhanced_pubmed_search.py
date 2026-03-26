#!/usr/bin/env python3
"""
Enhanced Production-Ready PubMed Search with Circuit Breaker and LLM Fallback
"""
import asyncio
import time
import json
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from Bio import Entrez
import random
from agents import Agent, Runner
import openai
from ..helpers import logger

# Configure Entrez
Entrez.email = "f420testing@ayassbioscience.com"

class PubMedArticle(BaseModel):
    pmid: str = Field(description="The PubMed ID of the article")
    title: str = Field(description="The title of the article")
    abstract: str = Field(description="The abstract of the article")
    url: str = Field(description="The URL of the article")
    dated: date = Field(description="The date of the article")
    journal: str = Field(description="The journal of the article")
    article_types: List[str] = Field(description="The types of the article")
    source: str = Field(default="pubmed", description="Source of the article (pubmed/llm)")

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3  # failures before opening circuit
    recovery_timeout: int = 60  # seconds before trying half-open
    success_threshold: int = 2  # successes needed to close circuit
    
@dataclass
class ErrorStats:
    consecutive_failures: int = 0
    total_failures: int = 0
    total_requests: int = 0
    last_failure_time: Optional[datetime] = None
    recent_errors: List[str] = field(default_factory=list)
    
    def add_failure(self, error: str):
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_requests += 1
        self.last_failure_time = datetime.now()
        self.recent_errors.append(f"{datetime.now()}: {error}")
        # Keep only last 10 errors
        self.recent_errors = self.recent_errors[-10:]
    
    def add_success(self):
        self.consecutive_failures = 0
        self.total_requests += 1

class PubMedCircuitBreaker:
    """Circuit breaker for PubMed API with intelligent failure detection."""
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = ErrorStats()
        self.last_state_change = datetime.now()
        
    def is_request_allowed(self) -> bool:
        """Check if request should be allowed based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if we should try half-open
            if datetime.now() - self.last_state_change > timedelta(seconds=self.config.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = datetime.now()
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        return False
    
    def record_success(self):
        """Record successful request."""
        self.stats.add_success()
        
        if self.state == CircuitState.HALF_OPEN:
            # Check if we can close the circuit
            if self.stats.consecutive_failures == 0:
                self.state = CircuitState.CLOSED
                self.last_state_change = datetime.now()
                logger.info("Circuit breaker CLOSED - PubMed is healthy")
    
    def record_failure(self, error: str):
        """Record failed request and potentially open circuit."""
        self.stats.add_failure(error)
        
        # Check if we should open the circuit
        if (self.state == CircuitState.CLOSED and 
            self.stats.consecutive_failures >= self.config.failure_threshold):
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now()
            logger.warning(f"Circuit breaker OPENED - PubMed consecutive failures: {self.stats.consecutive_failures}")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now()
            logger.warning("Circuit breaker reopened during HALF_OPEN test")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "state": self.state.value,
            "consecutive_failures": self.stats.consecutive_failures,
            "total_failures": self.stats.total_failures,
            "total_requests": self.stats.total_requests,
            "success_rate": (
                (self.stats.total_requests - self.stats.total_failures) / self.stats.total_requests 
                if self.stats.total_requests > 0 else 0
            ),
            "last_failure": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            "recent_errors": self.stats.recent_errors[-3:]  # Last 3 errors
        }

class EnhancedPubMedSearch:
    """Production-ready PubMed search with retry, circuit breaker, and LLM fallback."""
    
    def __init__(self):
        self.circuit_breaker = PubMedCircuitBreaker()
        self.request_times = []  # For rate limiting
        self.cache = {}  # Query cache
        self.llm_fallback_agent = self._create_llm_agent()
        
    def _create_llm_agent(self) -> Agent:
        """Create LLM agent for generating synthetic literature when PubMed fails."""
        return Agent(
            name="literature_generator",
            instructions="""
            You are a biomedical literature synthesis agent. When PubMed is unavailable, 
            you generate realistic, scientifically-grounded literature summaries for 
            gene-pathway-disease relationships.
            
            For each query about a gene, pathway, and disease:
            1. Generate 1-3 realistic article summaries
            2. Include plausible PMIDs (format: LLM-PMID-XXXXXX)
            3. Create realistic titles and abstracts based on current biomedical knowledge
            4. Include appropriate journals and article types
            5. Base content on established scientific understanding
            
            Mark all generated content clearly as "LLM-generated" in the source field.
            Focus on clinically relevant and mechanistically sound content.
            """,
            model="gpt-4o-mini"
        )
    
    async def _wait_for_rate_limit(self):
        """Implement intelligent rate limiting to avoid 429 errors."""
        now = time.time()
        # Remove requests older than 1 second
        self.request_times = [t for t in self.request_times if now - t < 1.0]
        
        # If we have made too many requests recently, wait
        if len(self.request_times) >= 3:  # Max 3 requests per second
            wait_time = 1.0 - (now - self.request_times[0])
            if wait_time > 0:
                logger.info(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        self.request_times.append(now)
    
    async def _retry_with_backoff(self, func, max_retries: int = 3):
        """Retry function with exponential backoff for transient errors."""
        for attempt in range(max_retries):
            try:
                await self._wait_for_rate_limit()
                result = func()
                self.circuit_breaker.record_success()
                return result
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a retryable error
                if any(code in error_str for code in ["500", "502", "503", "429"]):
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0.1, 0.5)  # Exponential backoff with jitter
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time:.2f}s for error: {error_str}")
                        await asyncio.sleep(wait_time)
                        continue
                
                # Record failure and re-raise
                self.circuit_breaker.record_failure(error_str)
                raise e
        
        # If we get here, all retries failed
        raise Exception(f"All {max_retries} retry attempts failed")
    
    def _pubmed_search_core(self, query: str) -> List[PubMedArticle]:
        """Core PubMed search function - synchronous for use with retry decorator."""
        search_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=3,
            sort="relevance"
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results.get("IdList", [])
        if not id_list:
            return []
        
        articles = []
        for pmid in id_list:
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=pmid,
                rettype="medline",
                retmode="xml"
            )
            fetch_results = Entrez.read(fetch_handle)
            fetch_handle.close()

            article = fetch_results["PubmedArticle"][0]
            citation = article["MedlineCitation"]
            article_info = citation["Article"]

            title = article_info["ArticleTitle"]
            abstract_parts = article_info.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available."
            journal = article_info["Journal"].get("Title", "Unknown")
            pub_types = article_info.get("PublicationTypeList", [])
            article_types = [str(pt) for pt in pub_types]

            raw_dated = citation.get("DateCompleted") or citation.get("DateRevised") or citation.get("DateCreated")
            if raw_dated:
                dated = date(int(raw_dated['Year']), int(raw_dated['Month']), int(raw_dated['Day']))
            else:
                dated = date.today()
            
            articles.append(PubMedArticle(
                pmid=str(pmid),
                title=title,
                abstract=abstract,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                dated=dated,
                journal=journal,
                article_types=article_types,
                source="pubmed"
            ))
        
        articles.sort(key=lambda x: x.dated, reverse=True)
        return articles
    
    async def _generate_llm_fallback(self, queries: List[str]) -> List[List[PubMedArticle]]:
        """Generate batch LLM fallback when PubMed is unavailable."""
        logger.info(f"Generating LLM fallback for {len(queries)} queries")
        
        # Process queries in smaller batches for LLM
        batch_size = 5
        all_results = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            
            prompt = {
                "task": "Generate scientific literature for gene-pathway-disease relationships",
                "queries": batch_queries,
                "requirements": {
                    "articles_per_query": "1-3 realistic articles",
                    "pmid_format": "LLM-PMID-XXXXXX (6 digit number)",
                    "include": ["title", "abstract", "journal", "article_types", "publication_date"],
                    "focus": "clinically relevant, mechanistically sound content",
                    "base_on": "established biomedical knowledge"
                }
            }
            
            try:
                # Use the LLM agent to generate content
                response = await Runner.run(self.llm_fallback_agent, json.dumps(prompt))
                
                # Parse the response and create PubMedArticle objects
                batch_results = self._parse_llm_response(response.final_output, batch_queries)
                all_results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"LLM fallback failed for batch: {e}")
                # Return empty results for this batch
                all_results.extend([[] for _ in batch_queries])
        
        return all_results
    
    def _parse_llm_response(self, llm_output: str, queries: List[str]) -> List[List[PubMedArticle]]:
        """Parse LLM output into PubMedArticle objects."""
        results = []
        
        try:
            # Try to parse as JSON first
            if isinstance(llm_output, str):
                try:
                    data = json.loads(llm_output)
                except:
                    # If not JSON, create simplified articles based on queries
                    data = self._create_simple_articles_from_queries(queries)
            else:
                data = llm_output
            
            # Process each query's results
            for i, query in enumerate(queries):
                query_articles = []
                
                # Extract articles for this query from LLM response
                if isinstance(data, dict) and 'articles' in data:
                    articles_data = data['articles'].get(str(i), [])
                elif isinstance(data, list) and i < len(data):
                    articles_data = data[i] if isinstance(data[i], list) else [data[i]]
                else:
                    articles_data = []
                
                # Create PubMedArticle objects
                for j, article_data in enumerate(articles_data):
                    if isinstance(article_data, dict):
                        query_articles.append(PubMedArticle(
                            pmid=article_data.get('pmid', f'LLM-PMID-{random.randint(100000, 999999)}'),
                            title=article_data.get('title', f'Research on {query.split("AND")[0].strip("()")}'),
                            abstract=article_data.get('abstract', f'This study investigates the relationship between {query.split("AND")[0].strip("()")} and disease pathways.'),
                            url=f"https://llm-generated-article.com/{random.randint(100000, 999999)}",
                            dated=date.today(),
                            journal=article_data.get('journal', 'Journal of Clinical Research'),
                            article_types=article_data.get('article_types', ['Journal Article']),
                            source="llm"
                        ))
                
                results.append(query_articles)
            
            return results
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            # Return empty results for all queries
            return [[] for _ in queries]
    
    def _create_simple_articles_from_queries(self, queries: List[str]) -> Dict:
        """Create simplified articles when LLM parsing fails."""
        articles = {}
        
        for i, query in enumerate(queries):
            # Extract gene, pathway, disease from query
            parts = query.split(" AND ")
            gene = parts[0].strip("()").replace("[TIAB]", "") if len(parts) > 0 else "Unknown"
            pathway = parts[1].strip("()").replace("[MeSH]", "") if len(parts) > 1 else "Unknown pathway"
            disease = parts[2].strip("()").replace("[MeSH]", "") if len(parts) > 2 else "Unknown disease"
            
            articles[str(i)] = [{
                'pmid': f'LLM-PMID-{random.randint(100000, 999999)}',
                'title': f'{gene} involvement in {pathway} in {disease}',
                'abstract': f'This study investigates the role of {gene} in {pathway} mechanisms related to {disease}. The research provides insights into potential therapeutic targets and biomarkers.',
                'journal': 'Biomedical Research Journal',
                'article_types': ['Journal Article'],
                'publication_date': date.today().isoformat()
            }]
        
        return {"articles": articles}
    
    async def search_batch(self, queries: List[str]) -> List[List[PubMedArticle]]:
        """Search PubMed for multiple queries with fallback to LLM."""
        logger.info(f"Starting batch search for {len(queries)} queries")
        
        # Check cache first
        cached_results = []
        uncached_queries = []
        uncached_indices = []
        
        for i, query in enumerate(queries):
            if query in self.cache:
                cached_results.append((i, self.cache[query]))
            else:
                uncached_queries.append(query)
                uncached_indices.append(i)
        
        logger.info(f"Cache hits: {len(cached_results)}, Cache misses: {len(uncached_queries)}")
        
        # Initialize results array
        results = [None] * len(queries)
        
        # Fill in cached results
        for i, cached_result in cached_results:
            results[i] = cached_result
        
        # Process uncached queries
        if uncached_queries:
            if self.circuit_breaker.is_request_allowed():
                # Try PubMed first
                try:
                    pubmed_results = await self._search_pubmed_batch(uncached_queries)
                    
                    # Cache and store results
                    for i, (query_idx, query) in enumerate(zip(uncached_indices, uncached_queries)):
                        self.cache[query] = pubmed_results[i]
                        results[query_idx] = pubmed_results[i]
                    
                    logger.info(f"Successfully retrieved {len(pubmed_results)} results from PubMed")
                    
                except Exception as e:
                    logger.error(f"PubMed batch search failed: {e}")
                    # Fall back to LLM for failed queries
                    llm_results = await self._generate_llm_fallback(uncached_queries)
                    
                    for i, (query_idx, query) in enumerate(zip(uncached_indices, uncached_queries)):
                        self.cache[query] = llm_results[i]
                        results[query_idx] = llm_results[i]
                    
                    logger.info(f"Used LLM fallback for {len(uncached_queries)} queries")
            else:
                # Circuit breaker is open, use LLM fallback
                logger.warning("Circuit breaker OPEN - using LLM fallback")
                llm_results = await self._generate_llm_fallback(uncached_queries)
                
                for i, (query_idx, query) in enumerate(zip(uncached_indices, uncached_queries)):
                    # Don't cache LLM results when circuit is open (we want to retry PubMed later)
                    results[query_idx] = llm_results[i]
        
        # Log circuit breaker status
        status = self.circuit_breaker.get_status()
        logger.info(f"Circuit breaker status: {status['state']} (success rate: {status['success_rate']:.2%})")
        
        return results
    
    async def _search_pubmed_batch(self, queries: List[str]) -> List[List[PubMedArticle]]:
        """Search PubMed for a batch of queries with proper error handling."""
        results = []
        
        for query in queries:
            try:
                articles = await self._retry_with_backoff(
                    lambda: self._pubmed_search_core(query)
                )
                results.append(articles)
                
            except Exception as e:
                logger.error(f"Failed to search PubMed for query '{query}': {e}")
                results.append([])  # Empty result for failed query
        
        return results
    
    async def search_single(self, query: str) -> List[PubMedArticle]:
        """Search for a single query - convenience method."""
        results = await self.search_batch([query])
        return results[0] if results else []
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the search system."""
        cb_status = self.circuit_breaker.get_status()
        
        return {
            "circuit_breaker": cb_status,
            "cache_size": len(self.cache),
            "system_status": "healthy" if cb_status["state"] == "closed" else "degraded",
            "recommendations": self._get_health_recommendations(cb_status)
        }
    
    def _get_health_recommendations(self, cb_status: Dict) -> List[str]:
        """Generate health recommendations based on current status."""
        recommendations = []
        
        if cb_status["state"] == "open":
            recommendations.append("PubMed API is experiencing issues - using LLM fallback")
            recommendations.append("Check NCBI service status")
        
        if cb_status["success_rate"] < 0.8:
            recommendations.append("High failure rate detected - consider reducing request frequency")
        
        if cb_status["consecutive_failures"] > 1:
            recommendations.append("Multiple consecutive failures - monitor API health")
        
        return recommendations

# Global instance for backward compatibility
enhanced_pubmed_search = EnhancedPubMedSearch()

# Legacy function wrapper
async def pubmed_search(query: str) -> List[PubMedArticle]:
    """Legacy wrapper for backward compatibility."""
    return await enhanced_pubmed_search.search_single(query)

# New batch function
async def pubmed_search_batch(queries: List[str]) -> List[List[PubMedArticle]]:
    """Batch search function."""
    return await enhanced_pubmed_search.search_batch(queries)
