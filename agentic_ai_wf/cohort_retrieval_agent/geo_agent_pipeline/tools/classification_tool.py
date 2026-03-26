"""
GPT Classification tool for the Cohort Retrieval Agent system.

This tool handles GPT-based classification of samples into disease/normal categories.
"""

import asyncio
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
from decouple import config

# Project Imports
from  ..base.base_tool import AsyncContextTool, ToolResult
from  ..config import CohortRetrievalConfig
from  ..exceptions import ClassificationError


@dataclass
class SampleInfo:
    """Information about a sample for classification."""
    sample_id: str
    tissue_type: str
    characteristics: List[str]
    library_source: str
    library_strategy: str
    extraction_protocol: str
    molecule: str
    additional_metadata: Dict[str, Any] = None


@dataclass
class ClassificationResult:
    """Result of sample classification."""
    sample_id: str
    classification: str
    confidence: float = 0.0
    reasoning: str = ""


class GPTClassificationTool(AsyncContextTool[List[ClassificationResult]]):
    """
    Tool for GPT-based sample classification.
    
    Uses OpenAI GPT-4 to classify samples as disease conditions or normal
    based on sample metadata and characteristics.
    """
    
    def __init__(self, cohort_retrieval_config: CohortRetrievalConfig):
        super().__init__(cohort_retrieval_config)
        
        # Initialize OpenAI client
        api_key = config("OPENAI_API_KEY")
        if not api_key:
            raise ClassificationError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        
        # Enhanced classification prompt template with best practices
        self.classification_prompt_template = '''You are an expert biomedical data scientist specializing in cancer genomics and sample classification. Your task is to classify biological samples based on their metadata.

        ## TASK
        Classify the sample as one of the following categories based on the provided metadata:
        - **Normal**: Healthy/control tissue samples
        - **Tumor**: General malignant tissue (when specific subtype unclear)
        - **ADC**: Adenocarcinoma
        - **SCC**: Squamous Cell Carcinoma  
        - **SCLC**: Small Cell Lung Carcinoma
        - **NSCLC**: Non-Small Cell Lung Carcinoma
        - **Metastatic**: Metastatic cancer tissue
        - **Primary**: Primary tumor tissue
        - **Recurrent**: Recurrent cancer tissue

        ## SAMPLE METADATA
        ```
        Sample ID: {sample_id}
        Tissue Type: {tissue_type}
        Characteristics: {characteristics}
        Library Source: {library_source}
        Library Strategy: {library_strategy}
        Extraction Protocol: {extraction_protocol}
        Molecule Type: {molecule}
        ```

        ## CLASSIFICATION GUIDELINES
        1. **Primary indicators**: Look for disease state, histology, and sample type in characteristics
        2. **Tissue context**: Consider tissue type and anatomical location
        3. **Experimental context**: RNA-seq suggests transcriptomic analysis
        4. **Keywords to identify**:
        - Normal: "normal", "healthy", "control", "adjacent normal", "non-tumor"
        - Adenocarcinoma: "adenocarcinoma", "ADC", "glandular"
        - Squamous: "squamous", "SCC", "epidermoid"
        - Tumor: "tumor", "malignant", "cancer", "carcinoma", "neoplasm"
        - Metastatic: "metastatic", "metastasis", "secondary"

        ## REASONING PROCESS
        1. Analyze characteristics for disease state keywords
        2. Consider tissue type and anatomical context
        3. Look for histological subtype information
        4. Determine confidence based on clarity of indicators

        ## OUTPUT FORMAT
        Respond with ONLY the classification label (one word) from the approved categories above.

        Examples:
        - If characteristics mention "adenocarcinoma" → ADC
        - If characteristics mention "normal epithelium" → Normal  
        - If characteristics mention "squamous cell carcinoma" → SCC
        - If characteristics mention "tumor" but no subtype → Tumor

        ## CLASSIFICATION
        Based on the metadata above, the sample classification is:'''

        # Enhanced prompt for confidence scoring (used internally)
        self.confidence_prompt_template = '''You are an expert biomedical data scientist. Rate your confidence in classifying this sample on a scale of 0.0-1.0.

        Sample Classification: {classification}
        Sample Metadata: {metadata_summary}

        Confidence factors:
        - 0.9-1.0: Clear, unambiguous indicators (e.g., "adenocarcinoma" in characteristics)
        - 0.7-0.9: Strong indicators with minor ambiguity
        - 0.5-0.7: Moderate indicators, some uncertainty
        - 0.3-0.5: Weak indicators, significant uncertainty  
        - 0.0-0.3: Very unclear or contradictory information

        Respond with only a number between 0.0 and 1.0:'''
    
    async def create_context(self) -> OpenAI:
        """Create OpenAI client context."""
        return self.client
    
    async def close_context(self, client: OpenAI):
        """Close OpenAI client context (no-op for OpenAI client)."""
        pass
    
    async def execute(self, 
                     samples: List[SampleInfo],
                     max_concurrent: int = 5) -> ToolResult[List[ClassificationResult]]:
        """
        Classify samples using GPT.
        
        Args:
            samples: List of sample information to classify
            max_concurrent: Maximum concurrent GPT requests
            
        Returns:
            ToolResult with classification results
        """
        if not self.validate_input(samples, max_concurrent):
            return ToolResult(
                success=False,
                data=[],
                error="Invalid input parameters",
                details={"samples_count": len(samples) if samples else 0}
            )
        
        try:
            result = await self._classify_samples(samples, max_concurrent)
            
            if not self.validate_output(result):
                return ToolResult(
                    success=False,
                    data=[],
                    error="Invalid output format",
                    details={"result_type": type(result).__name__}
                )
            
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            self.logger.error(f"Sample classification failed: {e}")
            return ToolResult(
                success=False,
                data=[],
                error=str(e),
                details={"exception": type(e).__name__}
            )
    
    async def _classify_samples(self, 
                               samples: List[SampleInfo], 
                               max_concurrent: int) -> List[ClassificationResult]:
        """Internal method to classify samples."""
        self.logger.info(f"Classifying {len(samples)} samples using GPT-4")
        
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create classification tasks
        tasks = []
        for sample in samples:
            task = self._classify_single_sample(sample, semaphore)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        classifications = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error classifying sample {samples[i].sample_id}: {result}")
                # Add failed classification
                classifications.append(ClassificationResult(
                    sample_id=samples[i].sample_id,
                    classification="Unknown",
                    confidence=0.0,
                    reasoning=f"Classification failed: {result}"
                ))
            else:
                classifications.append(result)
        
        self.logger.info(f"Successfully classified {len(classifications)} samples")
        return classifications
    
    async def _classify_single_sample(self, 
                                     sample: SampleInfo, 
                                     semaphore: asyncio.Semaphore) -> ClassificationResult:
        """Classify a single sample using GPT with enhanced prompting and confidence scoring."""
        async with semaphore:
            try:
                # Format characteristics as string
                characteristics_str = ", ".join(sample.characteristics) if sample.characteristics else "N/A"
                
                # Create enhanced prompt
                prompt = self.classification_prompt_template.format(
                    sample_id=sample.sample_id,
                    tissue_type=sample.tissue_type,
                    characteristics=characteristics_str,
                    library_source=sample.library_source,
                    library_strategy=sample.library_strategy,
                    extraction_protocol=sample.extraction_protocol,
                    molecule=sample.molecule
                )
                
                self.logger.debug(f"Classifying sample {sample.sample_id} with enhanced prompt...")
                
                # Make GPT request for classification
                response = await self._make_gpt_request(prompt, max_tokens=10)
                
                # Parse and validate classification
                classification = self._parse_classification(response, sample.sample_id)
                
                # Get confidence score
                confidence = await self._get_confidence_score(classification, sample, characteristics_str)
                
                # Generate reasoning
                reasoning = self._generate_reasoning(sample, classification, confidence)
                
                self.logger.debug(f"Classified sample {sample.sample_id} as: {classification} (confidence: {confidence:.2f})")
                
                return ClassificationResult(
                    sample_id=sample.sample_id,
                    classification=classification,
                    confidence=confidence,
                    reasoning=reasoning
                )
                
            except Exception as e:
                self.logger.error(f"Error classifying sample {sample.sample_id}: {e}")
                return ClassificationResult(
                    sample_id=sample.sample_id,
                    classification="Unknown",
                    confidence=0.0,
                    reasoning=f"Classification error: {e}"
                )
    
    def _parse_classification(self, response: str, sample_id: str) -> str:
        """Parse and validate GPT classification response."""
        if not response:
            self.logger.warning(f"Empty response for sample {sample_id}")
            return "Unknown"
        
        # Clean the response
        classification = response.strip().upper()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ["CLASSIFICATION:", "ANSWER:", "RESULT:", "THE SAMPLE IS:", "BASED ON"]
        for prefix in prefixes_to_remove:
            if classification.startswith(prefix):
                classification = classification[len(prefix):].strip()
        
        # Handle multi-word responses by taking the first valid classification
        words = classification.split()
        for word in words:
            word = word.strip('.,!?;:')
            if word in self.get_supported_classifications():
                return word
        
        # Check if response contains any supported classification
        supported_classifications = self.get_supported_classifications()
        for supported in supported_classifications:
            if supported.upper() in classification:
                return supported
        
        # Fallback mapping for common variations
        classification_mapping = {
            "ADENOCARCINOMA": "ADC",
            "SQUAMOUS": "SCC", 
            "SQUAMOUS CELL": "SCC",
            "NORMAL TISSUE": "Normal",
            "HEALTHY": "Normal",
            "CONTROL": "Normal",
            "MALIGNANT": "Tumor",
            "CANCER": "Tumor",
            "CARCINOMA": "Tumor",
            "NEOPLASM": "Tumor",
            "METASTASES": "Metastatic",
            "SECONDARY": "Metastatic"
        }
        
        for variant, standard in classification_mapping.items():
            if variant in classification:
                return standard
        
        self.logger.warning(f"Unrecognized classification '{response}' for sample {sample_id}, defaulting to 'Unknown'")
        return "Unknown"
    
    async def _get_confidence_score(self, classification: str, sample: SampleInfo, characteristics_str: str) -> float:
        """Get confidence score for the classification."""
        try:
            # Create metadata summary for confidence assessment
            metadata_summary = f"Tissue: {sample.tissue_type}, Characteristics: {characteristics_str[:100]}"
            
            confidence_prompt = self.confidence_prompt_template.format(
                classification=classification,
                metadata_summary=metadata_summary
            )
            
            # Make GPT request for confidence
            response = await self._make_gpt_request(confidence_prompt, max_tokens=5)
            
            # Parse confidence score
            confidence_str = response.strip()
            try:
                confidence = float(confidence_str)
                # Ensure confidence is in valid range
                confidence = max(0.0, min(1.0, confidence))
                return confidence
            except ValueError:
                self.logger.warning(f"Invalid confidence response '{response}', using default")
                return self._calculate_heuristic_confidence(classification, sample)
                
        except Exception as e:
            self.logger.warning(f"Error getting confidence score: {e}, using heuristic")
            return self._calculate_heuristic_confidence(classification, sample)
    
    def _calculate_heuristic_confidence(self, classification: str, sample: SampleInfo) -> float:
        """Calculate confidence using heuristic rules when GPT confidence fails."""
        characteristics_str = ", ".join(sample.characteristics).lower()
        
        # High confidence indicators
        high_confidence_keywords = {
            "ADC": ["adenocarcinoma", "adc"],
            "SCC": ["squamous cell carcinoma", "scc", "squamous"],
            "Normal": ["normal", "healthy", "control", "adjacent normal"],
            "Tumor": ["tumor", "malignant", "cancer"],
            "Metastatic": ["metastatic", "metastasis", "secondary"],
            "Primary": ["primary tumor", "primary"],
            "Recurrent": ["recurrent", "relapse"]
        }
        
        # Check for high confidence keywords
        if classification in high_confidence_keywords:
            for keyword in high_confidence_keywords[classification]:
                if keyword in characteristics_str:
                    return 0.9
        
        # Medium confidence for general cancer terms
        cancer_terms = ["carcinoma", "neoplasm", "malignancy"]
        if any(term in characteristics_str for term in cancer_terms):
            return 0.7
        
        # Lower confidence for ambiguous cases
        if classification == "Unknown":
            return 0.1
        
        # Default medium-low confidence
        return 0.6
    
    def _generate_reasoning(self, sample: SampleInfo, classification: str, confidence: float) -> str:
        """Generate human-readable reasoning for the classification."""
        characteristics_str = ", ".join(sample.characteristics).lower()
        
        reasoning_parts = []
        
        # Add primary reasoning based on characteristics
        if "adenocarcinoma" in characteristics_str or "adc" in characteristics_str:
            reasoning_parts.append("adenocarcinoma mentioned in characteristics")
        elif "squamous" in characteristics_str or "scc" in characteristics_str:
            reasoning_parts.append("squamous cell carcinoma indicators found")
        elif "normal" in characteristics_str or "healthy" in characteristics_str:
            reasoning_parts.append("normal/healthy tissue indicators")
        elif "tumor" in characteristics_str or "malignant" in characteristics_str:
            reasoning_parts.append("tumor/malignant tissue indicators")
        elif "metastatic" in characteristics_str:
            reasoning_parts.append("metastatic disease indicators")
        
        # Add tissue context
        if sample.tissue_type and sample.tissue_type != "unknown":
            reasoning_parts.append(f"tissue type: {sample.tissue_type}")
        
        # Add confidence level description
        if confidence >= 0.9:
            confidence_desc = "high confidence"
        elif confidence >= 0.7:
            confidence_desc = "good confidence"
        elif confidence >= 0.5:
            confidence_desc = "moderate confidence"
        else:
            confidence_desc = "low confidence"
        
        if reasoning_parts:
            return f"GPT-4 classification based on {', '.join(reasoning_parts)} ({confidence_desc})"
        else:
            return f"GPT-4 classification with {confidence_desc} based on available metadata"
    
    async def _make_gpt_request(self, prompt: str, max_tokens: int = 50) -> str:
        """Make a request to GPT API."""
        try:
            # Use asyncio to run the sync OpenAI client in a thread
            loop = asyncio.get_event_loop()
            
            def sync_request():
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            result = await loop.run_in_executor(None, sync_request)
            return result
            
        except Exception as e:
            raise ClassificationError(f"GPT API request failed: {e}")
    
    async def classify_geo_samples(self, 
                                  samples: List[Dict[str, Any]],
                                  dataset_id: str) -> ToolResult[List[str]]:
        """
        Classify GEO samples and return unique labels (compatible with original function).
        
        Args:
            samples: List of sample dictionaries with metadata
            dataset_id: Dataset ID for context
            
        Returns:
            ToolResult with list of unique classification labels
        """
        try:
            # Convert sample dictionaries to SampleInfo objects
            sample_infos = []
            for sample in samples:
                sample_info = SampleInfo(
                    sample_id=sample.get("sample_id", "unknown"),
                    tissue_type=sample.get("tissue_type", "unknown"),
                    characteristics=sample.get("characteristics", []),
                    library_source=sample.get("library_source", "unknown"),
                    library_strategy=sample.get("library_strategy", "unknown"),
                    extraction_protocol=sample.get("extraction_protocol", "unknown"),
                    molecule=sample.get("molecule", "unknown"),
                    additional_metadata=sample
                )
                sample_infos.append(sample_info)
            
            # Classify samples
            classification_result = await self.execute(sample_infos)
            
            if not classification_result.success:
                return ToolResult(
                    success=False,
                    error=f"Sample classification failed: {classification_result.error}",
                    details={"dataset_id": dataset_id}
                )
            
            # Extract unique labels
            unique_labels = set()
            for result in classification_result.data:
                unique_labels.add(result.classification)
            
            unique_labels_list = list(unique_labels)
            
            self.logger.info(f"Dataset {dataset_id} classified samples into: {unique_labels_list}")
            
            return ToolResult(
                success=True,
                data=unique_labels_list,
                details={
                    "dataset_id": dataset_id,
                    "total_samples": len(samples),
                    "classifications": {label: sum(1 for r in classification_result.data if r.classification == label) 
                                     for label in unique_labels_list}
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"GEO sample classification failed: {e}",
                details={"dataset_id": dataset_id}
            )
    
    def validate_input(self, samples: List[SampleInfo], max_concurrent: int) -> bool:
        """Validate input parameters."""
        if not isinstance(samples, list):
            self.logger.error("samples must be a list")
            return False
        
        if not samples:
            self.logger.error("samples list cannot be empty")
            return False
        
        if not isinstance(max_concurrent, int) or max_concurrent <= 0:
            self.logger.error("max_concurrent must be a positive integer")
            return False
        
        for sample in samples:
            if not isinstance(sample, SampleInfo):
                self.logger.error("All samples must be SampleInfo objects")
                return False
            
            if not sample.sample_id:
                self.logger.error("All samples must have a sample_id")
                return False
        
        return True
    
    def get_supported_classifications(self) -> List[str]:
        """Get list of supported classification types."""
        return [
            "Normal",
            "Tumor", 
            "ADC",  # Adenocarcinoma
            "SCC",  # Squamous Cell Carcinoma
            "SCLC", # Small Cell Lung Carcinoma
            "NSCLC", # Non-Small Cell Lung Carcinoma
            "Metastatic",
            "Primary",
            "Recurrent",
            "Unknown"  # Fallback for unclear cases
        ]
    
    def validate_output(self, result: List[ClassificationResult]) -> bool:
        """Validate output format."""
        if not isinstance(result, list):
            self.logger.error("Result must be a list")
            return False
        
        for item in result:
            if not isinstance(item, ClassificationResult):
                self.logger.error("All result items must be ClassificationResult objects")
                return False
            
            if not item.sample_id:
                self.logger.error("All results must have a sample_id")
                return False
            
            if not item.classification:
                self.logger.error("All results must have a classification")
                return False
        
        return True 