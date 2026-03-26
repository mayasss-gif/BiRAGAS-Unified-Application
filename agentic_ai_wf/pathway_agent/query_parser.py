"""
Query Parser Agent for LangGraph Pathway Analysis Workflow

Handles natural language query parsing with LLM-based extraction of
structured parameters for pathway analysis pipeline.
"""

import logging
from pathlib import Path
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from .agent_state import PipelineState
from .agent_config import get_config
from .ui_logging import emit_from_state
# Extract JSON from response
import json
import re


logger = logging.getLogger(__name__)


class PathwayAnalysisRequest(BaseModel):
    """Structured request model for pathway analysis"""
    deg_file_path: str = Field(description="Path to DEG file")
    disease_name: str = Field(description="Disease name for analysis")
    patient_prefix: str = Field(description="Patient ID prefix", default="59d6bcdc-9333-4aae-894b")


def parse_user_query(state: PipelineState) -> PipelineState:
    """
    Parse natural language query or extract structured parameters
    
    Args:
        state: Current pipeline state
        
    Returns:
        Updated pipeline state with parsed parameters
    """
    emit_from_state(state, "info", "Parsing analysis parameters and validating inputs")
    logger.info("-> Parsing user query and extracting parameters")

    # If structured parameters already provided, use them directly
    if state.get('deg_file_path') and state.get('disease_name'):
        emit_from_state(state, "info", f"Using provided parameters: {state.get('disease_name')} — DEG file ready")
        logger.info("-> Structured parameters already provided, skipping query parsing")
        state['progress_messages'].append("Using provided structured parameters")
        return state
    
    # If no user query provided, use fallback
    if not state.get('user_query'):
        logger.warning("-> No user query provided, using fallback parameters")
        state['progress_messages'].append("No query provided, using fallback")
        return state
    
    try:
        # Initialize LLM for query parsing
        config = get_config()
        llm = ChatOpenAI(
            model=config.openai_model, 
            temperature=config.openai_temperature,
            api_key=config.openai_api_key
        )
        
        # Create prompt for query parsing
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a biomedical analysis assistant. Extract structured parameters from user queries about pathway analysis.

                Extract the following information:
                1. DEG file path (look for file paths, CSV files, or data files)
                2. Disease name (look for cancer types, diseases, or medical conditions)
                3. Patient prefix (if mentioned, otherwise use default)

                Return a JSON object with these fields:
                {
                    "deg_file_path": "path/to/file.csv",
                    "disease_name": "Disease Name",
                    "patient_prefix": "patient_id_prefix"
                }

                If any information is missing, use reasonable defaults or ask for clarification."""),
            ("user", "Query: {query}")
        ])
        
        # Create chain
        chain = prompt | llm
        
        # Parse the query
        response = chain.invoke({"query": state['user_query']})
        
        
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            parsed_data = json.loads(json_match.group())
            
            # Update state with parsed parameters
            state['deg_file_path'] = Path(parsed_data.get('deg_file_path', ''))
            state['disease_name'] = parsed_data.get('disease_name', '')
            state['patient_prefix'] = parsed_data.get('patient_prefix', '59d6bcdc-9333-4aae-894b')
            
            logger.info(f"-> Query parsed successfully:")
            logger.info(f"   Disease: {state['disease_name']}")
            logger.info(f"   File: {state['deg_file_path']}")
            logger.info(f"   Patient prefix: {state['patient_prefix']}")
            
            state['progress_messages'].append(f"Query parsed: {state['disease_name']} analysis")
        else:
            raise ValueError("No valid JSON found in LLM response")
            
    except Exception as e:
        logger.error(f"-> Query parsing failed: {e}")
        logger.info("-> Falling back to default parameters")
        
        # Fallback to default parameters
        state['deg_file_path'] = Path("Pancreatic Cancer_DEGs_prioritized_20250625_193700.csv")
        state['disease_name'] = "Pancreatic Cancer"
        state['patient_prefix'] = "59d6bcdc-9333-4aae-894b"
        
        state['progress_messages'].append(f"Query parsing failed, using defaults: {e}")
    
    return state
