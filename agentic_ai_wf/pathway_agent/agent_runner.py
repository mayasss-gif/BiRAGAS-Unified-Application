"""
Autonomous Agentic Workflow Orchestration for Pathway Analysis Pipeline

Main entry point for the LangGraph-based autonomous agent system that orchestrates
the complete 5-stage pathway analysis pipeline with natural language query support,
automatic retry mechanisms, and real-time progress logging.
"""

import asyncio
import logging
import time
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import openai

from .langgraph_workflow import create_pathway_analysis_workflow, create_simple_workflow
from .agent_state import PipelineState
from .agent_config import get_config, validate_config
from .ui_logging import set_pathway_ui_context, clear_pathway_ui_context

# Configure logging
def setup_logging():
    """Setup comprehensive logging configuration"""
    config = get_config()
    
    # Create logs directory if it doesn't exist
    log_dir = Path(config.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler() if config.enable_console_logging else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()


def is_causal_request(user_query: Optional[str]) -> bool:
    """Detect if the user query requests causal-focused analysis."""
    if not user_query:
        return False
    lowered = user_query.lower()
    return (
        "causal discovery" in lowered
        or "causal analysis" in lowered
        or "causal inference" in lowered
        or "causal" in lowered
    )


async def generate_analysis_summary(
    final_state: Dict[str, Any],
    disease_name: Optional[str] = None
) -> str:
    """
    Generate a comprehensive analysis summary with clinical and functional insights
    from the pathway analysis results using LLM.
    
    Args:
        final_state: The final state from the pathway analysis workflow
        disease_name: Name of the disease being analyzed
        
    Returns:
        A paragraph summary with clinical and functional insights
    """
    try:
        # Extract key data from the final state
        consolidation_output = final_state.get('consolidation_output')
        completed_stages = final_state.get('completed_stages', [])
        processing_time = final_state.get('total_processing_time', 0)
        
        # If no consolidation output, return basic summary
        if not consolidation_output or not Path(consolidation_output).exists():
            return f"Pathway analysis completed for {disease_name or 'the specified disease'} with {len(completed_stages)} stages processed in {processing_time:.2f} seconds. Analysis results are being generated."
        
        # Read the consolidation results
        try:
            df = pd.read_csv(consolidation_output)
            
            # Extract key statistics
            total_pathways = len(df)
            top_pathways = df.head(10) if total_pathways > 0 else df
            
            # Get pathway categories distribution
            if 'Main_Class' in df.columns:
                category_dist = df['Main_Class'].value_counts().to_dict()
            else:
                category_dist = {}
            
            # Get top scoring pathways
            if 'LLM_Score' in df.columns:
                avg_score = df['LLM_Score'].mean()
                max_score = df['LLM_Score'].max()
                high_confidence = len(df[df.get('Confidence_Level', '') == 'High'])
            else:
                avg_score = max_score = high_confidence = 0
            
            # Get clinical relevance insights
            clinical_pathways = []
            if 'Clinical_Relevance' in df.columns:
                clinical_pathways = df[df['Clinical_Relevance'].notna() & (df['Clinical_Relevance'] != '')]['Pathway_Name'].tolist()[:5]
            
            # Get functional relevance insights  
            functional_pathways = []
            if 'Functional_Relevance' in df.columns:
                functional_pathways = df[df['Functional_Relevance'].notna() & (df['Functional_Relevance'] != '')]['Pathway_Name'].tolist()[:5]
            
        except Exception as e:
            logger.warning(f"Could not read consolidation results: {e}")
            return f"Pathway analysis completed for {disease_name or 'the specified disease'} with {len(completed_stages)} stages processed in {processing_time:.2f} seconds. Results are available in the output file."
        
        # Prepare data for LLM analysis
        analysis_data = {
            'disease_name': disease_name or 'the analyzed disease',
            'total_pathways': total_pathways,
            'processing_time': processing_time,
            'completed_stages': completed_stages,
            'category_distribution': category_dist,
            'average_score': avg_score,
            'max_score': max_score,
            'high_confidence_pathways': high_confidence,
            'top_pathways': top_pathways[['Pathway_Name', 'LLM_Score', 'Clinical_Relevance', 'Functional_Relevance']].to_dict('records')[:5] if len(top_pathways) > 0 else [],
            'clinical_pathways': clinical_pathways,
            'functional_pathways': functional_pathways
        }
        
        # Create LLM prompt for analysis summary
        prompt = f"""
        As a biomedical expert, analyze the following pathway analysis results and provide a comprehensive summary paragraph highlighting the most clinically and functionally relevant insights.

        Analysis Data:
        - Disease: {analysis_data['disease_name']}
        - Total pathways analyzed: {analysis_data['total_pathways']}
        - Processing time: {analysis_data['processing_time']:.2f} seconds
        - Completed stages: {', '.join(analysis_data['completed_stages'])}
        
        Pathway Categories Distribution:
        {analysis_data['category_distribution']}
        
        Scoring Statistics:
        - Average LLM Score: {analysis_data['average_score']:.2f}
        - Maximum Score: {analysis_data['max_score']:.2f}
        - High Confidence Pathways: {analysis_data['high_confidence_pathways']}
        
        Top Pathways:
        {analysis_data['top_pathways']}
        
        Clinically Relevant Pathways:
        {analysis_data['clinical_pathways']}
        
        Functionally Relevant Pathways:
        {analysis_data['functional_pathways']}

        Please provide a paragraph summary that:
        1. Highlights the most significant clinical insights and therapeutic implications
        2. Discusses the functional relevance of key pathways to disease mechanisms
        3. Identifies the most promising pathways for further investigation
        4. Mentions any notable patterns in pathway categories or scoring

        IMPORTANT: Do not mention about the LLM score directly make it like "The most clinically relevant pathways are..." or "the confidence level".

        Focus on actionable insights that would be valuable for researchers and clinicians.
        """
        
        # Get OpenAI client
        config = get_config()
        client = openai.OpenAI(api_key=config.openai_api_key)
        
        # Generate summary using LLM
        response = client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {"role": "system", "content": "You are a biomedical expert specializing in pathway analysis and disease mechanisms. Provide clear, actionable insights for researchers and clinicians."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content.strip()
        logger.info("Analysis summary generated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Failed to generate analysis summary: {e}")
        return f"Pathway analysis completed for {disease_name or 'the specified disease'} with {len(final_state.get('completed_stages', []))} stages processed. Detailed results are available in the output files."


async def run_autonomous_analysis(
    user_query: str = None,
    deg_file_path: Optional[Path] = None,
    disease_name: Optional[str] = None,
    patient_prefix: str = "",
    output_dir: Optional[Path] = None,
    max_retries: int = 3,
    use_simple_workflow: bool = False,
    causal: bool = False,
    workflow_logger: Optional[Any] = None,
    event_loop: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run autonomous pathway analysis workflow using LangGraph orchestration
    
    Args:
        user_query: Natural language query (e.g., "Analyze melanoma DEG data from file X")
        deg_file_path: Direct file path (fallback if query parsing fails)
        disease_name: Direct disease name (fallback if query parsing fails)
        patient_prefix: Patient ID prefix
        output_dir: Directory path for output files (used by enrichment_pipeline)
        max_retries: Maximum retry attempts per stage
        use_simple_workflow: Use simplified workflow without complex routing
        causal: When True or when the query mentions causal analysis/discovery,
                stop the pipeline after enrichment
        
    Returns:
        Dictionary containing final state and results
    """
    
    # Validate configuration
    if not validate_config():
        raise ValueError("Configuration validation failed")
    
    causal_mode = causal or is_causal_request(user_query)
    config = get_config()
    start_time = time.time()

    # Run-scoped UI logging context (workflow_logger, event_loop)
    pathway_ui_key: Optional[str] = None
    if workflow_logger and event_loop:
        pathway_ui_key = f"pathway_{int(time.time())}_{id(workflow_logger) % 100000}"
        set_pathway_ui_context(pathway_ui_key, workflow_logger, event_loop)
        try:
            await workflow_logger.info(
                agent_name="Pathway Enrichment Agent",
                message=f"Starting pathway pipeline for {disease_name or 'disease'} — enrichment → deduplication → categorization → literature → consolidation",
                step="pathway_enrichment"
            )
        except Exception:
            pass

    logger.info("->Starting Autonomous Pathway Analysis Workflow")
    logger.info(f"Query: {user_query or f'{disease_name} - {deg_file_path}'}")
    logger.info(
        f"Configuration: max_retries={max_retries}, simple_workflow={use_simple_workflow}, causal_mode={causal_mode}"
    )
    
    # Initialize state
    initial_state: Dict[str, Any] = {
        "user_query": user_query or "",
        "deg_file_path": deg_file_path,
        "disease_name": disease_name,
        "patient_prefix": patient_prefix,
        "output_dir": output_dir,
        "enrichment_output": None,
        "deduplication_output": None,
        "categorization_output": None,
        "literature_output": None,
        "consolidation_output": None,
        "causal": causal_mode,
        "errors": [],
        "retry_counts": {},
        "max_retries": max_retries,
        "current_stage": "parse_query",
        "completed_stages": [],
        "progress_messages": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "total_processing_time": None,
        "_pathway_ui_ctx_key": pathway_ui_key,
    }
    
    if causal_mode:
        logger.info("-> Causal mode detected: pipeline will stop after enrichment stage")
        initial_state['progress_messages'].append(
            "Causal mode enabled: pipeline will stop after enrichment"
        )
    
    # Create workflow
    try:
        if use_simple_workflow:
            logger.info("-> Using simplified workflow (no retry logic)")
            app = create_simple_workflow()
        else:
            logger.info("-> Using full workflow with retry logic and checkpointing")
            app = create_pathway_analysis_workflow()
    except Exception as e:
        logger.error(f"-> Failed to create workflow: {e}")
        raise
    
    # Configuration for LangGraph execution
    config_dict = {"configurable": {"thread_id": f"pathway_analysis_{int(time.time())}"}}

    try:
        # Execute workflow
        logger.info("-> Executing LangGraph workflow...")
        final_state = await app.ainvoke(initial_state, config=config_dict)
        
        # Calculate processing time
        end_time = time.time()
        total_time = end_time - start_time
        final_state['end_time'] = datetime.now().isoformat()
        final_state['total_processing_time'] = total_time
        
        # Log results
        if final_state['current_stage'] == 'completed':
            logger.info("-> Pipeline completed successfully!")
            logger.info(f"-> Final output: {final_state['consolidation_output']}")
            logger.info(f"-> Completed stages: {final_state['completed_stages']}")
            logger.info(f"-> Total processing time: {total_time:.2f} seconds")
            
            # Log progress messages
            for msg in final_state['progress_messages']:
                logger.info(f"-> {msg}")
                
        else:
            logger.error("-> Pipeline failed")
            logger.error(f"-> Final stage: {final_state['current_stage']}")
            logger.error(f"-> Errors: {len(final_state['errors'])} total")
            
            # Log error details
            for error in final_state['errors']:
                logger.error(f"   Stage: {error['stage']}, Error: {error['error']}")
        
        # Generate analysis summary
        logger.info("-> Generating analysis summary...")
        analysis_summary = await generate_analysis_summary(final_state, disease_name)
        logger.info("-> Analysis summary generated")
        
        output_file = (
            final_state.get("consolidation_output")
            or final_state.get("enrichment_output")
        )

        return {
            "success": final_state["current_stage"] == "completed",
            "final_state": final_state,
            "processing_time": total_time,
            "output_file": output_file,
            "enrichment_output": final_state.get("enrichment_output"),
            "errors": final_state["errors"],
            "completed_stages": final_state["completed_stages"],
            "analysis_summary": analysis_summary,
        }
        
    except Exception as e:
        logger.error(f"-> Workflow execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time,
            "analysis_summary": f"Analysis failed due to error: {str(e)}. Please check the logs for more details."
        }
    finally:
        if pathway_ui_key:
            clear_pathway_ui_context(pathway_ui_key)


if __name__ == "__main__":
    # Example usage
    print("🤖 Autonomous Pathway Analysis Agent")
    print("=" * 50)
    
   
    # Example 
    result = asyncio.run(run_autonomous_analysis(
        user_query="Perform enrichment analysis for melanoma disease using the DEG file",
        deg_file_path=Path(r"C:\Users\Raafeh\Downloads\Melanoma_DEGs_prioritized.csv"),
        disease_name="Melanoma",
        patient_prefix="59d6bcdc-9333-4aae-894b",
        output_dir=Path("./ALL_ANALYSIS")  # Optional: specify output directory
    ))
    
    print(f"\n🎯 Analysis Result: {'Success' if result['success'] else 'Failed'}")
    if result['success']:
        print(f"📁 Output: {result['output_file']}")
        print(f"⏱️  Time: {result['processing_time']:.2f}s")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown')}")
    
    print(f"\n📊 Analysis Summary:")
    print(f"{result.get('analysis_summary', 'Summary not available')}")

    print(result)
