# src/pathway_prioritization/cli/commands.py
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from ..core import PathwayPrioritizer, PathwayDataProcessor
from ..models import ProcessingConfig
from ..utils import setup_logging
from ..config import settings

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Pathway Prioritization System - Score and prioritize biological pathways using LLM evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  pathway-prioritization --input pathways.csv --disease "Lupus Cancer"
  
  # Custom output and workers
  pathway-prioritization -i pathways.csv -d "Pancreatic Cancer" -o ./results --max-workers 10
  
  # With specific batch size and top N pathways
  pathway-prioritization -i pathways.csv -d "Breast Cancer" --batch-size 5 --top-n 50
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i", 
        type=Path, 
        required=True,
        help="Path to input CSV file with pathway data"
    )
    
    parser.add_argument(
        "--disease", "-d", 
        type=str, 
        required=True,
        help="Disease name for analysis (e.g., 'Lupus Cancer', 'Pancreatic Cancer')"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o", 
        type=Path, 
        default=settings.output_dir,
        help=f"Output directory for results (default: {settings.output_dir})"
    )
    
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=settings.max_workers,
        help=f"Maximum number of parallel workers (default: {settings.max_workers})"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=settings.batch_size,
        help=f"Batch size for processing (default: {settings.batch_size})"
    )
    
    parser.add_argument(
        "--top-n", 
        type=int, 
        default=100,
        help="Number of top pathways to select (default: 100)"
    )
    
    parser.add_argument(
        "--kegg-only",
        action="store_true",
        help="Apply LLM processing only to KEGG pathways"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Pathway Prioritization System v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Run the analysis
    asyncio.run(run_analysis(args))

async def run_analysis(args):
    """Run the pathway analysis"""
    try:
        # Validate input file
        if not args.input.exists():
            print(f"❌ Error: Input file not found: {args.input}")
            sys.exit(1)
        
        # Validate settings
        try:
            settings.validate()
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            print("Please set OPENAI_API_KEY environment variable or in .env file")
            sys.exit(1)
        
        # Setup directories
        settings.setup_directories()
        
        # Setup logging level
        log_level = "DEBUG" if args.verbose else settings.log_level
        logger = setup_logging(args.output, level=log_level)
        
        print("🚀 Starting Pathway Prioritization System")
        print("=" * 50)
        print(f"📁 Input file: {args.input}")
        print(f"🎯 Disease: {args.disease}")
        print(f"📊 Output directory: {args.output}")
        print(f"👷 Max workers: {args.max_workers}")
        print(f"📦 Batch size: {args.batch_size}")
        print(f"🏆 Top N pathways: {args.top_n}")
        print(f"🔬 KEGG only: {args.kegg_only}")
        print("=" * 50)
        
        # Initialize system
        prioritizer = PathwayPrioritizer(disease_name=args.disease)
        prioritizer.apply_llm_to_kegg_only = args.kegg_only
        
        processor = PathwayDataProcessor(prioritizer)
        
        # Create processing config
        config = ProcessingConfig(
            batch_size=args.batch_size,
            top_n_pathways=args.top_n,
            max_workers=args.max_workers,
            output_dir=args.output,
            apply_llm_to_kegg_only=args.kegg_only
        )
        
        # Process pathway prioritization
        results = await processor.process_pathway_prioritization(
            pathways_file=args.input,
            disease_name=args.disease,
            output_dir=args.output,
            config=config
        )
        
        # Print comprehensive summary
        summary = results["summary"]
        print("\n" + "="*60)
        print("🎉 PATHWAY PRIORITIZATION - COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"🏥 Disease: {summary['disease_name']}")
        print(f"📊 Total pathways analyzed: {summary['total_pathways_analyzed']}")
        print(f"⭐ Average LLM score: {summary['average_score']:.2f}")
        print(f"💾 Output file: {summary['output_file']}")
        
        # Score distribution
        dist = summary['score_distribution']
        print(f"\n📈 Score Distribution:")
        print(f"  🏅 High (90-100): {dist['high_score_90_plus']} pathways")
        print(f"  ✅ Good (70-89): {dist['good_score_70_89']} pathways")
        print(f"  🔶 Moderate (50-69): {dist['moderate_score_50_69']} pathways")
        print(f"  📉 Low (<50): {dist['low_score_below_50']} pathways")
        
        # Top pathways
        print(f"\n🏆 Top 5 Prioritized Pathways:")
        for i, pathway in enumerate(summary['top_pathways'][:5], 1):
            print(f"  {i}. {pathway['pathway_name']}")
            print(f"     Score: {pathway['llm_score']} | Confidence: {pathway['confidence']} | P-value: {pathway['p_value']:.2e}")
            
        print(f"\n✅ Analysis completed successfully!")
        print(f"📋 Detailed results saved to: {summary['output_file']}")
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()