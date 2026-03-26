"""Main entry point for Pathway Prioritization System"""
from pathlib import Path

from .pathway_prioritization.core import PathwayPrioritizer, PathwayDataProcessor
from .pathway_prioritization.models import ProcessingConfig
from .pathway_prioritization.utils import setup_logging
from .pathway_prioritization.config import settings

def cleanup_directory(directory: Path):
    """Clean up directory if it exists as a file"""
    if directory.exists() and directory.is_file():
        print(f"⚠️  Removing file {directory} to create directory")
        directory.unlink()
    elif directory.exists() and directory.is_dir():
        # Directory already exists, that's fine
        pass

async def main(categoriezed_pathways_path: Path = None, disease_name: str = None, output_dir_path: Path = None):
    """Main function for pathway prioritization"""
    
    # Configuration
    pathways_file = categoriezed_pathways_path
    disease_name = disease_name
    output_dir = output_dir_path
    
    # Clean up directories first
    try:
        cleanup_directory(output_dir)
    except Exception as e:
        print(f"❌ Error cleaning up directories: {e}")
        return None
    
    # Validate settings
    try:
        settings.validate()
        settings.setup_directories()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please set OPENAI_API_KEY environment variable or create a .env file")
        return None
    
    # Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating output directory {output_dir}: {e}")
        return None
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    if not pathways_file.exists():
        print(f"❌ Error: File not found: {pathways_file}")
        print("Please check the file path and try again")
        return None
    
    print("🚀 Starting Pathway Prioritization System")
    print("=" * 50)
    print(f"📁 Input file: {pathways_file}")
    print(f"🎯 Disease: {disease_name}")
    print(f"📊 Output directory: {output_dir}")
    print("=" * 50)
    
    try:
        # Initialize system
        prioritizer = PathwayPrioritizer(disease_name=disease_name)
        processor = PathwayDataProcessor(prioritizer)
        
        # Create processing config
        config = ProcessingConfig(
            batch_size=10,
            top_n_pathways=100,
            max_workers=5,
            output_dir=output_dir
        )
        
        # Load pathway data
        print("📥 Loading pathway data...")
        all_pathways = processor.load_pathway_data(file_path=pathways_file)
        
        if not all_pathways:
            print("❌ No pathways loaded from file")
            print("Please check the CSV file format and required columns")
            return None
        
        print(f"✅ Loaded {len(all_pathways)} pathways")
        
        # Process pathway prioritization
        print("🔬 Starting pathway processing with LLM evaluation...")
        print("This may take several minutes depending on the number of pathways...")
        
        results = await processor.process_pathway_prioritization(
            pathways_file=pathways_file,
            disease_name=disease_name,
            output_dir=output_dir,
            config=config
        )
        
        final_output_file = results["output_file"]
        summary = results["summary"]
        
        print("\n" + "="*50)
        print("🎉 PROCESSING COMPLETED")
        print("="*50)
        print(f"✅ Results saved to: {final_output_file}")
        print(f"🏆 Top pathway: {summary['top_pathways'][0]['pathway_name']}")
        print(f"   Score: {summary['top_pathways'][0]['llm_score']} | Confidence: {summary['top_pathways'][0]['confidence']}")
        print(f"📈 Average score: {summary['average_score']:.2f}")
        print(f"📊 Total pathways: {summary['total_pathways_analyzed']}")
        
        # Show score distribution
        dist = summary['score_distribution']
        print(f"\n📈 Score Distribution:")
        print(f"  🏅 High (90-100): {dist['high_score_90_plus']} pathways")
        print(f"  ✅ Good (70-89): {dist['good_score_70_89']} pathways")
        print(f"  🔶 Moderate (50-69): {dist['moderate_score_50_69']} pathways")
        print(f"  📉 Low (<50): {dist['low_score_below_50']} pathways")
        
        return final_output_file
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        return None
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        print("\nDetailed error traceback:")
        traceback.print_exc()
        return None