#!/usr/bin/env python3
"""
Main entry point for CRISPR Perturb-seq pipeline.

This demonstrates how to call the pipeline programmatically with parameter discovery.
"""

from pathlib import Path

from agentic_ai_wf.crispr_pipeline_agent.crispr import (
    run_pipeline,
    discover_pipeline_inputs,
    get_available_samples,
    get_metadata_groups,
    get_scrna_config_options,
)



def example_discover_and_run():
    """
    Example: Discover available parameters before running the pipeline.
    
    This approach is recommended to avoid errors from invalid parameter values.
    """
    # Define input path
    input_data = Path("input_data")
    gse_dir = input_data / "GSE90546_RAW"
    
    # ==================================================
    # Step 1: Discover all available inputs and options
    # ==================================================
    print("=" * 60)
    print("DISCOVERING PIPELINE INPUTS...")
    print("=" * 60)
    
    try:
        info = discover_pipeline_inputs(gse_dir)
        
        print(f"\n📁 GSE: {info['gse_name']}")
        print(f"   Path: {info['gse_path']}")
        
        print(f"\n🧬 Available Samples ({len(info['samples'])}):")
        for sample in info['samples']:
            print(f"   - {sample}")
        
        if info['metadata']['available']:
            print(f"\n📊 Metadata Groups:")
            for group in info['metadata']['groups']:
                count = info['metadata']['group_counts'][group]
                print(f"   - {group}: {count} samples")
            
            print(f"\n💡 Suggested Configuration:")
            print(f"   Control: {info['metadata']['suggested_control']}")
            print(f"   Disease: {info['metadata']['suggested_disease']}")
        else:
            print(f"\n⚠️  No metadata.csv found (condition analysis not available)")
        
        print(f"\n🔧 Available Integration Methods:")
        for method in info['config_options']['integration_methods']['available']:
            desc = info['config_options']['integration_methods']['descriptions'][method]
            print(f"   - {method}: {desc}")
        
        print(f"\n🏷️  Available Annotation Engines:")
        for engine in info['config_options']['annotation_engines']['available']:
            print(f"   - {engine}")
        
        print(f"\n🤖 Available ML Models:")
        for model in info['config_options']['ml_models']['available']:
            desc = info['config_options']['ml_models']['descriptions'][model]
            print(f"   - {model}: {desc}")
        
        # ==================================================
        # Step 2: Run pipeline with discovered parameters
        # ==================================================
        print("\n" + "=" * 60)
        print("RUNNING PIPELINE WITH DISCOVERED PARAMETERS...")
        print("=" * 60 + "\n")
        
        run_pipeline(
            input_gse_dirs=gse_dir,
            samples=info['samples'],  # Use all discovered samples
            output_dir=Path("crispr_output"),
            full_bn=True,
            models="xgb,rf",  # Valid models from discovery
            parallel_training=True,
            run_scrna=True,
            scrna_config={
                "conditions": {
                    "enabled": info['metadata']['available'],
                    "control_groups": [info['metadata']['suggested_control']] 
                        if info['metadata']['available'] else [],
                    "disease_groups": info['metadata']['suggested_disease'] 
                        if info['metadata']['available'] else [],
                },
                "integration": {
                    "methods": ["harmony", "scvi"]  # From discovered options
                },
            } if info['metadata']['available'] else None
        )
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nPlease ensure the GSE directory exists:")
        print(f"  {gse_dir}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def example_manual_discovery():
    """
    Example: Use individual helper functions for targeted discovery.
    """
    input_data = Path("input_data")
    gse_dir = input_data / "GSE90546_RAW"
    
    print("=" * 60)
    print("MANUAL PARAMETER DISCOVERY")
    print("=" * 60)
    
    # Discover samples only
    print("\n1️⃣  Discovering samples...")
    samples = get_available_samples(gse_dir)
    print(f"   Found {len(samples)} samples: {samples[:3]}..." if len(samples) > 3 else f"   {samples}")
    
    # Discover metadata groups
    print("\n2️⃣  Discovering metadata groups...")
    metadata = get_metadata_groups(gse_dir)
    if metadata['available']:
        print(f"   Groups: {metadata['groups']}")
        print(f"   Suggested control: {metadata['suggested_control']}")
        print(f"   Suggested disease: {metadata['suggested_disease']}")
    else:
        print("   No metadata.csv found")
    
    # Get configuration options
    print("\n3️⃣  Getting configuration options...")
    options = get_scrna_config_options()
    print(f"   Integration methods: {options['integration_methods']['available']}")
    print(f"   Annotation engines: {options['annotation_engines']['available']}")
    print(f"   ML models: {options['ml_models']['available']}")
    
    # Use discovered values
    print("\n4️⃣  Running pipeline with discovered values...")
    run_pipeline(
        input_gse_dirs=gse_dir,
        samples=samples[0:2],  # First 2 samples
        output_dir=Path("results"),
        models="xgb,rf",
        run_scrna=True,
        scrna_config={
            "conditions": {
                "enabled": metadata['available'],
                "control_groups": [metadata['suggested_control']] if metadata['available'] else [],
                "disease_groups": metadata['suggested_disease'] if metadata['available'] else [],
            }
        } if metadata['available'] else None
    )


def example_error_handling():
    """
    Example: Demonstrate error handling with invalid parameters.
    
    This shows what happens when you provide incorrect values.
    """
    input_data = Path("input_data")
    gse_dir = input_data / "GSE90546_RAW"
    
    print("=" * 60)
    print("ERROR HANDLING EXAMPLES")
    print("=" * 60)
    
    # Example 1: Invalid sample
    print("\n❌ Example 1: Invalid sample ID")
    try:
        run_pipeline(
            input_gse_dirs=gse_dir,
            samples=["INVALID_SAMPLE_ID"],
            output_dir=Path("results")
        )
    except ValueError as e:
        print(f"Caught error:\n{e}")
    
    # Example 2: Invalid model
    print("\n❌ Example 2: Invalid model")
    try:
        run_pipeline(
            input_gse_dirs=gse_dir,
            samples="all",
            models="xgb,invalid_model",
            output_dir=Path("results")
        )
    except ValueError as e:
        print(f"Caught error:\n{e}")
    
    # Example 3: Invalid integration method
    print("\n❌ Example 3: Invalid integration method")
    try:
        run_pipeline(
            input_gse_dirs=gse_dir,
            samples="all",
            output_dir=Path("results"),
            scrna_config={
                "integration": {
                    "method": "invalid_method"
                }
            }
        )
    except ValueError as e:
        print(f"Caught error:\n{e}")
    
    # Example 4: Invalid metadata group
    print("\n❌ Example 4: Invalid metadata group")
    try:
        run_pipeline(
            input_gse_dirs=gse_dir,
            samples="all",
            output_dir=Path("results"),
            scrna_config={
                "conditions": {
                    "enabled": True,
                    "control_groups": ["NonExistentGroup"],
                    "disease_groups": []
                }
            }
        )
    except ValueError as e:
        print(f"Caught error:\n{e}")


def main():
    """
    Main entry point - choose which example to run.
    """
    # ==================================================
    # Traditional examples (without discovery)
    # ==================================================
    
    # Example: Process all samples with default scRNA config
    # Report generation is enabled by default — set OPENAI_API_KEY in .env
    # for LLM-powered scientific interpretations in the HTML report.
    gse_dir = Path("agentic_ai_wf/crispr_pipeline_agent/input_data/GSE90546_RAW")
    run_pipeline(
        input_gse_dirs=gse_dir,
        samples=["GSM2406675_10X001"],
        output_dir=Path("agentic_ai_wf/shared/crispr_data/"),
        generate_report=True,
    )


if __name__ == "__main__":
    main()
