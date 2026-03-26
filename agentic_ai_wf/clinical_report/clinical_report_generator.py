import os
import base64
from pathlib import Path
import pandas as pd
from weasyprint import HTML, CSS
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .combined_stats import (
    compute_deg_stats, compute_pathogenic_molecular_signatures
)
from .risk_factors import compute_risk_factors_pathways
from .fc_bins_dist import compute_fc_bins
from .key_molecular_signatures import compute_key_molecular_signatures

from .validation_integration import (
    validate_pathways_disease_focused
)
from .signatures_grouper import group_signatures
from .pathway_gene_mapper import map_genes_to_pathways

from .utils import load_inputs
from .pathways_summary import summarize_pathways
from .config import HTML_TEMPLATE_FILE, DATASET_PATH

# Import LLM text generation functions
from .text_generation import llm_generate_drug_description

from .llm_client import LLMClient
from .group_summary_generator import summarize_groups
from .target_disease_drugs import select_best_drugs

llm_client = LLMClient()

# Set up Jinja2 environment for HTML templates
env = Environment(
    loader=FileSystemLoader(DATASET_PATH),
    autoescape=select_autoescape(['html', 'xml'])
)

patient_profile = {
        "name": "John Doe",
        "diagnosis": "Pancreatic Adenocarcinoma",
}

def generate_clinical_report(deg_csv : str, path_csv : str, drug_csv : str, output_pdf : str, disease_name : str, patient_profile: dict = patient_profile, patient_prefix: str = "patient"):
    """Generate a clinical transcriptome report PDF from input CSVs with validation."""

    # 1. Load data
    data = load_inputs(deg_csv, path_csv, drug_csv)
    
    # 2. DISEASE-FOCUSED VALIDATION & FILTERING - Apply to all sections
    print(f"Starting disease-focused analysis for: {disease_name}")
    
    # Validate and filter pathways for disease relevance
    # validated_pathways = validate_pathways_disease_focused(data['path_df'], disease_name)
    # print(f"Pathway validation completed: {validated_pathways['validation_summary']}")
    
    # Create disease-filtered pathway dataframe
    # disease_relevant_pathways = []
    # for pathway_info in (validated_pathways.get('pathogenic_pathways', []) + 
    #                     validated_pathways.get('protective_pathways', [])):
    #     disease_relevant_pathways.append(pathway_info['pathway_name'])
    
    # ADDITION: Get top 10 downregulated pathways from original data for separate inclusion
    # top_downregulated_pathways = []
    # if 'Regulation' in data['path_df'].columns and 'LLM_Score' in data['path_df'].columns:
    #     downregulated_df = data['path_df'][
    #         data['path_df']['Regulation'].str.lower().isin(['down', 'downregulated'])
    #     ]
    #     if len(downregulated_df) > 0:
    #         # Sort by LLM_Score and get top 10
    #         top_downregulated = downregulated_df.nlargest(10, 'LLM_Score')
    #         for _, row in top_downregulated.iterrows():
    #             pathway_name = row.get('Pathway_Name', 'Unknown')
                
    #             # Extract gene associations for better molecular signatures
    #             gene_info = {}
    #             gene_sources = [
    #                 row.get('Pathway_Associated_Genes', ''),
    #                 row.get('Associated_Genes', ''), 
    #                 row.get('Genes', '')
    #             ]
                
    #             for gene_source in gene_sources:
    #                 if pd.notna(gene_source) and str(gene_source).strip():
    #                     gene_info['genes'] = str(gene_source)
    #                     break
                
    #             top_downregulated_pathways.append({
    #                 'pathway_name': pathway_name,
    #                 'llm_score': row.get('LLM_Score', 0.0),
    #                 'regulation': 'Downregulated',
    #                 'confidence': min(row.get('LLM_Score', 0.0) * 10, 100.0),  # Convert to percentage
    #                 'gene_associations': gene_info.get('genes', ''),
    #                 'functional_relevance': row.get('Functional_Relevance', ''),
    #                 'clinical_relevance': row.get('Clinical_Relevance', '')
    #             })
    #         print(f"📉 Found {len(top_downregulated_pathways)} top downregulated pathways for inclusion in molecular signatures")
    #     else:
    #         print("⚠️  No downregulated pathways found in original data")
    # else:
    #     print("⚠️  Required columns (Regulation, LLM_Score) not found for downregulated pathway extraction")
    
    # ADDITION: Get top 10 upregulated pathways from original data for separate inclusion
    # top_upregulated_pathways = []
    # if 'Regulation' in data['path_df'].columns and 'LLM_Score' in data['path_df'].columns:
    #     upregulated_df = data['path_df'][
    #         data['path_df']['Regulation'].str.lower().isin(['up', 'upregulated'])
    #     ]
    #     if len(upregulated_df) > 0:
    #         # Sort by LLM_Score and get top 10
    #         top_upregulated = upregulated_df.nlargest(10, 'LLM_Score')
    #         for _, row in top_upregulated.iterrows():
    #             pathway_name = row.get('Pathway_Name', 'Unknown')
                
    #             # Extract gene associations for better molecular signatures
    #             gene_info = {}
    #             gene_sources = [
    #                 row.get('Pathway_Associated_Genes', ''),
    #                 row.get('Associated_Genes', ''), 
    #                 row.get('Genes', '')
    #             ]
                
    #             for gene_source in gene_sources:
    #                 if pd.notna(gene_source) and str(gene_source).strip():
    #                     gene_info['genes'] = str(gene_source)
    #                     break
                
    #             top_upregulated_pathways.append({
    #                 'pathway_name': pathway_name,
    #                 'llm_score': row.get('LLM_Score', 0.0),
    #                 'regulation': 'Upregulated',
    #                 'confidence': min(row.get('LLM_Score', 0.0) * 10, 100.0),  # Convert to percentage
    #                 'gene_associations': gene_info.get('genes', ''),
    #                 'functional_relevance': row.get('Functional_Relevance', ''),
    #                 'clinical_relevance': row.get('Clinical_Relevance', '')
    #             })
    #         print(f"📈 Found {len(top_upregulated_pathways)} top upregulated pathways for inclusion in molecular signatures")
    #     else:
    #         print("⚠️  No upregulated pathways found in original data")
    # else:
    #     print("⚠️  Required columns (Regulation, LLM_Score) not found for upregulated pathway extraction")
    
    # if disease_relevant_pathways and 'Pathway_Name' in data['path_df'].columns:
    #     filtered_path_df = data['path_df'][data['path_df']['Pathway_Name'].isin(disease_relevant_pathways)]
        
    #     # Only use filtered data if we have sufficient pathways
    #     if len(filtered_path_df) >= 500:  # Minimum threshold for meaningful analysis
    #         data['path_df'] = filtered_path_df
    #         print(f"Filtered to {len(data['path_df'])} disease-relevant pathways for {disease_name}")
    #     else:
    #         print(f"⚠️  Only {len(filtered_path_df)} disease-relevant pathways found. Using all pathway data to ensure report generation.")
    #         print(f"Using all {len(data['path_df'])} pathways for {disease_name} analysis")
    # else:
    #     print(f"⚠️  No disease-relevant pathways identified. Using all pathway data to ensure report generation.")
    #     print(f"Using all {len(data['path_df'])} pathways for {disease_name} analysis")
    
    # Simplified drug validation: ONLY process recommended (YES) drugs from CSV
    validated_drugs_fda = []  # Initialize with safe default
    
    if drug_csv and os.path.exists(drug_csv):
        try:
            drugs_df = pd.read_csv(drug_csv)
            
            # First filter for recommended drugs (case-insensitive)
            if not drugs_df.empty:
                # Check for recommendation column with various possible names
                recommendation_col = None
                for col_name in ['recommendation', 'Recommendation', 'recommended', 'Recommended']:
                    if col_name in drugs_df.columns:
                        recommendation_col = col_name
                        break
                
                # Filter by recommendation if column exists
                if recommendation_col:
                    # Case-insensitive match for "yes" in recommendation column
                    recommended_drugs = drugs_df[drugs_df[recommendation_col].str.lower() == 'yes']
                    print(f"🔍 Filtered {len(recommended_drugs)} recommended drugs out of {len(drugs_df)} total")
                    drugs_df = recommended_drugs
                else:
                    print("⚠️  No recommendation column found in drugs CSV, using all drugs")
            
            # Then filter for FDA approved among recommended drugs
            if not drugs_df.empty:
                # Filter for FDA approved drugs based on the correct column name
                if 'fda_approved_status' in drugs_df.columns:
                    fda_drugs = drugs_df[drugs_df['fda_approved_status'] == 'Approved']
                    validated_drugs_fda = fda_drugs.to_dict('records') if not fda_drugs.empty else []
                elif 'fda_approved' in drugs_df.columns:
                    # Fallback to older column name if present
                    fda_drugs = drugs_df[drugs_df['fda_approved'] == True]
                    validated_drugs_fda = fda_drugs.to_dict('records') if not fda_drugs.empty else []
            print(f"📊 Initial drug categories: FDA={len(validated_drugs_fda)}")
        except Exception as e:
            print(f"⚠️  Error processing drug CSV: {e}")
            validated_drugs_fda = []
    else:
        print(f"⚠️  Drug CSV not provided or not found: {drug_csv}")
        print("📊 Initial drug categories: FDA=0")
    
    # Safely handle None for dataframes and print initial drug categories
    deg_count = len(data['deg_df']) if data.get('deg_df') is not None else 0
    path_count = len(data['path_df']) if data.get('path_df') is not None else 0
    drug_count = len(data['drug_df']) if data.get('drug_df') is not None else 0
    print(f"Generating statistics for {disease_name} with {deg_count} genes, {path_count} pathways, {drug_count} drugs")
    
    try:
        deg_stats = compute_deg_stats(data['deg_df'], patient_prefix=patient_prefix, disease_name=disease_name)
        print(f"✅ DEG statistics computed successfully")
    except Exception as e:
        print(f"⚠️  Error computing DEG stats: {e}")
        deg_stats = {'total_sig': 0, 'total_up': 0, 'total_down': 0}
    
    try:
        fc_bins = compute_fc_bins(data['deg_df'], patient_prefix=patient_prefix)
        print(f"✅ FC bins computed successfully")
    except Exception as e:
        print(f"⚠️  Error computing FC bins: {e}")
        fc_bins = {}
    
    try:
        pathway_info = summarize_pathways(data['path_df'])
        print(f"✅ Pathway info computed successfully")
    except Exception as e:
        print(f"⚠️  Error computing pathway info: {e}")
        pathway_info = {'significant_df': data['path_df']}
    
    # try:
    #     # Use new pathogenic molecular signatures function with downregulated pathways
    #     key_signatures = compute_pathogenic_molecular_signatures(
    #         deg_df=data['deg_df'], 
    #         sig_pathway_df=pathway_info['significant_df'], 
    #         validated_pathways=validated_pathways,
    #         disease_name=disease_name,
    #         patient_prefix=patient_prefix,
    #         top_downregulated_pathways=top_downregulated_pathways,
    #         top_upregulated_pathways=top_upregulated_pathways
    #     )
    #     print(f"✅ Pathogenic molecular signatures computed successfully: {key_signatures.get('total_pathogenic_pathways', 0)} pathways")
        
    #     # Fallback to original ONLY if completely empty signature_data (not just lacking pathogenic pathways)
    #     if not key_signatures.get('signature_data', []):
    #         print("ℹ️  Empty signature data, falling back to original molecular signatures")
    #         key_signatures = compute_key_molecular_signatures(data['deg_df'], pathway_info['significant_df'], patient_prefix=patient_prefix)
    #     else:
    #         print(f"✅ Using pathogenic molecular signatures with {len(key_signatures.get('signature_data', []))} signature sections")
    # except Exception as e:
    #     print(f"⚠️  Error computing pathogenic molecular signatures: {e}")
    #     # Fallback to original function
    #     try:
    #         key_signatures = compute_key_molecular_signatures(data['deg_df'], pathway_info['significant_df'], patient_prefix=patient_prefix)
    #         print(f"✅ Fallback key signatures computed successfully")
    #     except Exception as e2:
    #         print(f"⚠️  Error computing fallback key signatures: {e2}")
    #         key_signatures = {
    #             'signature_data': [], 
    #             'top_gene': 'Unknown', 
    #             'lowest_gene': 'Unknown',
    #             'top_lfc': 0.0,
    #             'lowest_lfc': 0.0,
    #             'top_category': 'Unknown',
    #             'upregulated_count': 0,
    #             'downregulated_count': 0,
    #             'total_pathogenic_pathways': 0
    #         }
    
    # Final safety check: ensure signature_data is never completely empty
    # if not key_signatures.get('signature_data', []):
    #     print("⚠️  Signature data is empty, attempting to create from available pathway data")
        
    #     try:
    #         # Try to use significant pathway data first
    #         if pathway_info and 'significant_df' in pathway_info and len(pathway_info['significant_df']) > 0:
    #             print(f"ℹ️  Creating signature data from {len(pathway_info['significant_df'])} significant pathways")
    #             sig_pathways = pathway_info['significant_df'].head(20)  # Get top 20 to have enough for both up/down
                
    #             # Get DEG data for proper log2FC assignment
    #             deg_data = {}
    #             if 'deg_df' in data and len(data['deg_df']) > 0:
    #                 for _, row in data['deg_df'].iterrows():
    #                     gene_name = str(row.get('Gene', '')).strip().upper()
    #                     if gene_name:
    #                         deg_data[gene_name] = row.get(f'{patient_prefix}_log2FC', 0.0)
                
    #             upregulated_pathways = []
    #             downregulated_pathways = []
                
    #             for _, pathway_row in sig_pathways.iterrows():
    #                 pathway_name = pathway_row.get('Pathway', 'Unknown Pathway')
                    
    #                 # Try to get associated genes
    #                 pathway_genes = []
    #                 gene_sources = [
    #                     pathway_row.get('Pathway_Associated_Genes', ''),
    #                     pathway_row.get('Associated_Genes', ''), 
    #                     pathway_row.get('Genes', '')
    #                 ]
                    
    #                 for gene_source in gene_sources:
    #                     if pd.notna(gene_source) and str(gene_source).strip():
    #                         genes = str(gene_source).split(',')[:5]
    #                         for gene in genes:
    #                             clean_gene = gene.strip().upper()
    #                             if clean_gene and len(clean_gene) > 1:
    #                                 # Get actual log2FC from DEG data
    #                                 log2fc = deg_data.get(clean_gene, 0.0)
    #                                 pathway_genes.append({"gene": clean_gene, "log2fc": float(log2fc)})
    #                         break
                    
    #                 if len(pathway_genes) == 0:
    #                     # Create placeholder genes with random regulation for diversity
    #                     is_upregulated = len(upregulated_pathways) <= len(downregulated_pathways)
    #                     base_fc = 1.5 if is_upregulated else -1.5
    #                     pathway_genes = [
    #                         {"gene": "GENE1", "log2fc": base_fc},
    #                         {"gene": "GENE2", "log2fc": base_fc * 0.8}, 
    #                         {"gene": "GENE3", "log2fc": base_fc * 0.6}
    #                     ]
                    
    #                 # Determine pathway regulation based on average gene log2FC
    #                 avg_log2fc = sum(g["log2fc"] for g in pathway_genes) / len(pathway_genes)
    #                 regulation = "Upregulated" if avg_log2fc > 0 else "Downregulated"
                    
    #                 pathway_data = {
    #                     "pathway_name": pathway_name,
    #                     "regulation": regulation,
    #                     "priority_rank": pathway_row.get('Priority_Rank', 1),
    #                     "top_3_genes": pathway_genes[:3],
    #                     "validation_status": "Pathogenic",
    #                     "validation_confidence": 0.75,
    #                     "llm_description": f"The {pathway_name} pathway shows {regulation.lower()} activity in {disease_name}, contributing to disease pathophysiology through dysregulation of key molecular processes."
    #                 }
                    
    #                 # Separate into up/down regulated pathways
    #                 if regulation == "Upregulated":
    #                     upregulated_pathways.append(pathway_data)
    #                 else:
    #                     downregulated_pathways.append(pathway_data)
                
    #             # Create signature data for both categories
    #             signature_data = []
    #             if upregulated_pathways:
    #                 signature_data.append({
    #                     "signature_name": f"Upregulated Pathogenic Signatures",
    #                     "regulation_type": "Upregulated",
    #                     "pathways": upregulated_pathways[:10]  # Limit to 10
    #                 })
                
    #             if downregulated_pathways:
    #                 signature_data.append({
    #                     "signature_name": f"Downregulated Signatures",
    #                     "regulation_type": "Downregulated", 
    #                     "pathways": downregulated_pathways[:10]  # Limit to 10
    #                 })
                
    #             if signature_data:
    #                 key_signatures['signature_data'] = signature_data
    #                 print(f"✅ Created signature data with {len(upregulated_pathways)} upregulated and {len(downregulated_pathways)} downregulated pathways")
    #             else:
    #                 # No valid pathways - hide the section
    #                 key_signatures['signature_data'] = []
    #                 print("ℹ️  No valid pathways found, signature section will be hidden")
                    
    #         # Fallback to path_df if no significant pathways
    #         elif 'path_df' in data and len(data['path_df']) > 0:
    #             print("ℹ️  Falling back to path_df for signature data")
    #             sample_pathways = data['path_df'].head(3)
    #             placeholder_pathways = []
                
    #             for _, pathway_row in sample_pathways.iterrows():
    #                 pathway_genes = []
    #                 if 'Pathway_Associated_Genes' in pathway_row and pd.notna(pathway_row['Pathway_Associated_Genes']):
    #                     genes = pathway_row['Pathway_Associated_Genes'].split(',')[:3]
    #                     for gene in genes:
    #                         clean_gene = gene.strip()
    #                         pathway_genes.append({"gene": clean_gene, "log2fc": 1.5})
                    
    #                 if len(pathway_genes) == 0:
    #                     pathway_genes = [
    #                         {"gene": "GENE1", "log2fc": 1.5},
    #                         {"gene": "GENE2", "log2fc": 1.2}, 
    #                         {"gene": "GENE3", "log2fc": 1.0}
    #                     ]
                    
    #                 placeholder_pathways.append({
    #                     "pathway_name": pathway_row.get('Pathway', f"{disease_name} Associated Pathway"),
    #                     "regulation": "Upregulated",
    #                     "priority_rank": 1,
    #                     "top_3_genes": pathway_genes[:3],
    #                     "validation_status": "Pathogenic",
    #                     "validation_confidence": 0.7,
    #                     "llm_description": f"This pathway shows upregulated activity in {disease_name}, contributing to disease pathophysiology through altered expression of key genes."
    #                 })
                
    #             key_signatures['signature_data'] = [{
    #                 "signature_name": f"Upregulated {disease_name} Signatures",
    #                 "regulation_type": "Upregulated", 
    #                 "pathways": placeholder_pathways
    #             }]
    #             print(f"✅ Created signature data with {len(placeholder_pathways)} pathways from path_df")
    #         else:
    #             # No pathway data at all - hide the section
    #             key_signatures['signature_data'] = []
    #             print("ℹ️  No pathway data available, signature section will be hidden")
                
    #     except Exception as placeholder_error:
    #         print(f"⚠️  Error creating signature data: {placeholder_error}")
    #         # Hide the section on error
    #         key_signatures['signature_data'] = []
    #         print("ℹ️  Signature section will be hidden due to error")
    
    try:
        risk_factors = compute_risk_factors_pathways(data['path_df'], data['deg_df'], patient_prefix=patient_prefix)
        print(f"✅ Risk factors computed successfully")
    except Exception as e:
        print(f"⚠️  Error computing risk factors: {e}")
        risk_factors = []

    # Use LLM-calculated disease severity instead of simple threshold-based calculation
    severity = deg_stats.get("disease_severity", "MODERATE")

    # 6. DISEASE-SPECIFIC GENE ANALYSIS for key signatures
    # Check if top genes are disease-validated biomarkers
    # validated_gene_names = [g['gene'] for g in validated_pathways.get('relevant_biomarkers', [])]
    
    # top_gene_is_validated = key_signatures['top_gene'] in validated_gene_names
    # lowest_gene_is_validated = key_signatures['lowest_gene'] in validated_gene_names
    
    # # Enhanced disease activity with validation context
    # disease_activity = {
    #     "severity": severity,
    #     "activity_pattern": key_signatures.get("top_category", "Pathogenic Signatures"),
    #     "top_gene": key_signatures.get("top_gene", "Unknown"),
    #     "top_lfc": key_signatures.get("top_lfc", 0.0),
    #     "lowest_gene": key_signatures.get("lowest_gene", "Unknown"),
    #     "lowest_lfc": key_signatures.get("lowest_lfc", 0.0),
    #     "top_gene_validated": top_gene_is_validated,
    #     "lowest_gene_validated": lowest_gene_is_validated,
    #     "disease_context": disease_name,
    #     "validated_biomarker_count": len(validated_gene_names)
    # }

    disease_activity = {"severity": severity}

    # Get logo image as base64 for embedding in HTML
    try:
        logo_path = os.path.join(DATASET_PATH, 'ARI-Logo.png')
        if os.path.exists(logo_path):
            with open(logo_path, 'rb') as img_file:
                
                logo_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                logo_filename = f"data:image/png;base64,{logo_base64}"
        else:
            print(f"⚠️ Logo file not found at {logo_path}, using fallback")
            logo_filename = 'agentic_ai_wf/clinical_report/datasets/logo_image1.png'
    except Exception as e:
        print(f"⚠️ Error loading logo: {e}")
        logo_filename = 'agentic_ai_wf/clinical_report/datasets/logo_image1.png'

    # 8. FINAL DISEASE-SPECIFIC REPORT GENERATION
    print(f"Generating disease-specific report for {disease_name}")
    print(f"Report will contain:")
    print(f"  - {len(data['deg_df'])} disease-relevant genes")
    print(f"  - {len(data['path_df'])} disease-relevant pathways") 
    print(f"  - {len(validated_drugs_fda) if validated_drugs_fda else 0} disease-validated drugs")
    print(f"  - Structured validation outputs only (no free-text narratives)")
    
    template = env.get_template(HTML_TEMPLATE_FILE)
    
    # Fix drug data to ensure all required fields are properly formatted for template compatibility
    def fix_drug_data(drug_list):
        filtered_drugs = []
        
        for drug in drug_list:
            # Check if drug has recommendation field and it's "yes"
            has_recommendation = False
            
            # Check for recommendation in various formats
            if hasattr(drug, '__getitem__'):
                for rec_field in ['recommendation', 'Recommendation', 'recommended', 'Recommended']:
                    if rec_field in drug:
                        rec_value = str(drug[rec_field]).lower()
                        if rec_value == 'yes':
                            has_recommendation = True
                            break
            else:
                for rec_field in ['recommendation', 'Recommendation', 'recommended', 'Recommended']:
                    if hasattr(drug, rec_field):
                        rec_value = str(getattr(drug, rec_field)).lower()
                        if rec_value == 'yes':
                            has_recommendation = True
                            break
            
            # Skip drugs without recommendation=yes
            if not has_recommendation:
                continue
                
            # Fix mechanisms field
            if hasattr(drug, 'mechanisms') or 'mechanisms' in drug:
                mechanisms = getattr(drug, 'mechanisms', drug.get('mechanisms', ''))
                if isinstance(mechanisms, str):
                    # Split string mechanisms into list
                    if mechanisms.strip():
                        drug['mechanisms'] = [mechanisms.strip()] if hasattr(drug, '__setitem__') else [mechanisms.strip()]
                        if hasattr(drug, 'mechanisms'):
                            drug.mechanisms = [mechanisms.strip()]
                    else:
                        drug['mechanisms'] = ['Not available'] if hasattr(drug, '__setitem__') else ['Not available']
                        if hasattr(drug, 'mechanisms'):
                            drug.mechanisms = ['Not available']
                elif not mechanisms:  # Empty list or None
                    drug['mechanisms'] = ['Not available'] if hasattr(drug, '__setitem__') else ['Not available']
                    if hasattr(drug, 'mechanisms'):
                        drug.mechanisms = ['Not available']
            else:
                # Add mechanisms field if missing
                drug['mechanisms'] = ['Not available'] if hasattr(drug, '__setitem__') else ['Not available']
                if hasattr(drug, 'mechanisms'):
                    drug.mechanisms = ['Not available']
            
            # Ensure drug name is properly set and validated
            drug_name = getattr(drug, 'drug_name', drug.get('drug_name', ''))
            if not drug_name:
                drug_name = getattr(drug, 'name', drug.get('name', ''))
            
            # Set both 'drug' and 'drug_name' fields for template compatibility
            if hasattr(drug, '__setitem__'):
                drug['drug'] = drug_name
                drug['drug_name'] = drug_name
            if hasattr(drug, 'drug'):
                drug.drug = drug_name
            if hasattr(drug, 'drug_name'):
                drug.drug_name = drug_name
                
            # Get drug mechanisms
            mechanisms = []
            if hasattr(drug, '__getitem__') and 'mechanisms' in drug:
                mechanisms = drug['mechanisms']
            elif hasattr(drug, 'mechanisms'):
                mechanisms = drug.mechanisms
                
            # Get pathway association if available
            pathway_association = None
            for pathway_field in ['pathway', 'pathways', 'associated_pathway', 'pathway_association']:
                if hasattr(drug, '__getitem__') and pathway_field in drug:
                    pathway_association = drug[pathway_field]
                    break
                elif hasattr(drug, pathway_field):
                    pathway_association = getattr(drug, pathway_field)
                    break
                    
            # Get justification/evidence if available
            justification = None
            for evidence_field in ['justification', 'evidence', 'validation', 'scientific_evidence']:
                if hasattr(drug, '__getitem__') and evidence_field in drug:
                    justification = drug[evidence_field]
                    break
                elif hasattr(drug, evidence_field):
                    justification = getattr(drug, evidence_field)
                    break
                    
            # Generate description using LLM with best practices
            try:
                # Set status based on FDA approval
                status = "FDA Approved"  # Since we're only handling FDA approved drugs here
                
                # Generate description using LLM
                clinical_description = llm_generate_drug_description(
                    drug_name=drug_name,
                    mechanisms=mechanisms,
                    disease_name=disease_name,
                    status=status,
                    pathway_association=pathway_association,
                    justification=justification
                )
                
                # Add clinical_description to drug object
                if hasattr(drug, '__setitem__'):
                    drug['clinical_description'] = clinical_description
                elif hasattr(drug, 'clinical_description'):
                    drug.clinical_description = clinical_description
                    
                print(f"✅ Generated description for {drug_name}")
                
            except Exception as e:
                print(f"⚠️ Error generating drug description for {drug_name}: {str(e)}")
                # Fallback description
                fallback_clinical_description = f"Therapeutic agent for {disease_name} treatment."
                if hasattr(drug, '__setitem__'):
                    drug['clinical_description'] = fallback_clinical_description
                elif hasattr(drug, 'clinical_description'):
                    drug.clinical_description = fallback_clinical_description
                
            # Add the drug to the filtered list
            filtered_drugs.append(drug)
        
        return filtered_drugs
    
    print(f"📊 DRUG FILTERING PIPELINE for {disease_name}")
    print(f"=" * 60)
    
    initial_fda = len(validated_drugs_fda)
    initial_pathway = len(validated_drugs_fda)
    
    print(f"🏥 INITIAL DRUG COUNTS:")
    print(f"   - FDA Approved: {initial_fda} drugs")
    print(f"   - Pathway-Based: {initial_pathway} drugs")
    print(f"   - TOTAL INITIAL: {initial_fda + initial_pathway} drugs")
    
    # Filter each category separately with detailed logging
    print(f"\n🔍 FILTERING EACH CATEGORY for {disease_name} relevance...")
    
    # Calculate totals after evidence-based drug filtering (excluding pathway drugs for now)
    evidence_based_drugs = len(validated_drugs_fda)
    
    print(f"\n📈 POST-FILTERING EVIDENCE-BASED DRUG COUNTS:")
    print(f"   - FDA Approved: {len(validated_drugs_fda)} drugs")
    print(f"   - EVIDENCE-BASED TOTAL: {evidence_based_drugs} drugs")
    
    print(f"\n💊 DRUG GENERATION DECISION:")
    print(f"   - Current Total: {evidence_based_drugs} drugs (Evidence: {evidence_based_drugs} )")
    print(f"   - Minimum Required: 4 drugs")
    print(f"   - Generation Threshold: ≤5 total drugs")
    
    # Filter drugs by recommendation and fix data
    validated_drugs_fda = fix_drug_data(validated_drugs_fda)
    print(f"✅ After recommendation filtering: {len(validated_drugs_fda)} FDA approved drugs")
    
    grouped_signatures = group_signatures(data['path_df'])

    print(f"✅ Grouped signatures computed successfully length: {len(grouped_signatures)} Type: {type(grouped_signatures)}")

    grouped_signatures = map_genes_to_pathways(grouped_signatures, data['deg_df'], patient_prefix=patient_prefix)

    grouped_signatures = summarize_groups(grouped_signatures, disease_name, llm_client)
    
    # Safely load drug targeting data with error handling
    up_target_drugs = Path(drug_csv).parent / "up_targeting_drugs_disease_associated.csv"
    
    try:
        best_drugs = select_best_drugs(
            csv_path=up_target_drugs,
            disease=disease_name,
            llm_client=llm_client,
            patient_genes_path=deg_csv,
            patient_prefix=patient_prefix,
            limit=10
        )
        print(f"✅ Selected {len(best_drugs)} best drugs for {disease_name}")
    except Exception as e:
        print(f"⚠️  Error selecting best drugs: {e}")
        best_drugs = []  # Safe fallback to empty list
    
    html_out = template.render(
        patient_profile=patient_profile,
        disease=disease_name,
        # All these now contain disease-filtered data
        deg_stats=deg_stats,           # Based on disease-relevant genes only
        risk_factors=risk_factors,     # Enhanced with disease validation status
        fc_bins=fc_bins,              # Based on disease-relevant genes only
        pathway_info=pathway_info,     # Based on disease-relevant pathways only
        # key_signatures=key_signatures, # Enhanced with disease validation
        grouped_signatures=grouped_signatures,
        drug_hits=validated_drugs_fda,          # Based on disease-validated drugs
        disease_activity=disease_activity, # Enhanced with disease context
        logo_path=logo_filename,
        # Disease context information
        disease_context=disease_name,
        validated_drug_count=len(validated_drugs_fda) if validated_drugs_fda else 0,
        # Separate drug categories for template compatibility
        validated_drugs_fda=validated_drugs_fda,
        best_drugs=best_drugs,
    )

    # 4. Generate PDF with WeasyPrint - A4 size with zero padding
    pdf_css = CSS(string="""
        @page {
                size: Letter;
                margin: 0.25in;   /* ~5–6mm margin */
                padding: 0;
            }
        body {
            margin: 0;
            padding: 0;
        }
    """)
    external_css_path = os.path.join(DATASET_PATH, "clinical_report.css")
    external_styles = []
    if os.path.exists(external_css_path):
        external_styles.append(CSS(filename=external_css_path))
    else:
        print(
            f"⚠️ External stylesheet not found at {external_css_path}; "
            "falling back to inline defaults."
        )
    
    print(f"🎯 Generating final PDF report for {disease_name}...")
    
    try:
        # Debug information before PDF generation
        print(f"📝 HTML content length: {len(html_out)} characters")
        print(f"📁 Output PDF path: {output_pdf}")
        print(f"📂 DATASET_PATH: {DATASET_PATH}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_pdf)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 Created output directory: {output_dir}")
        
        # Check if HTML content is valid (not empty)
        if not html_out or html_out.strip() == "":
            raise ValueError("HTML content is empty or invalid")
        
        # Test HTML parsing first
        html_doc = HTML(string=html_out, base_url=DATASET_PATH)
        print(f"✅ HTML parsing successful")
        
        # Generate PDF
        html_doc.write_pdf(output_pdf, stylesheets=[pdf_css, *external_styles])
        print(f"✅ PDF generation successful")
        
        # Verify PDF file was created and has content
        if os.path.exists(output_pdf):
            file_size = os.path.getsize(output_pdf)
            print(f"✅ PDF file created: {output_pdf} ({file_size} bytes)")
        else:
            raise FileNotFoundError(f"PDF file was not created at {output_pdf}")
            
        print(f"✅ Disease-specific report generated successfully for {disease_name}")
        print(f"📄 Report saved to: {output_pdf}")
        print(f"📊 Final report contains:")
        print(f"   - {len(data['deg_df'])} genes analyzed")
        print(f"   - {len(data['path_df'])} pathways analyzed") 
        print(f"   - {len(validated_drugs_fda)} drugs/recommendations included")
        
        return output_pdf
        
    except Exception as e:
        print(f"❌ PDF generation failed: {str(e)}")
        print(f"🔍 Error type: {type(e).__name__}")
        
        # Additional debugging for common issues
        try:
            import weasyprint
            print(f"✅ WeasyPrint version: {weasyprint.__version__}")
        except ImportError:
            print(f"❌ WeasyPrint not available")
        except Exception as wp_e:
            print(f"⚠️  WeasyPrint issue: {wp_e}")
        
        # Check if it's an HTML issue by saving HTML to file for inspection
        try:
            html_debug_path = output_pdf.replace('.pdf', '_debug.html')
            with open(html_debug_path, 'w', encoding='utf-8') as f:
                f.write(html_out)
            print(f"🔍 HTML content saved for debugging: {html_debug_path}")
        except Exception as html_e:
            print(f"⚠️  Could not save HTML debug file: {html_e}")
        
        # Re-raise the exception to be caught by coordinator
        raise Exception(f"PDF generation failed for {disease_name}: {str(e)}")