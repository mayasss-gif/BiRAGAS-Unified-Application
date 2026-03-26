#------------------------------------------------------------------------
# Risk Factors Pathways
#------------------------------------------------------------------------
import pandas as pd
from typing import Dict
from .utils import _infer_columns_deg
from .text_generation import llm_generate_pathway_consequence

def compute_risk_factors_pathways(path_df: pd.DataFrame, deg_df: pd.DataFrame, patient_prefix: str = "patient", disease_name: str = None) -> Dict:
    """
    Compute risk factor pathways and generate consequences for reporting.
    
    Args:
        path_df: Pathway dataframe
        deg_df: Differential expression gene dataframe
        patient_prefix: Prefix for patient columns
        disease_name: Name of the disease for pathway consequence generation
        
    Returns:
        Dictionary with pathway information including consequences
    """

    
    # ENHANCED: Include all databases (no filtering by DB_ID - accept all pathway sources)
    # Only filter by confidence if needed
    if "Confidence_Level" in path_df.columns:
        filtered_df = path_df[
            (path_df["Confidence_Level"] == "High") | 
            (path_df["Confidence_Level"] == "Medium")
        ]
        if len(filtered_df) < 10:
            filtered_df = path_df
        path_df = filtered_df
    # Prepare DEG gene log2fc mapping
    gene_col, fc_col, p_value_col = _infer_columns_deg(deg_df, patient_prefix)
    deg_df = deg_df.rename(columns={gene_col: "Gene", fc_col: "log2FC"})
    # Clean gene names for matching
    deg_df["Gene"] = deg_df["Gene"].astype(str).str.strip().str.split(".").str[0].str.upper()
    log2fc_map = deg_df.set_index("Gene")["log2FC"].astype(float).to_dict()

    # ENHANCED: Prefer upregulated pathways, but include high-priority downregulated if needed
    print(f"🔍 RISK FACTORS PATHWAY FILTERING (ENHANCED):")
    print(f"   → Total pathways available: {len(path_df)}")
    
    upregulated_pathways = []
    downregulated_pathways = []
    
    for _, row in path_df.iterrows():
        pathway_name = row.get("Pathway_Name", "Unknown Pathway")
        # Use multiple possible column names for pathway name
        if pathway_name == "Unknown Pathway":
            pathway_name = row.get("Pathway", row.get("pathway_name", "Unknown Pathway"))
        
        # Split and clean associated genes to calculate regulation
        genes = []
        gene_sources = [
            row.get("Pathway_Associated_Genes"),
            row.get("Associated_Genes"), 
            row.get("Genes")
        ]
        
        for gene_source in gene_sources:
            if pd.notna(gene_source):
                genes = [g.strip().split(".")[0].upper() for g in str(gene_source).split(",")]
                break
        
        # Calculate average log2FC for pathway to determine regulation
        total_fc = 0
        valid_gene_count = 0
        
        for gene in genes:
            log2fc = log2fc_map.get(gene)
            if log2fc is not None:
                total_fc += log2fc
                valid_gene_count += 1
        
        # Classify pathways by regulation with priority ranking
        if valid_gene_count > 0:
            avg_fc = total_fc / valid_gene_count
            priority_rank = row.get("Priority_Rank", float('inf'))
            
            if avg_fc > 0:  # UPREGULATED - PREFERRED
                upregulated_pathways.append((row, avg_fc, pathway_name, priority_rank))
                print(f"   ✅ Upregulated pathway: {pathway_name} (avg_log2FC: {avg_fc:.3f}, rank: {priority_rank})")
            else:  # DOWNREGULATED - BACKUP OPTION
                downregulated_pathways.append((row, avg_fc, pathway_name, priority_rank))
                print(f"   🔶 Downregulated pathway: {pathway_name} (avg_log2FC: {avg_fc:.3f}, rank: {priority_rank})")
        else:
            print(f"   ⚠️  Pathway with no valid genes excluded: {pathway_name}")
    
    # Sort both lists by Priority_Rank (lower rank = higher priority)
    upregulated_pathways.sort(key=lambda x: x[3])  # Sort by priority_rank
    downregulated_pathways.sort(key=lambda x: x[3])  # Sort by priority_rank
    
    # ENHANCED SELECTION LOGIC: Prefer upregulated, supplement with downregulated if needed
    selected_pathways = []
    
    # First, take all available upregulated pathways (up to 4)
    selected_pathways.extend(upregulated_pathways[:4])
    print(f"   → Selected {len(selected_pathways)} upregulated pathways")
    
    # If we need more pathways, supplement with highest-priority downregulated
    if len(selected_pathways) < 4:
        needed = 4 - len(selected_pathways)
        additional_downregulated = downregulated_pathways[:needed]
        selected_pathways.extend(additional_downregulated)
        print(f"   → Added {len(additional_downregulated)} high-priority downregulated pathways")
        print(f"   → Total pathways for risk factors: {len(selected_pathways)}")
        
        # Log the scientific rationale
        if additional_downregulated:
            print(f"   📋 SCIENTIFIC RATIONALE: Including high-priority downregulated pathways")
            print(f"       • These represent loss-of-function mechanisms in disease pathophysiology")
            print(f"       • Consequence descriptions will be scientifically accurate for each direction")
    else:
        print(f"   → Sufficient upregulated pathways found: {len(selected_pathways)}")
    
    top_4_pathways = selected_pathways[:4]  # Ensure exactly 4

    pathway_gene_log2fc = []
    used_genes = set()  # Track genes already used as primary genes to encourage diversity
    
    for pathway_tuple in top_4_pathways:
        row, avg_fc, pathway_name, priority_rank = pathway_tuple  # Enhanced tuple with priority_rank
        
        # Split and clean associated genes
        genes = []
        gene_sources = [
            row.get("Pathway_Associated_Genes"),
            row.get("Associated_Genes"), 
            row.get("Genes")
        ]
        
        for gene_source in gene_sources:
            if pd.notna(gene_source):
                genes = [g.strip().split(".")[0].upper() for g in str(gene_source).split(",")]
                break
        
        gene_log2fc_list = []
        for gene in genes:
            log2fc = log2fc_map.get(gene)
            if log2fc is not None:  # Only include genes with actual log2fc values
                gene_log2fc_list.append({
                    "gene": gene,
                    "log2fc": log2fc
                })
        
        # Sort the genes by absolute log2fc, highest first
        gene_log2fc_list_sorted = sorted(
            gene_log2fc_list,
            key=lambda x: abs(x["log2fc"]),
            reverse=True
        )
        
        # Try to find a unique top gene that hasn't been used yet
        selected_gene = None
        for gene_info in gene_log2fc_list_sorted:
            if gene_info["gene"] not in used_genes:
                selected_gene = gene_info
                used_genes.add(gene_info["gene"])
                break
        
        # If all genes are already used, take the top gene anyway
        if not selected_gene and gene_log2fc_list_sorted:
            selected_gene = gene_log2fc_list_sorted[0]
            used_genes.add(selected_gene["gene"])
        
        # Ensure there's always at least one gene entry for the template
        if not selected_gene:
            selected_gene = {
                "gene": "No genes available",
                "log2fc": 0.0
            }
        
        # Put the selected gene first, followed by other genes
        final_gene_list = [selected_gene]
        for gene_info in gene_log2fc_list_sorted:
            if gene_info["gene"] != selected_gene["gene"]:
                final_gene_list.append(gene_info)
        
        # Determine regulation status based on average log2FC
        regulation_status = "Upregulated" if avg_fc > 0 else "Downregulated"
        
        # Generate pathway consequence for reporting
        validation_status = "Pathogenic"  # Assume pathogenic for risk factors
        validation_justification = f"Average log2FC: {avg_fc:.3f}, Priority Rank: {priority_rank}"
        
        # Use disease name if provided, otherwise use a generic term
        disease_context = disease_name if disease_name else "this condition"
        
        # Generate pathway consequence using LLM
        pathway_consequence = llm_generate_pathway_consequence(
            disease_name=disease_context,
            pathway_name=pathway_name,
            regulation_status=regulation_status,
            validation_status=validation_status,
            validation_justification=validation_justification
        )
        
        pathway_gene_log2fc.append({
            "pathway_name": pathway_name,
            "genes": final_gene_list,
            "regulation_status": regulation_status,
            "consequence": pathway_consequence
        })
    
    # Ensure we always have at least 4 pathways for the template (which expects risk_factors[0] through risk_factors[3])
    # ENHANCED: If we don't have enough upregulated pathways, fill with placeholders that indicate this is due to regulation filtering
    while len(pathway_gene_log2fc) < 4:
        placeholder_index = len(pathway_gene_log2fc) + 1
        pathway_gene_log2fc.append({
            "pathway_name": "No pathway data available",
            "genes": [{
                "gene": "Only upregulated pathways included in risk factors",
                "log2fc": 0.0
            }],
            "regulation_status": "Unknown",
            "consequence": "No pathogenic pathway identified."
        })
        print(f"   ⚠️  Added placeholder entry {placeholder_index} due to insufficient upregulated pathways")
    
    return pathway_gene_log2fc