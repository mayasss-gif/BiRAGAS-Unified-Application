import pandas as pd
from typing import Dict
from .utils import _infer_columns_deg

def compute_key_molecular_signatures(deg_df: pd.DataFrame, sig_pathway_df: pd.DataFrame, patient_prefix: str = "patient") -> Dict:
    """Compute key molecular signatures from DEG and pathway data."""
    # Get significant DEGs and pathways
    gene_col, fc_col, p_value_col = _infer_columns_deg(deg_df, patient_prefix)
    deg_df = deg_df.rename(columns={gene_col: "Gene", fc_col: "log2FC", p_value_col: "p_value"})
    # deg_df = deg_df[(deg_df.adj_p < 0.05) & (deg_df.log2FC.abs() >= 1)]
    pathway_df = sig_pathway_df[sig_pathway_df.FDR < 0.05]
    # ENHANCED: Include all databases (no DB_ID restrictions)
    filtered_df = pathway_df[((pathway_df["Confidence_Level"] == "High") | (pathway_df["Confidence_Level"] == "Medium"))]
    if len(filtered_df) < 10:
        filtered_df = pathway_df
    pathway_df = filtered_df
    
    # print(pathway_df["Sub_Class"].value_counts())
    categories = pathway_df["Main_Class"].unique()
    clean_deg = (
                    deg_df.assign(Gene=lambda d:
                        d["Gene"].astype(str)
                                .str.strip()             
                                .str.split(".").str[0]   
                                .str.upper()             
                    )
                    .drop_duplicates(subset="Gene", keep="first")
                    .set_index("Gene")["log2FC"]          
                    .astype(float)
                )
    log2fc_map = clean_deg.to_dict()          

    main_class_counts = pathway_df["Main_Class"].value_counts()
    top_category = main_class_counts.index[0] if len(main_class_counts) > 0 else None
    
    signatures = {}
    
    for category in categories:
        genes_stats = []
        cat_genes = (
            pathway_df.loc[pathway_df["Main_Class"] == category, "Pathway_Associated_Genes"]
                    .str.split(",")
                    .explode()
                    .str.strip()          
                    .str.split(".").str[0]
                    .str.upper()
                    .dropna()
                    .unique()
        )

        for gene in cat_genes:
            try:
                if abs(float(log2fc_map[gene])) < 1:
                    continue
                genes_stats.append({"gene": gene,
                                    "log2fc": float(log2fc_map[gene])})
            except KeyError:
                # keep going but record the miss; you can inspect later
                genes_stats.append({"gene": gene, "log2fc": None})
                print(f"WARNING: {gene!r} not found in deg_df after cleaning")
                

        signatures[category] = genes_stats

    # Get the gene with highest log2FC from top category
    # top_gene = None
    # top_lfc = -float('inf')
    
    # if top_category and top_category in signatures:
    #     for gene_info in signatures[top_category]:
    #         if gene_info["log2fc"] is not None and gene_info["log2fc"] > top_lfc:
    #             top_lfc = gene_info["log2fc"]
    #             top_gene = gene_info["gene"]
    # Handle empty DEG DataFrame case
    if len(deg_df) == 0:
        top_gene = "No genes available"
        top_lfc = 0.0
        lowest_gene = "No genes available"
        lowest_lfc = 0.0
    else:
        top_genes = deg_df.nlargest(1, "log2FC")
        if len(top_genes) > 0:
            top_gene = top_genes["Gene"].values[0]
            top_lfc = top_genes["log2FC"].values[0]
        else:
            top_gene = "No genes available"
            top_lfc = 0.0
            
        lowest_genes = deg_df.nsmallest(1, "log2FC")
        if len(lowest_genes) > 0:
            lowest_gene = lowest_genes["Gene"].values[0]
            lowest_lfc = lowest_genes["log2FC"].values[0]
        else:
            lowest_gene = "No genes available"
            lowest_lfc = 0.0

    # get gene with lowest log2fc
    # lowest_gene = None
    # lowest_lfc = float('inf')
    # for gene_info in signatures[top_category]:
    #     if gene_info["log2fc"] is not None and gene_info["log2fc"] < lowest_lfc:
    #         lowest_lfc = gene_info["log2fc"]
    #         lowest_gene = gene_info["gene"]



    # get the top 3 key signatures and the top 8 genes and their log2fc
    top_3_signatures = list(main_class_counts.index[:3])
    
    # Create list with top 3 signatures and their top 8 genes with log2fc
    signature_data = []
    used_pathways = set()  # Track used pathways to avoid duplicates
    
    for signature in top_3_signatures:
        if signature in signatures:
            genes_with_fc = signatures[signature]
            # Filter out None values
            valid_genes = [g for g in genes_with_fc if g["log2fc"] is not None]
            # Get top 4 with highest positive log2fc
            top_4_pos = sorted(
                [g for g in valid_genes if g["log2fc"] > 0],
                key=lambda x: x["log2fc"],
                reverse=True
            )[:4]
            # Get top 4 with lowest (most negative) log2fc
            top_4_neg = sorted(
                [g for g in valid_genes if g["log2fc"] < 0],
                key=lambda x: x["log2fc"]
            )[:4]
            # Combine for top 8 (4 positive, 4 negative)
            top_8_genes = top_4_pos + top_4_neg

            # Get top 2 pathways for this signature based on lowest Priority_Rank
            # signature_pathways = pathway_df[pathway_df["Main_Class"] == signature]
            # top_2_pathways = signature_pathways.nsmallest(2, "Priority_Rank")
            
            # Get pathways with lowest Priority_Rank that haven't been used yet
            available_pathways = pathway_df[~pathway_df["Pathway"].isin(used_pathways)]
            lowest_priority_pathway = available_pathways.nsmallest(2, "Priority_Rank")
            
            pathways_data = []
            for _, pathway_row in lowest_priority_pathway.iterrows():
                pathway_name = pathway_row["Pathway"]
                # Add to used pathways set
                used_pathways.add(pathway_name)
                
                pathway_genes = (
                    pathway_row["Pathway_Associated_Genes"]
                    .split(",") if pd.notna(pathway_row["Pathway_Associated_Genes"]) else []
                )
                
                # Clean and get log2fc for pathway genes
                pathway_genes_with_fc = []
                for gene in pathway_genes:
                    clean_gene = gene.strip().split(".")[0].upper()
                    if clean_gene in log2fc_map:
                        log2fc_val = log2fc_map[clean_gene]
                        if abs(log2fc_val) >= 1:  # Only include significant genes
                            pathway_genes_with_fc.append({
                                "gene": clean_gene,
                                "log2fc": log2fc_val
                            })
                
                # Get top 3 genes (sorted by absolute log2fc)
                top_3_pathway_genes = sorted(
                    pathway_genes_with_fc,
                    key=lambda x: abs(x["log2fc"]),
                    reverse=True
                )[:3]

                pathways_data.append({
                    "pathway_name": pathway_name,
                    "priority_rank": pathway_row["Priority_Rank"],
                    "top_3_genes": top_3_pathway_genes,
                })

            # Convert to format expected by template
            if pathways_data:  # Only add if we have pathways
                # Determine overall regulation type for this signature
                avg_fc = sum(sum(g["log2fc"] for g in p["top_3_genes"]) / len(p["top_3_genes"]) for p in pathways_data) / len(pathways_data) if pathways_data else 0
                regulation_type = "Upregulated" if avg_fc > 0 else "Downregulated"
                
                # Convert pathways to expected format with enhanced pathogenic classification
                formatted_pathways = []
                for pathway in pathways_data:
                    pathway_name_lower = pathway["pathway_name"].lower()
                    
                    # Use AI-only approach (no static keyword rules)
                    if regulation_type == "Upregulated":
                        # Default to non-pathogenic unless AI validation determines otherwise
                        is_pathogenic = False
                        validation_status = "Non-Pathogenic"
                        confidence = 0.7  # Default confidence
                        llm_description = f"The {pathway['pathway_name']} pathway shows upregulated activity in this analysis, representing a disease-associated alteration."
                    
                    else:  # Downregulated
                        # Use AI-only approach (no static keyword rules)
                        is_pathogenic = False  # Default to non-pathogenic unless AI validation determines otherwise
                        validation_status = "Non-Pathogenic"
                        confidence = 0.7  # Default confidence
                        llm_description = f"Downregulated {pathway['pathway_name']} activity may represent compensatory changes rather than direct pathogenic mechanisms."
                    
                    formatted_pathways.append({
                        "pathway_name": pathway["pathway_name"],
                        "regulation": regulation_type,
                        "priority_rank": pathway["priority_rank"],
                        "top_3_genes": pathway["top_3_genes"],
                        "validation_status": validation_status,
                        "validation_confidence": confidence,
                        "is_pathogenic": is_pathogenic,
                        "llm_description": llm_description
                    })
                
                # UPDATED: Remove category-based naming to avoid multiple sections
                # Instead use simplified naming without Main_Class categories
                if regulation_type == "Upregulated":
                    signature_name = f"Top Upregulated Signatures ({len(formatted_pathways)})"
                else:
                    signature_name = f"Top Downregulated Signatures ({len(formatted_pathways)})"
                
                signature_data.append({
                    "signature_name": signature_name,
                    "regulation_type": regulation_type,
                    "pathways": formatted_pathways
                })

    return {
        "top_category": top_category,
        "top_gene": top_gene,
        "top_lfc": top_lfc,
        "lowest_gene": lowest_gene,
        "lowest_lfc": lowest_lfc,
        "signatures": signatures,
        "signature_data": signature_data,
    }