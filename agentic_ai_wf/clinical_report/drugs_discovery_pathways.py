import pandas as pd
from typing import List, Dict, Tuple
import os
from openai import OpenAI
from .utils import _infer_columns_deg

def llm_drug_discovery_for_all_pathways(drug_df: pd.DataFrame, sig_path_df: pd.DataFrame, deg_df: pd.DataFrame = None, patient_prefix: str = "patient", disease_name: str = None, drug_csv_path: str = None) -> pd.DataFrame:
    """
    Find drugs for all pathways in sig_path_df using LLM when drug_df has fewer than 8 unique pathways.
    Appends new drugs to the existing drug_df and saves to the same file path.
    
    Args:
        drug_df: Existing drug dataframe
        sig_path_df: Significant pathways dataframe
        deg_df: DEG dataframe for gene expression data
        patient_prefix: Prefix for patient columns
        disease_name: Disease context for LLM
        drug_csv_path: Path to save the updated drug CSV file
    
    Returns:
        Updated drug dataframe with new LLM-generated drugs
    """
    if drug_df is None:
        print("No existing drug dataframe provided. Initializing empty dataframe for LLM drug discovery.")
        drug_df = pd.DataFrame()
    
    # Count unique pathways in existing drug_df
    unique_pathways = drug_df['pathway_name'].nunique() if 'pathway_name' in drug_df.columns else 0
    print(f"Current drug_df contains {unique_pathways} unique pathways")
    
    if unique_pathways >= 8:
        print("Drug dataframe already has 8 or more unique pathways. No LLM drug discovery needed.")
        return None
    
    print(f"Drug dataframe has fewer than 8 unique pathways ({unique_pathways}). Activating LLM drug discovery for all pathways...")
    print(f"Available columns in sig_path_df: {list(sig_path_df.columns)}")
    
    # Prepare DEG gene log2fc mapping if deg_df is provided
    log2fc_map = {}
    if deg_df is not None:
        gene_col, fc_col, p_value_col = _infer_columns_deg(deg_df, patient_prefix)
        deg_df_clean = deg_df.rename(columns={gene_col: "Gene", fc_col: "log2FC"})
        # Clean gene names for matching
        deg_df_clean["Gene"] = deg_df_clean["Gene"].astype(str).str.strip().str.split(".").str[0].str.upper()
        log2fc_map = deg_df_clean.set_index("Gene")["log2FC"].astype(float).to_dict()
    
    # Get all pathways from sig_path_df (including KEGG pathways for comprehensive drug discovery)
    all_pathways_df = sig_path_df.copy()
    print(f"Total pathways in sig_path_df: {len(all_pathways_df)}")
    
    # Don't exclude KEGG pathways - they can still be useful for drug discovery
    # if "DB_ID" in all_pathways_df.columns:
    #     all_pathways_df = all_pathways_df[all_pathways_df["DB_ID"] != "KEGG"]
    
    # Prepare pathway information for LLM
    pathway_info_for_llm = []
    
    for _, prow in all_pathways_df.iterrows():
        # Handle both "Pathway" and "Pathway_Name" column names
        pathway_col = "Pathway" if "Pathway" in prow.index else "Pathway_Name"
        pathway_name = prow.get(pathway_col, "Unknown Pathway")
        pathway_id = prow.get("Pathway_ID", f"pathway_{len(pathway_info_for_llm) + 1}")
        
        # Get associated genes and their log2fc for this pathway
        pathway_genes_with_fc = []
        if deg_df is not None and pd.notna(prow.get("Pathway_Associated_Genes", None)):
            genes = [g.strip().split(".")[0].upper() for g in prow["Pathway_Associated_Genes"].split(",")]
            for gene in genes:
                log2fc = log2fc_map.get(gene)
                pathway_genes_with_fc.append({
                    "gene": gene,
                    "log2fc": log2fc
                })
            # Sort genes by absolute log2fc (highest first, None values last)
            pathway_genes_with_fc = sorted(
                pathway_genes_with_fc,
                key=lambda x: (abs(x["log2fc"]) if x["log2fc"] is not None else -float("inf")),
                reverse=True
            )
        
        pathway_info_for_llm.append({
            "pathway_id": pathway_id,
            "pathway_name": pathway_name,
            "genes": pathway_genes_with_fc,
            "priority_rank": prow.get("Priority_Rank", float('inf')),
            "llm_score": prow.get("LLM_Score", 0.0),
            "score_justification": prow.get("Score_Justification", ""),
            "confidence_level": prow.get("Confidence_Level", 0.0),
            "pathway_regulation": prow.get("Reg", ""),
            "pathway_associated_genes_raw": prow.get("Pathway_Associated_Genes", ""),
            "regulation": prow.get("Reg", "Unknown")  # Add regulation field for LLM fallback compatibility
        })
        
        if len(pathway_info_for_llm) <= 3:  # Only print first few for debugging
            print(f"Added pathway: {pathway_name} with {len(pathway_genes_with_fc)} genes")
    
    # Sort pathways by priority rank
    pathway_info_for_llm = sorted(pathway_info_for_llm, key=lambda x: x.get("priority_rank", float('inf')))
    print(f"Sorted {len(pathway_info_for_llm)} pathways by priority rank")
    
    print(f"Prepared {len(pathway_info_for_llm)} pathways for LLM drug discovery")
    print(f"Generating LLM drug recommendations for {len(pathway_info_for_llm)} pathways...")
    
    # Use LLM to generate drug recommendations for all pathways
    try:
        print(f"Calling LLM drug discovery with {len(pathway_info_for_llm)} pathways")
        llm_drug_results = llm_drug_discovery_fallback(pathway_info_for_llm, disease_context=disease_name)
        print(f"LLM generated {len(llm_drug_results)} drug recommendations.")
        
        # Convert LLM results to DataFrame format
        new_drugs_data = []
        llm_counter = 1
        
        for result in llm_drug_results:
            pathway_name = result.get("pathway", "Unknown Pathway")
            
            # Find corresponding pathway info for additional columns
            pathway_info = next((p for p in pathway_info_for_llm if p["pathway_name"] == pathway_name), None)
            
            # Get pathway associated genes if available
            pathway_associated_genes = ""
            if pathway_info and pathway_info.get("genes"):
                pathway_associated_genes = ", ".join([g["gene"] for g in pathway_info["genes"]])
            elif pathway_info and pathway_info.get("pathway_associated_genes_raw"):
                pathway_associated_genes = pathway_info["pathway_associated_genes_raw"]
            
            drug_name = result.get("drug", "")
            
            # CRITICAL: Only create drug row if we have a valid drug name
            if not validate_drug_name(drug_name):
                print(f"❌ Skipping LLM result - invalid drug name: '{drug_name}'")
                continue
            
            new_drug_row = {
                'final_rank': "",  # Will be calculated later
                'pathway_id': pathway_info.get("pathway_id", f"pathway_{llm_counter}") if pathway_info else f"pathway_{llm_counter}",
                'drug_id': f"llm{llm_counter:02d}",  # llm01, llm02, llm03, etc.
                'drug_name': drug_name,
                'priority_score': pathway_info.get("llm_score", 0.0) if pathway_info else 0.0,
                'confidence': pathway_info.get("confidence_level", 0.0) if pathway_info else 0.0,  # Map from Confidence_Level
                'justification': result.get("evidence_summary", "Evidence not available"),  # Map evidence_summary to justification
                'pathway_name': pathway_name,
                'pathway_associated_genes': pathway_associated_genes,
                'target_genes': "",  # Will be filled if available from LLM response
                'gene_overlap': "",
                'patient_log2fc': "",
                'llm_score': pathway_info.get("llm_score", 0.0) if pathway_info else 0.0,
                'score_justification': pathway_info.get("score_justification", "") if pathway_info else "",
                'target_mechanism': result.get("mechanism", "Mechanism not specified"),
                'mechanism_of_action': result.get("mechanism", "Mechanism not specified"),  # Duplicate for compatibility
                'fda_approved_status': result.get("approved", "Unknown"),
                'route_of_administration': "",  # Not provided by LLM
                'molecular_evidence_score': 0.0,  # Default for LLM-generated drugs
                'pathway_regulation': pathway_info.get("pathway_regulation", "") if pathway_info else ""  # Map from Reg
            }
            
            new_drugs_data.append(new_drug_row)
            llm_counter += 1
        
        # Create DataFrame for new drugs
        new_drugs_df = pd.DataFrame(new_drugs_data)
        
        # Combine with existing drug_df
        # Ensure column names match between existing and new dataframes
        required_columns = [
            'final_rank', 'pathway_id', 'drug_id', 'drug_name', 'priority_score', 
            'confidence', 'justification', 'pathway_name', 'pathway_associated_genes', 
            'target_genes', 'gene_overlap', 'patient_log2fc', 'llm_score', 
            'score_justification', 'target_mechanism', 'mechanism_of_action', 
            'fda_approved_status', 'route_of_administration', 'molecular_evidence_score', 
            'pathway_regulation'
        ]

        # Backward-compat mapping on existing df, if present
        column_mapping = {
            'evidence_summary': 'justification',
            'target_mechanism': 'mechanism_of_action'
        }
        for old_col, new_col in column_mapping.items():
            if old_col in drug_df.columns and new_col not in drug_df.columns:
                drug_df[new_col] = drug_df[old_col]
            elif old_col in drug_df.columns and new_col in drug_df.columns:
                if new_col == 'mechanism_of_action' and getattr(drug_df[new_col], 'isna', lambda: False)().all():
                    drug_df[new_col] = drug_df[old_col]

        # Ensure required columns exist on both frames
        for col in required_columns:
            if col not in drug_df.columns:
                drug_df[col] = ""
            if col not in new_drugs_df.columns:
                new_drugs_df[col] = ""
        
        # Standardize column names and merge duplicates to avoid repeated columns
        def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
            try:
                df2 = df.copy()
                # Canonical rename map (case-insensitive synonyms → canonical snake_case)
                rename_map = {
                    'Drug_Name': 'drug_name',
                    'Drug name': 'drug_name',
                    'DRUG_NAME': 'drug_name',
                    'Status': 'status',
                    'Confidence': 'confidence',
                    'Clinical_Recommendation': 'clinical_recommendation',
                    'Evidence': 'justification',
                    'Mechanism': 'mechanism_of_action',
                    'Disease_Relevance': 'disease_relevance',
                    'Pathway_Association': 'pathway_association',
                    'Priority_Score': 'priority_score',
                    'Final_Rank': 'final_rank',
                    'Pathway_ID': 'pathway_id',
                    'Pathway_Name': 'pathway_name',
                    'Target_Genes': 'target_genes',
                    'Gene_Overlap': 'gene_overlap',
                    'Patient_Log2FC': 'patient_log2fc',
                    'LLM_Score': 'llm_score',
                    'Score_Justification': 'score_justification',
                    'Target_Mechanism': 'target_mechanism',
                    'Mechanism_of_Action': 'mechanism_of_action',
                    'FDA_Approved_Status': 'fda_approved_status',
                    'Route_of_Administration': 'route_of_administration',
                    'Molecular_Evidence_Score': 'molecular_evidence_score',
                    'Pathway_Regulation': 'pathway_regulation',
                }
                # Apply direct rename where exact key exists
                df2 = df2.rename(columns={k: v for k, v in rename_map.items() if k in df2.columns})

                # After renaming, merge duplicate columns by name
                col_counts = pd.Series(df2.columns).value_counts()
                duplicated_names = [c for c, n in col_counts.items() if n > 1]
                for name in duplicated_names:
                    dup_cols = [c for c in df2.columns if c == name]
                    base = df2[dup_cols[0]].copy()
                    for c in dup_cols[1:]:
                        base = base.combine_first(df2[c])
                    df2[dup_cols[0]] = base
                    df2 = df2.drop(columns=dup_cols[1:])
                return df2
            except Exception:
                return df

        drug_df_std = _std_cols(drug_df)
        new_drugs_df = _std_cols(new_drugs_df)

        # Combine dataframes (with standardized columns)
        combined_drug_df = pd.concat([drug_df_std, new_drugs_df], ignore_index=True)
        
        # Calculate final ranking for all drugs
        try:
            # Convert priority_score to numeric for ranking
            combined_drug_df['priority_score'] = pd.to_numeric(combined_drug_df['priority_score'], errors='coerce').fillna(0.0)
            
            # Sort by priority_score in descending order (higher score = better rank)
            combined_drug_df = combined_drug_df.sort_values('priority_score', ascending=False)
            
            # Assign final ranks (1-based ranking)
            combined_drug_df['final_rank'] = range(1, len(combined_drug_df) + 1)

            # Final safeguard: standardize again post-calculation
            combined_drug_df = _std_cols(combined_drug_df)
            
            print(f"Assigned final ranks to {len(combined_drug_df)} drugs based on priority scores")
        except Exception as e:
            print(f"Warning: Could not calculate final ranks: {str(e)}")
            # Assign sequential ranks as fallback
            combined_drug_df['final_rank'] = range(1, len(combined_drug_df) + 1)
        
        # Save to the same file path if provided
        if drug_csv_path:
            try:
                combined_drug_df.to_csv(drug_csv_path, index=False)
                print(f"Updated drug dataframe saved to: {drug_csv_path}")
            except Exception as e:
                print(f"Warning: Could not save updated drug dataframe to {drug_csv_path}: {str(e)}")
        
        print(f"Successfully added {len(new_drugs_data)} new LLM-generated drugs to the drug dataframe.")
        return combined_drug_df
        
    except Exception as e:
        print(f"LLM drug discovery failed: {str(e)}")
        return drug_df

# ----------------------------------------------------------------------
# DRUG ↔︎ PATHWAY MATCHING                                              
# ----------------------------------------------------------------------

def llm_drug_discovery_fallback(top_pathways: List[Dict], disease_context: str = None) -> List[Dict]:
    """
    LLM-powered fallback to suggest drugs for pathways when no direct matches are found.
    
    Args:
        top_pathways: List of pathway dictionaries with pathway names and associated genes
        disease_context: Optional disease context for more specific recommendations
    
    Returns:
        List of drug recommendation dictionaries matching the expected format
    """
    print(f"LLM fallback called with {len(top_pathways)} pathways")
    if not top_pathways:
        print("No pathways provided to LLM fallback function")
        return []
    def llm_generate_drug_recommendation(prompt):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENAI_API_KEY not found in environment variables")
                return "Unable to generate recommendation (API key not found)"
            
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a clinical pharmacology expert with extensive knowledge of FDA-approved drugs, their mechanisms, and pathway targets. Respond with precise, evidence-based drug recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1,
            )
            result = response.choices[0].message.content.strip()
            print(f"LLM response received: {len(result)} characters")
            return result
        except Exception as e:
            print(f"LLM API call failed: {str(e)}")
            # Fallback: return a placeholder with error info
            return f"Unable to generate recommendation (LLM unavailable: {str(e)})"

    fallback_results = []
    
    print(f"Processing {len(top_pathways)} pathways for LLM drug discovery...")
    
    for i, pathway_info in enumerate(top_pathways):
        pathway_name = pathway_info.get("pathway_name", "Unknown Pathway")
        associated_genes = pathway_info.get("genes", [])
        pathway_regulation = pathway_info.get("regulation", "Unknown")
        print(f"Processing pathway {i+1}: {pathway_name} with {len(associated_genes)} genes (Regulation: {pathway_regulation})")
        
        # Get top genes for context (up to 5 genes with their log2fc values)
        top_genes_info = ""
        if associated_genes:
            top_genes = associated_genes[:5]  # Take top 5 genes
            gene_details = []
            for gene_data in top_genes:
                gene = gene_data.get("gene", "")
                log2fc = gene_data.get("log2fc")
                if log2fc is not None:
                    gene_details.append(f"{gene} (log2FC: {log2fc:.2f})")
                else:
                    gene_details.append(gene)
            top_genes_info = ", ".join(gene_details)
        
        # Create the prompt for drug recommendation
        disease_context_str = f" in the context of {disease_context}" if disease_context else ""
        
        prompt = f"""
        Context: Need a therapeutic drug recommendation for pathway disruption{disease_context_str}.

        Pathway: {pathway_name}
        Pathway Regulation: {pathway_regulation}
        Key affected genes: {top_genes_info if top_genes_info else "Not available"}

        Task: Recommend ONE specific FDA-approved drug that targets this pathway.

        Respond in EXACTLY this format:
        Drug Name: [specific drug name]
        Mechanism: [brief mechanism of action]
        FDA Status: [Approved/Not Approved]
        Evidence: [brief evidence summary (max 50 words) explaining why this drug targets the pathway with the most relevant genes]

        Guidelines:
        - Prioritize FDA-approved drugs
        - Focus on drugs with established clinical evidence
        - If pathway regulation is "UP": Find inhibitors/blockers of the upregulated pathway
        - If pathway regulation is "DOWN": Find activators/inducers of the downregulated pathway
        - Consider the pathway regulation direction when selecting drug mechanism
        - Recommend drug based on the disease context
        - Keep mechanism and evidence brief but informative
        - If no FDA-approved drug exists, suggest the most promising investigational drug and mark FDA Status as "Not Approved"
        - Do NOT recommend any topical creams, ointments, gels, OTC (over-the-counter) drugs, Panadol, vitamins, or acetaminophen products under any circumstances.

        """
        
        # Get LLM recommendation
        llm_response = llm_generate_drug_recommendation(prompt)
        
        # Parse the LLM response
        drug_name = None
        mechanism = "Mechanism not specified"
        fda_status = "Unknown"
        evidence = "Evidence not available"
        
        try:
            lines = llm_response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Drug Name:"):
                    candidate_name = line.split(":", 1)[1].strip()
                    if validate_drug_name(candidate_name):
                        drug_name = candidate_name
                elif line.startswith("Mechanism:"):
                    mechanism = line.split(":", 1)[1].strip()
                elif line.startswith("FDA Status:"):
                    fda_status = line.split(":", 1)[1].strip()
                elif line.startswith("Evidence:"):
                    evidence = line.split(":", 1)[1].strip()
        except Exception:
            # If parsing fails, use the raw response as evidence
            evidence = llm_response[:200] + "..." if len(llm_response) > 200 else llm_response
        
        # Create pathway genes structure matching the expected format
        pathway_genes_with_fc = []
        if associated_genes:
            pathway_genes_with_fc = [
                {
                    "gene": gene_data.get("gene", ""),
                    "log2fc": gene_data.get("log2fc")
                }
                for gene_data in associated_genes[:10]  # Limit to top 10 genes
            ]
        
        # CRITICAL: Only create result if we have a valid drug name
        if drug_name and validate_drug_name(drug_name):
            # Create result matching the expected format
            result = {
                "drug": drug_name,
                "pathway": pathway_name,
                "mechanism": mechanism,
                "approved": "Approved" if fda_status.lower().strip() == "approved" else "Not Approved",
                "pathway_genes": pathway_genes_with_fc,
                "evidence_summary": evidence,
                "priority_rank": i + 1,  # Use index as priority rank
                "final_rank": i + 1,  # Use index as final rank for fallback drugs
                "additional_drugs": []  # No additional drugs in fallback
            }
            
            fallback_results.append(result)
        else:
            print(f"❌ Skipping pathway result - no valid drug name found for pathway: {pathway_name}")
    
    return fallback_results

def validate_drug_name(drug_name: str) -> bool:
    """
    Validate that a drug name is legitimate and not a placeholder.
    
    Args:
        drug_name: Drug name to validate
        
    Returns:
        True if valid, False if placeholder or invalid
    """
    if not drug_name or not isinstance(drug_name, str):
        return False
    
    drug_name = drug_name.strip()
    
    # Invalid/placeholder patterns - use exact matching for common terms
    exact_invalid_patterns = [
        "unknown drug",
        "unknown",
        "not specified",
        "n/a",
        "na",
        "none",
        "placeholder",
        "drug name",
        "generic drug",
        "medication",
        "treatment",
        "therapy",
        "drug",
        "medicine",
        "",
    ]
    
    # Check for exact invalid patterns
    drug_lower = drug_name.lower().strip()
    if drug_lower in exact_invalid_patterns:
        return False
    
    # Check for specific invalid starting patterns
    invalid_start_patterns = [
        "unknown ",
        "generic ",
        "drug ",
        "medication ",
        "treatment ",
        "therapy ",
    ]
    
    for pattern in invalid_start_patterns:
        if drug_lower.startswith(pattern):
            return False
    
    # Must be at least 3 characters and contain letters
    if len(drug_name) < 3 or not any(c.isalpha() for c in drug_name):
        return False
    
    return True

def llm_patient_history_drug_prioritization(drug_candidates: List[Dict], patient_history: str, disease_name: str = None) -> Tuple[List[Dict], str]:
    """
    LLM-powered drug prioritization based on patient's medication and diagnosis history.
    
    Args:
        drug_candidates: List of drug recommendation dictionaries from pathway matching
        patient_history: Text format of patient's medication and diagnosis history
        disease_name: Optional disease context for more specific recommendations
    
    Returns:
        Re-ranked list of drug recommendation dictionaries
    """
    def llm_prioritize_drugs(prompt):
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a clinical oncologist with expertise in personalized medicine and drug resistance mechanisms. You analyze patient treatment history to optimize drug selection and avoid cross-resistance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM prioritization failed: {str(e)}")
            return None

    if not drug_candidates or len(drug_candidates) == 0:
        return drug_candidates, "No drug candidates available for prioritization."
    
    # Prepare drug candidates list for LLM
    drug_list = []
    for i, drug in enumerate(drug_candidates):
        drug_info = f"{i+1}. {drug['drug']} - {drug['mechanism']} (FDA: {drug['approved']}) - Pathway: {drug['pathway']}"
        drug_list.append(drug_info)
    
    drug_candidates_text = "\n".join(drug_list)
    
    # Create the prioritization prompt
    disease_context = f" for {disease_name}" if disease_name else ""
    
    prompt = f"""
        Patient: {disease_name} with extensive prior treatments (see treatment history below).

        Treatment history: {patient_history}

        Analysis: I have performed transcriptome analysis, identified dysregulated pathways, and matched them to an in-house drug database.

        Next step: Here are the drug candidates from my pathway-based ranking:
        {drug_candidates_text}

        Instruction: Please rank these drugs for likely clinical benefit considering:
        1. The patient's prior treatment exposure.
        2. Mechanisms of action already used.
        3. Potential cross-resistance.
        4. Overlap with previously failed therapies.

        Output a ranked list with rationale in EXACTLY this format:

        RANKED DRUGS:
        1. [Drug Name] - [Brief rationale for ranking]
        2. [Drug Name] - [Brief rationale for ranking]
        3. [Drug Name] - [Brief rationale for ranking]
        [Continue for all drugs...]

        RATIONALE SUMMARY:
        [2-3 sentences explaining overall prioritization strategy and key considerations]

        Guidelines:
        - Prioritize drugs with novel mechanisms not previously used
        - Avoid drugs with similar mechanisms to failed treatments
        - Consider FDA approval status but don't let it override clinical logic
        - Focus on drugs that target pathways not addressed by previous treatments
        - Find inhibitors of the pathway that is upregulated in the disease
        - Find activators of the pathway that is downregulated in the disease
        - Recommend drug based on the disease 
        - Consider potential synergistic combinations
        - Be specific about why each drug is ranked where it is
        """
    
    # Get LLM prioritization
    llm_response = llm_prioritize_drugs(prompt)
    
    if not llm_response:
        print("LLM prioritization failed, returning original drug order")
        return drug_candidates, "LLM prioritization failed - using original drug ranking."
    
    # Parse the LLM response to extract ranked drug names, individual rationales, and rationale summary
    ranked_drugs_with_rationales = []
    rationale_summary = "No rationale summary available."
    try:
        lines = llm_response.split('\n')
        in_ranked_section = False
        in_rationale_section = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("RANKED DRUGS:"):
                in_ranked_section = True
                in_rationale_section = False
                continue
            elif line.startswith("RATIONALE SUMMARY:"):
                in_ranked_section = False
                in_rationale_section = True
                continue
            elif in_ranked_section and line and line[0].isdigit():
                # Extract drug name and individual rationale from ranked line (format: "1. Drug Name - rationale")
                parts = line.split(" - ", 1)
                if len(parts) >= 2:
                    drug_part = parts[0]
                    individual_rationale = parts[1]
                    # Remove the number and dot
                    drug_name = drug_part.split(". ", 1)[1] if ". " in drug_part else drug_part
                    ranked_drugs_with_rationales.append({
                        "drug_name": drug_name.strip(),
                        "rationale": individual_rationale.strip()
                    })
                elif len(parts) >= 1:
                    # Fallback if no rationale provided
                    drug_part = parts[0]
                    drug_name = drug_part.split(". ", 1)[1] if ". " in drug_part else drug_part
                    ranked_drugs_with_rationales.append({
                        "drug_name": drug_name.strip(),
                        "rationale": "No specific rationale provided"
                    })
            elif in_rationale_section and line:
                # Collect rationale summary
                if rationale_summary == "No rationale summary available.":
                    rationale_summary = line
                else:
                    rationale_summary += " " + line
    except Exception as e:
        print(f"Error parsing LLM response: {str(e)}")
        return drug_candidates, "Error parsing LLM response - using original drug ranking."
    
    # Re-rank the drug candidates based on LLM prioritization
    if ranked_drugs_with_rationales:
        # Create a mapping of drug names to their original indices
        drug_name_to_index = {drug['drug']: i for i, drug in enumerate(drug_candidates)}
        
        # Create new ranked list
        re_ranked_candidates = []
        used_indices = set()
        
        # Add drugs in LLM-ranked order with their individual rationales
        for ranked_drug in ranked_drugs_with_rationales:
            drug_name = ranked_drug["drug_name"]
            individual_rationale = ranked_drug["rationale"]
            
            # Find the best match for this drug name
            best_match = None
            best_score = 0
            
            for i, drug in enumerate(drug_candidates):
                if i in used_indices:
                    continue
                
                # Calculate similarity score (simple string matching)
                original_name = drug['drug'].lower()
                ranked_name = drug_name.lower()
                
                # Check for exact match or contains
                if original_name == ranked_name:
                    score = 100
                elif original_name in ranked_name or ranked_name in original_name:
                    score = 50
                else:
                    # Calculate partial match score
                    score = sum(1 for word in ranked_name.split() if word in original_name)
                
                if score > best_score:
                    best_score = score
                    best_match = i
            
            if best_match is not None and best_score > 0:
                # Add the individual rationale to the drug
                drug_candidates[best_match]["individual_rationale"] = individual_rationale
                re_ranked_candidates.append(drug_candidates[best_match])
                used_indices.add(best_match)
        
        # Add any remaining drugs that weren't matched
        for i, drug in enumerate(drug_candidates):
            if i not in used_indices:
                re_ranked_candidates.append(drug)
        
        print(f"LLM prioritization completed. Re-ranked {len(re_ranked_candidates)} drugs based on patient history.")
        return re_ranked_candidates, rationale_summary
    
    return drug_candidates, "No drugs were re-ranked - using original order."

def match_drugs_to_pathways(drug_df: pd.DataFrame, sig_path_df: pd.DataFrame, deg_df: pd.DataFrame = None, patient_prefix: str = "patient", disease_name: str = None, patient_history: str = None) -> List[Dict]:
    """
    Robust drug-to-pathway matching with comprehensive error handling and column detection.
    This function will never fail and always returns a valid result.
    """
    try:
        if drug_df is None or drug_df.empty:
            return []
        
        # Create a robust column mapping function
        def _get_column_safe(df, possible_names, default_name):
            """Safely get column name from possible alternatives"""
            available_cols = df.columns.tolist()
            for name in possible_names:
                if name in available_cols:
                    return name
                # Case-insensitive search
                for col in available_cols:
                    if col.lower() == name.lower():
                        return col
            return default_name  # Will be created with default values if missing
        
        # Robust column mapping with fallbacks
        pathway_col = _get_column_safe(drug_df, ["pathway_name", "Pathway", "pathway", "Pathway_Name"], "Pathway")
        name_col = _get_column_safe(drug_df, ["drug_name", "Name", "Drug", "drug", "Drug_Name"], "Name")  
        mechanism_col = _get_column_safe(drug_df, ["mechanism_of_action", "Mechanism", "mechanism", "MOA"], "Mechanism")
        fda_col = _get_column_safe(drug_df, ["fda_approved_status", "FDA_Approved", "fda_status", "approved"], "FDA_Approved")
        evidence_col = _get_column_safe(drug_df, ["justification", "Evidence_Summary", "evidence", "description"], "Evidence_Summary")
        
        # Create safe copy and add missing columns with defaults
        drug_df_safe = drug_df.copy()
        if pathway_col not in drug_df.columns:
            drug_df_safe["Pathway"] = f"Unknown Pathway for {disease_name or 'Disease'}"
        if name_col not in drug_df.columns:
            # Don't create placeholder drug names - handle this later
            drug_df_safe["Name"] = None
        if mechanism_col not in drug_df.columns:
            drug_df_safe["Mechanism"] = "Mechanism under investigation"
        if fda_col not in drug_df.columns:
            drug_df_safe["FDA_Approved"] = "Unknown"
        if evidence_col not in drug_df.columns:
            drug_df_safe["Evidence_Summary"] = "Evidence summary not available"
            
        # Standardize column names
        column_mapping = {}
        if pathway_col != "Pathway" and pathway_col in drug_df.columns:
            column_mapping[pathway_col] = "Pathway"
        if name_col != "Name" and name_col in drug_df.columns:
            column_mapping[name_col] = "Name"
        if mechanism_col != "Mechanism" and mechanism_col in drug_df.columns:
            column_mapping[mechanism_col] = "Mechanism"
        if fda_col != "FDA_Approved" and fda_col in drug_df.columns:
            column_mapping[fda_col] = "FDA_Approved"
        if evidence_col != "Evidence_Summary" and evidence_col in drug_df.columns:
            column_mapping[evidence_col] = "Evidence_Summary"
            
        if column_mapping:
            drug_df_safe = drug_df_safe.rename(columns=column_mapping)
            
    except Exception as e:
        print(f"⚠️  Error in drug DataFrame preparation: {e}")
        return []

    # Prepare DEG gene log2fc mapping if deg_df is provided
    log2fc_map = {}
    try:
        if deg_df is not None and not deg_df.empty:
            gene_col, fc_col, p_value_col = _infer_columns_deg(deg_df, patient_prefix)
            deg_df_clean = deg_df.rename(columns={gene_col: "Gene", fc_col: "log2FC"})
            # Clean gene names for matching
            deg_df_clean["Gene"] = deg_df_clean["Gene"].astype(str).str.strip().str.split(".").str[0].str.upper()
            log2fc_map = deg_df_clean.set_index("Gene")["log2FC"].astype(float).to_dict()
    except Exception as e:
        print(f"⚠️  Error processing DEG data: {e}")
        log2fc_map = {}
    
    # First, collect all matches with robust error handling
    all_matches: List[Dict] = []
    try:
        if sig_path_df is None or sig_path_df.empty:
            return []
            
        for _, prow in sig_path_df.iterrows():
            try:
                # Handle both "Pathway" and "Pathway_Name" column names with fallbacks
                pathway_col = None
                for col_name in ["Pathway", "Pathway_Name", "pathway", "pathway_name"]:
                    if col_name in prow.index:
                        pathway_col = col_name
                        break
                
                if pathway_col is None:
                    print(f"⚠️  No pathway column found in sig_path_df")
                    continue
                    
                pname = str(prow[pathway_col]).lower()
                
                # Safe pathway matching with multiple fallback strategies
                hits = pd.DataFrame()
                try:
                    # Ensure Pathway column exists in drug_df_safe
                    if "Pathway" not in drug_df_safe.columns:
                        print(f"⚠️  Pathway column missing in drug_df, skipping pathway matching")
                        continue
                        
                    # Convert to string and handle NaN values
                    pathway_series = drug_df_safe["Pathway"].astype(str).fillna("unknown")
                    hits = drug_df_safe[pathway_series.str.lower().str.contains(pname, regex=False, na=False)]
                except Exception as matching_error:
                    print(f"⚠️  Error in pathway matching for {pname}: {matching_error}")
                    continue
                
                for _, d in hits.iterrows():
                    try:
                        # Get associated genes and their log2fc for this pathway
                        pathway_genes_with_fc = []
                        if deg_df is not None and pd.notna(prow.get("Pathway_Associated_Genes", None)):
                            try:
                                genes = [g.strip().split(".")[0].upper() for g in str(prow["Pathway_Associated_Genes"]).split(",")]
                                for gene in genes:
                                    log2fc = log2fc_map.get(gene)
                                    pathway_genes_with_fc.append({
                                        "gene": gene,
                                        "log2fc": log2fc
                                    })
                                # Sort genes by absolute log2fc (highest first, None values last)
                                pathway_genes_with_fc = sorted(
                                    pathway_genes_with_fc,
                                    key=lambda x: (abs(x["log2fc"]) if x["log2fc"] is not None else -float("inf")),
                                    reverse=True
                                )
                            except Exception as gene_error:
                                print(f"⚠️  Error processing pathway genes: {gene_error}")
                                pathway_genes_with_fc = []
                        
                        # Safe access to drug properties with validation
                        drug_name = d.get("Name", "")
                        if not validate_drug_name(drug_name):
                            continue  # Skip invalid drug names
                        pathway_name = prow.get(pathway_col, "Unknown Pathway")
                        mechanism = d.get("Mechanism", "Mechanism under investigation")
                        approved_status = d.get("FDA_Approved", "Unknown")
                        evidence_summary = d.get("Evidence_Summary", "Evidence summary not available")
                        priority_rank = prow.get("Priority_Rank", float('inf'))
                        final_rank = d.get("final_rank", float('inf'))
                        
                        all_matches.append({
                            "drug": drug_name,
                            "pathway": pathway_name,
                            "mechanism": mechanism,
                            "approved": approved_status,
                            "pathway_genes": pathway_genes_with_fc,
                            "evidence_summary": evidence_summary,
                            "priority_rank": priority_rank,
                            "final_rank": final_rank,
                        })
                    except Exception as drug_error:
                        print(f"⚠️  Error processing drug entry: {drug_error}")
                        continue
                        
            except Exception as pathway_error:
                print(f"⚠️  Error processing pathway row: {pathway_error}")
                continue
                
    except Exception as e:
        print(f"⚠️  Error in pathway matching loop: {e}")
        return []
    
    # Group matches by pathway with error handling
    pathway_groups = {}
    try:
        for match in all_matches:
            try:
                pathway = match.get("pathway", "Unknown Pathway")
                if pathway not in pathway_groups:
                    pathway_groups[pathway] = []
                pathway_groups[pathway].append(match)
            except Exception as e:
                print(f"⚠️  Error grouping match: {e}")
                continue
    except Exception as e:
        print(f"⚠️  Error in pathway grouping: {e}")
        return []
    
    # Process each pathway group with comprehensive error handling
    final_results = []
    try:
        for pathway, drugs in pathway_groups.items():
            try:
                # Sort drugs within this pathway by: FDA approval first, then by final rank (lower is better), then alphabetical
                sorted_drugs = sorted(drugs, key=lambda x: (
                    x.get("approved", "Unknown") != "Approved",  # FDA approved first
                    x.get("final_rank", float('inf')),  # Lower final rank first (better ranking)
                    x.get("drug", "")  # Alphabetical order as tiebreaker
                ))
                
                # Remove duplicates while preserving order
                seen_drugs = set()
                unique_drugs = []
                for drug in sorted_drugs:
                    try:
                        drug_name = drug.get("drug", "")
                        if validate_drug_name(drug_name) and drug_name not in seen_drugs:
                            unique_drugs.append(drug)
                            seen_drugs.add(drug_name)
                    except Exception as e:
                        print(f"⚠️  Error processing drug for deduplication: {e}")
                        continue
                
                # Take top 6 drugs for this pathway
                top_6_pathway_drugs = unique_drugs[:6]
                
                if top_6_pathway_drugs:
                    try:
                        # Primary drug (first in the list)
                        primary_drug = top_6_pathway_drugs[0]
                        
                        # Additional drugs (up to 5 more)
                        additional_drugs = top_6_pathway_drugs[1:4]
                        
                        # Create the pathway result with safe access
                        primary_drug_name = primary_drug.get("drug", "")
                        if not validate_drug_name(primary_drug_name):
                            continue  # Skip invalid primary drug names
                        
                        result = {
                            "drug": primary_drug_name,
                            "pathway": pathway,
                            "mechanism": primary_drug.get("mechanism", "Mechanism under investigation"),
                            "approved": primary_drug.get("approved", "Unknown"),
                            "pathway_genes": primary_drug.get("pathway_genes", []),
                            "evidence_summary": primary_drug.get("evidence_summary", "Evidence summary not available"),
                            "priority_rank": primary_drug.get("priority_rank", float('inf')),
                            "final_rank": primary_drug.get("final_rank", float('inf')),
                            "additional_drugs": []
                        }
                        
                        # Safely add additional drugs
                        try:
                            additional_valid_drugs = []
                            for drug in additional_drugs:
                                drug_name = drug.get("drug", "")
                                if validate_drug_name(drug_name):
                                    additional_valid_drugs.append({
                                        "drug": drug_name,
                                        "mechanism": drug.get("mechanism", "Mechanism under investigation"),
                                        "approved": drug.get("approved", "Unknown"),
                                        "final_rank": drug.get("final_rank", float('inf'))
                                    })
                            
                            result["additional_drugs"] = additional_valid_drugs
                        except Exception as e:
                            print(f"⚠️  Error adding additional drugs: {e}")
                            result["additional_drugs"] = []
                        
                        final_results.append(result)
                    except Exception as e:
                        print(f"⚠️  Error creating pathway result for {pathway}: {e}")
                        continue
            except Exception as e:
                print(f"⚠️  Error processing pathway group {pathway}: {e}")
                continue
    except Exception as e:
        print(f"⚠️  Error in final results processing: {e}")
        return []
    
    # Sort pathway results by priority rank and return top 6 pathway groups
    try:
        final_results = sorted(final_results, key=lambda x: (
            x.get("approved", "Unknown") != "Approved",  # FDA approved first
            x.get("final_rank", float('inf')),        # Lower priority rank first
            x.get("pathway", "")                         # Alphabetical by pathway name
        ))
    except Exception as e:
        print(f"⚠️  Error sorting final results: {e}")
        # Return unsorted results if sorting fails
        pass
    
    # Fallback mechanism: If fewer than 6 results found, use LLM to suggest additional drugs for top pathways
    try:
        if len(final_results) < 6:
            missing_count = 6 - len(final_results)
            print(f"Only {len(final_results)} drug matches found. Activating LLM fallback to generate {missing_count} additional drug recommendations...")
            
            try:
                print(f"Available columns in sig_path_df: {list(sig_path_df.columns)}")
            except Exception as e:
                print(f"⚠️  Error displaying sig_path_df columns: {e}")
            
            # Get top pathways from sig_path_df sorted by Priority_Rank (excluding already matched pathways)
            try:
                matched_pathways = {result.get("pathway", "") for result in final_results}
                print(f"Already matched pathways: {matched_pathways}")
            except Exception as e:
                print(f"⚠️  Error getting matched pathways: {e}")
                matched_pathways = set()
            try:
                print(f"Total pathways in sig_path_df: {len(sig_path_df)}")
            except Exception as e:
                print(f"⚠️  Error getting sig_path_df length: {e}")
            
            # Safe pathway filtering with column detection
            available_pathways_df = sig_path_df.copy()
            try:
                # Find the correct pathway column
                pathway_col_filter = None
                for col_name in ["Pathway", "Pathway_Name", "pathway", "pathway_name"]:
                    if col_name in sig_path_df.columns:
                        pathway_col_filter = col_name
                        break
                
                if pathway_col_filter:
                    available_pathways_df = sig_path_df[~sig_path_df[pathway_col_filter].isin(matched_pathways)]
                    print(f"Available pathways after filtering: {len(available_pathways_df)}")
                else:
                    print(f"⚠️  No pathway column found for filtering, using all pathways")
            except Exception as e:
                print(f"⚠️  Error filtering pathways: {e}")
                available_pathways_df = sig_path_df.copy()
            
            # Safe pathway selection for LLM
            try:
                if "Priority_Rank" in available_pathways_df.columns:
                    top_pathways_df = available_pathways_df.nsmallest(6, "Priority_Rank")
                    print(f"Using Priority_Rank for sorting, found {len(top_pathways_df)} pathways")
                else:
                    top_pathways_df = available_pathways_df.head(6)
                    print(f"Priority_Rank column not found, using first 6 pathways")
            except Exception as e:
                print(f"⚠️  Error selecting pathways: {e}")
                top_pathways_df = available_pathways_df.head(6)
            
            print(f"Found {len(top_pathways_df)} pathways for LLM fallback processing")
            
            # Prepare pathway information for LLM with associated genes
            pathway_info_for_llm = []
            
            try:
                for _, prow in top_pathways_df.iterrows():
                    try:
                        # Handle both "Pathway" and "Pathway_Name" column names with fallbacks
                        pathway_col = None
                        for col_name in ["Pathway", "Pathway_Name", "pathway", "pathway_name"]:
                            if col_name in prow.index:
                                pathway_col = col_name
                                break
                        
                        pathway_name = prow.get(pathway_col, "Unknown Pathway") if pathway_col else "Unknown Pathway"
                        
                        # Get associated genes and their log2fc for this pathway
                        pathway_genes_with_fc = []
                        try:
                            if deg_df is not None and pd.notna(prow.get("Pathway_Associated_Genes", None)):
                                genes = [g.strip().split(".")[0].upper() for g in str(prow["Pathway_Associated_Genes"]).split(",")]
                                for gene in genes:
                                    log2fc = log2fc_map.get(gene)
                                    pathway_genes_with_fc.append({
                                        "gene": gene,
                                        "log2fc": log2fc
                                    })
                                # Sort genes by absolute log2fc (highest first, None values last)
                                pathway_genes_with_fc = sorted(
                                    pathway_genes_with_fc,
                                    key=lambda x: (abs(x["log2fc"]) if x["log2fc"] is not None else -float("inf")),
                                    reverse=True
                                )
                        except Exception as gene_error:
                            print(f"⚠️  Error processing genes for {pathway_name}: {gene_error}")
                            pathway_genes_with_fc = []
                            
                        pathway_info_for_llm.append({
                            "pathway_name": pathway_name,
                            "genes": pathway_genes_with_fc,
                            "regulation": prow.get("Reg", "Unknown")  # Add pathway regulation information
                        })
                    except Exception as pathway_error:
                        print(f"⚠️  Error processing pathway row in LLM prep: {pathway_error}")
                        continue
            except Exception as e:
                print(f"⚠️  Error preparing pathway info for LLM: {e}")
                pathway_info_for_llm = []
            
            # Use LLM fallback to generate additional drug recommendations
            try:
                if pathway_info_for_llm:
                    print(f"Calling LLM fallback with {len(pathway_info_for_llm)} pathways")
                    additional_results = llm_drug_discovery_fallback(pathway_info_for_llm, disease_context=disease_name)
                    print(f"LLM fallback generated {len(additional_results)} additional drug recommendations.")
                    
                    # Combine existing results with additional LLM results
                    final_results.extend(additional_results)
                else:
                    print(f"⚠️  No pathway info available for LLM fallback")
            except Exception as e:
                print(f"⚠️  LLM fallback failed: {str(e)}")
                # Keep existing results even if LLM fails
    except Exception as e:
        print(f"⚠️  Error in fallback mechanism: {e}")
        # Continue with existing results
    
    # Apply patient history-based prioritization if patient history is provided
    rationale_summary = None
    try:
        if patient_history and final_results:
            print("Applying patient history-based drug prioritization...")
            try:
                final_results, rationale_summary = llm_patient_history_drug_prioritization(final_results, patient_history, disease_name)
                print("Patient history-based prioritization completed successfully.")
            except Exception as e:
                print(f"⚠️  Patient history prioritization failed: {str(e)}")
                # Keep original results if prioritization fails
    except Exception as e:
        print(f"⚠️  Error in patient history section: {e}")
    
    # Add rationale summary to the first drug result for display in the report
    try:
        if rationale_summary and final_results:
            final_results[0]["patient_history_rationale"] = rationale_summary
    except Exception as e:
        print(f"⚠️  Error adding rationale summary: {e}")
    
    # Safe return with fallback
    try:
        return final_results[:6] if final_results else []
    except Exception as e:
        print(f"⚠️  Error returning final results: {e}")
        return []