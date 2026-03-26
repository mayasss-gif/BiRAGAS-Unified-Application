import os
from openai import OpenAI
from decouple import config

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

def llm_generate_molecular_signature_description(disease_name: str, pathway_name: str, regulation_status: str,
                                               top_genes: list, validation_status: str = None) -> str:
    """
    Generate LLM-based description for molecular signatures pathways (20-30 words).
    
    Args:
        disease_name: Name of the disease
        pathway_name: Name of the pathway
        regulation_status: "Upregulated" or "Downregulated"
        top_genes: List of top genes with log2fc values
        validation_status: Optional validation status
    
    Returns:
        20-30 word description of how the pathway contributes to disease
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found for molecular signature description generation")
            return f"In {disease_name}, {regulation_status.lower()} {pathway_name} affects immune signaling and contributes to disease pathophysiology."
        
        client = OpenAI(api_key=api_key)
        
        # Format genes for prompt
        genes_text = ""
        if top_genes:
            gene_strs = []
            for gene in top_genes[:3]:  # Use top 3 genes
                if isinstance(gene, dict) and 'gene' in gene and 'log2fc' in gene:
                    direction = "↑" if gene['log2fc'] > 0 else "↓"
                    gene_strs.append(f"{gene['gene']} {direction}")
                elif isinstance(gene, str):
                    gene_strs.append(gene)
            genes_text = ", ".join(gene_strs)
        
        prompt = f"""You are a biomedical expert specializing in pathway analysis.

            Context:
            - Disease: {disease_name}
            - Pathway: {pathway_name}
            - Regulation: {regulation_status}
            - Key Genes: {genes_text}
            {f"- Validation: {validation_status}" if validation_status else ""}

            Task: Write a precise 20-30 word description explaining how this pathway's dysregulation contributes to {disease_name}.

            Requirements:
            - Focus on the mechanistic impact on immune/cellular function
            - Mention the key genes and their direction of change
            - Explain clinical relevance to {disease_name}
            - Keep to exactly 20-30 words
            - Be scientifically accurate

            Respond with only the description, no additional text."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a clinical pathologist expert in molecular signatures and pathway analysis. Provide precise, evidence-based descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80,
            temperature=0.1,
        )
        
        result = response.choices[0].message.content.strip()
        
        # Validate word count (20-30 words)
        word_count = len(result.split())
        if word_count < 15 or word_count > 35:
            # Fallback if word count is off
            return f"In {disease_name}, {regulation_status.lower()} {pathway_name} with key genes {genes_text.split(',')[0] if genes_text else 'identified'} affects immune signaling and disease progression."
        
        return result
        
    except Exception as e:
        print(f"LLM molecular signature description generation failed: {str(e)}")
        return f"In {disease_name}, {regulation_status.lower()} {pathway_name} affects immune signaling and contributes to disease pathophysiology."


def llm_generate_pathway_consequence(disease_name: str, pathway_name: str, regulation_status: str, 
                                   validation_status: str, validation_justification: str) -> str:
    """
    Generate LLM-based pathway consequence for pathogenic pathways only.
    
    Args:
        disease_name: Name of the disease
        pathway_name: Name of the pathway
        regulation_status: "Upregulated" or "Downregulated"
        validation_status: Should be "Pathogenic" for this function
        validation_justification: Evidence from validation
    
    Returns:
        One sentence describing how the pathway contributes to disease
    """
    # Only generate for pathogenic pathways
    if validation_status != "Pathogenic":
        return f"Pathway promotes disease activity in {disease_name}."
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found for pathway consequence generation")
            return f"{regulation_status} {pathway_name} promotes disease activity in {disease_name}."
        
        client = OpenAI(api_key=api_key)
        
        prompt = f"""You are a biomedical expert.

            Context:
            - Disease: {disease_name}
            - Pathway: {pathway_name}
            - Dysregulation: {regulation_status}
            - Validation Status: {validation_status}
            - Evidence: {validation_justification}

            Task:
            Write ONE concise, scientifically accurate sentence describing how dysregulation
            of this pathway (classified as Pathogenic) contributes to {disease_name}.

            Rules:
            - Only write if validation_status == "Pathogenic".
            - Focus on how the pathway promotes disease activity (inflammation, autoimmunity, immune deficiency, etc.).
            - Mention immune/tissue consequence directly relevant to {disease_name}.
            - Keep it strictly one sentence (max 20 words).
            - Do not invent mechanisms; if unclear, say "Pathway promotes disease activity in {disease_name}."

            Respond with only the single sentence, no additional text."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a clinical pathology expert specializing in pathway-disease mechanisms. Provide precise, evidence-based responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.1,
        )
        
        result = response.choices[0].message.content.strip()
        
        # Validate result length and format
        if len(result.split()) > 25:  # Allow slight flexibility
            result = f"{regulation_status} {pathway_name} promotes disease activity in {disease_name}."
        
        return result
        
    except Exception as e:
        print(f"LLM pathway consequence generation failed: {str(e)}")
        return f"{regulation_status} {pathway_name} promotes disease activity in {disease_name}."


def llm_generate_downregulated_pathway_description(disease_name: str, pathway_name: str, 
                                              top_genes: list, validation_status: str = None) -> str:
    """
    Generate LLM-based description specifically for downregulated pathways (20-30 words).
    
    Args:
        disease_name: Name of the disease
        pathway_name: Name of the pathway
        top_genes: List of top genes with log2fc values
        validation_status: Optional validation status
    
    Returns:
        20-30 word description of how the downregulated pathway contributes to disease
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found for downregulated pathway description generation")
            return f"In {disease_name}, reduced {pathway_name} activity impairs normal cellular function, contributing to disease progression."
        
        client = OpenAI(api_key=api_key)
        
        # Format genes for prompt
        genes_text = ""
        if top_genes:
            gene_strs = []
            for gene in top_genes[:3]:  # Use top 3 genes
                if isinstance(gene, dict) and 'gene' in gene and 'log2fc' in gene:
                    direction = "↓"  # Always down for downregulated pathways
                    gene_strs.append(f"{gene['gene']} {direction}")
                elif isinstance(gene, str):
                    gene_strs.append(f"{gene} ↓")
            genes_text = ", ".join(gene_strs)
        
        prompt = f"""You are a biomedical expert specializing in pathway analysis.

            Context:
            - Disease: {disease_name}
            - Pathway: {pathway_name}
            - Regulation: Downregulated
            - Key Genes: {genes_text}
            {f"- Validation: {validation_status}" if validation_status else ""}

            Task: Write a precise 20-30 word description explaining how this downregulated pathway contributes to {disease_name}.

            Requirements:
            - Focus on loss-of-function consequences (e.g., impaired defense, reduced protection, insufficient response)
            - Emphasize how decreased pathway activity contributes to disease pathology
            - Mention the key genes and their reduced expression
            - Explain clinical relevance to {disease_name}
            - Keep to exactly 20-30 words
            - Be scientifically accurate
            - Use terms like "reduced," "decreased," "impaired," "insufficient," or "compromised"

            Respond with only the description, no additional text."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a clinical pathologist expert in molecular signatures and pathway analysis, specializing in downregulated pathways. Provide precise, evidence-based descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80,
            temperature=0.1,
        )
        
        result = response.choices[0].message.content.strip()
        
        # Validate word count (20-30 words)
        word_count = len(result.split())
        if word_count < 15 or word_count > 35:
            # Fallback if word count is off
            return f"In {disease_name}, reduced {pathway_name} activity with decreased {genes_text.split(',')[0] if genes_text else 'gene expression'} impairs cellular function, promoting disease progression."
        
        return result
        
    except Exception as e:
        print(f"LLM downregulated pathway description generation failed: {str(e)}")
        return f"In {disease_name}, reduced {pathway_name} activity impairs normal cellular function, contributing to disease progression."


def llm_generate_drug_description(drug_name: str, mechanisms: list, disease_name: str, status: str, 
                                  pathway_association: str = None, justification: str = None) -> str:
    """
    Generate a clear, clinical description for a drug's role in treating the disease using LLM.
    
    Args:
        drug_name: Name of the drug
        mechanisms: List of drug mechanisms
        disease_name: Target disease
        status: Drug status (FDA Approved, Clinical Trial, Experimental)
        pathway_association: Associated pathways
        justification: Scientific justification
    
    Returns:
        Clear, clinical description of drug's therapeutic role
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found for drug description generation")
            return f"Therapeutic agent for {disease_name} treatment."
        
        if not drug_name or not disease_name:
            return f"Therapeutic agent for {disease_name} treatment."
        
        # Prepare mechanism text
        mechanism_text = "Unknown mechanism"
        if mechanisms:
            if isinstance(mechanisms, list):
                mechanism_text = ', '.join([str(m) for m in mechanisms[:2]])  # Limit to 2 mechanisms
            else:
                mechanism_text = str(mechanisms)
        
        client = OpenAI(api_key=api_key)
        
        # Create focused prompt for drug description
        prompt = f"""You are a clinical pharmacology expert.

            Context:
            - Drug: {drug_name}
            - Disease: {disease_name}
            - Status: {status}
            - Mechanisms: {mechanism_text}
            {f"- Pathways: {pathway_association}" if pathway_association else ""}
            {f"- Evidence: {justification}" if justification else ""}

            Task:
            Write 1-2 clear, clinical sentences describing this drug's therapeutic role for {disease_name}.

            Rules:
            - Focus on therapeutic benefit and clinical impact
            - Use medical terminology appropriate for clinical reports
            - Mention mechanism briefly if relevant
            - Do not repeat the drug name in the description
            - Sound authoritative and evidence-based
            - Maximum 40 words total
            - End with period

            Examples:
            "Targets inflammatory pathways to reduce joint destruction and disease activity."
            "Inhibits tumor growth through dual kinase blockade, showing efficacy in advanced treatment."
            "Modulates immune responses to control autoimmune inflammation and improve outcomes."

            Respond with only the description, no additional text."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a clinical pharmacology expert specializing in drug mechanisms and therapeutic applications. Provide precise, evidence-based responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80,
            temperature=0.1,
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean the response
        description = result.strip('"\'')
        if not description.endswith('.'):
            description += '.'
        
        # Validate length
        if len(description.split()) > 45:
            description = f"Therapeutic agent with {mechanism_text.lower()} for {disease_name} treatment."
        
        return description
        
    except Exception as e:
        print(f"LLM drug description generation failed: {str(e)}")
        # Fallback to mechanism-based description
        if mechanisms and str(mechanisms) != "Unknown mechanism":
            return f"Therapeutic agent targeting {str(mechanisms).lower()} for {disease_name} treatment."
        else:
            return f"Evidence-based therapeutic option for {disease_name} management."