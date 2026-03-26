import pandas as pd
import requests
import openai
import time
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from tqdm import tqdm
from rich import print
from typing import List, Tuple, Dict, Optional
from .config import CATEGORIZER_CSV_PATH
from .helpers import logger
from pathlib import Path
from datetime import datetime

# === CONFIG ===
# input_file = r"C:\Ayass Bio Work\Pathways.csv" # Use local test file
# output_file = "categorized_pathways.csv"
# sqlite_db_file = "classification_memory.sqlite"  # Use local database

# === DATABASE SETUP ===
# Create directory if needed (only if path has directory)
if os.path.dirname(CATEGORIZER_CSV_PATH):
    os.makedirs(os.path.dirname(CATEGORIZER_CSV_PATH), exist_ok=True)

# Initialize CSV cache file
def init_csv_cache():
    """Initialize the CSV cache file with proper headers if it doesn't exist."""
    if not Path(CATEGORIZER_CSV_PATH).exists():
        cache_df = pd.DataFrame(columns=['pathway', 'main_class', 'sub_class', 'source', 'confidence', 'created_at'])
        cache_df.to_csv(CATEGORIZER_CSV_PATH, index=False)
        print("[yellow]Created new CSV cache file[/yellow]")
    else:
        # Check if we need to add missing columns
        cache_df = pd.read_csv(CATEGORIZER_CSV_PATH)
        if 'confidence' not in cache_df.columns:
            cache_df['confidence'] = 1.0
        if 'created_at' not in cache_df.columns:
            cache_df['created_at'] = datetime.now().isoformat()
        cache_df.to_csv(CATEGORIZER_CSV_PATH, index=False)

# Initialize the cache
init_csv_cache()

# === UTILITY FUNCTIONS ===
def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def insert_into_cache(pathway: str, main_class: str, sub_class: str, source: str, confidence: float = 1.0):
    """Insert pathway classification into cache."""
    try:
        # Read existing cache
        cache_df = pd.read_csv(CATEGORIZER_CSV_PATH)
        
        # Check if pathway already exists
        if pathway in cache_df['pathway'].values:
            # Update existing entry
            cache_df.loc[cache_df['pathway'] == pathway, 'main_class'] = main_class
            cache_df.loc[cache_df['pathway'] == pathway, 'sub_class'] = sub_class
            cache_df.loc[cache_df['pathway'] == pathway, 'source'] = source
            cache_df.loc[cache_df['pathway'] == pathway, 'confidence'] = confidence
            cache_df.loc[cache_df['pathway'] == pathway, 'created_at'] = datetime.now().isoformat()
        else:
            # Add new entry
            new_row = pd.DataFrame({
                'pathway': [pathway],
                'main_class': [main_class],
                'sub_class': [sub_class],
                'source': [source],
                'confidence': [confidence],
                'created_at': [datetime.now().isoformat()]
            })
            cache_df = pd.concat([cache_df, new_row], ignore_index=True)
        
        # Save back to CSV
        cache_df.to_csv(CATEGORIZER_CSV_PATH, index=False)
        
    except Exception as e:
        print(f"[red]Error inserting into cache: {e}[/red]")

def find_similar_pathway(pathway: str, threshold: float = 0.8) -> Optional[Tuple[str, str, str, float]]:
    """Find similar pathways in cache using fuzzy matching."""
    try:
        if not Path(CATEGORIZER_CSV_PATH).exists():
            return None
            
        cache_df = pd.read_csv(CATEGORIZER_CSV_PATH)
        if cache_df.empty:
            return None
            
        best_match = None
        best_similarity = 0
        
        for _, row in cache_df.iterrows():
            similarity = calculate_similarity(pathway, row['pathway'])
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = (row['pathway'], row['main_class'], row['sub_class'], row['source'], similarity)
        
        return best_match
    except Exception as e:
        print(f"[red]Error finding similar pathway: {e}[/red]")
        return None

def get_cached_pathway(pathway: str) -> Optional[Tuple[str, str, str]]:
    """Get cached pathway classification."""
    try:
        if not Path(CATEGORIZER_CSV_PATH).exists():
            return None
            
        cache_df = pd.read_csv(CATEGORIZER_CSV_PATH)
        if cache_df.empty:
            return None
            
        match = cache_df[cache_df['pathway'] == pathway]
        if not match.empty:
            row = match.iloc[0]
            return (row['main_class'], row['sub_class'], row['source'])
        
        return None
    except Exception as e:
        print(f"[red]Error getting cached pathway: {e}[/red]")
        return None

# === ENHANCED KEGG CLASSIFICATION ===
def get_kegg_classification(pathway_name: str) -> Optional[Tuple[str, str, str]]:
    """Enhanced KEGG classification with better error handling."""
    try:
        # Check cache first
        cached_result = get_cached_pathway(pathway_name)
        if cached_result:
            return cached_result

        # Try fuzzy matching
        similar = find_similar_pathway(pathway_name, threshold=0.85)
        if similar:
            cached_pathway, main_class, sub_class, source, similarity = similar
            print(f"[yellow] Using fuzzy match ({similarity:.2f}): {cached_pathway}[/yellow]")
            # Cache the new pathway with the same classification
            insert_into_cache(pathway_name, main_class, sub_class, source, similarity)
            return main_class, sub_class, source

        # Try multiple KEGG API approaches
        main_class = ""
        sub_class = ""
        
        # Method 1: Direct pathway search
        try:
            url = f"http://rest.kegg.jp/find/pathway/{pathway_name}"
            response = requests.get(url, timeout=10)
            if response.ok and response.text.strip():
                first_line = response.text.strip().split("\n")[0]
                kegg_id = first_line.split("\t")[0]

                # Get detailed information
                detail_url = f"http://rest.kegg.jp/get/{kegg_id}"
                detail_resp = requests.get(detail_url, timeout=10)
                if detail_resp.ok:
                    lines = detail_resp.text.split("\n")

                    # Look for classification information
                    class_line = [line for line in lines if line.startswith("CLASS")]
                    if class_line:
                        class_info = class_line[0].replace("CLASS       ", "").strip()
                        if ";" in class_info:
                            parts = class_info.split(";")
                            main_class = parts[0].strip()
                            sub_class = parts[1].strip() if len(parts) > 1 else ""
                        else:
                            main_class = class_info
                            sub_class = "General Process"
        except Exception as e:
            print(f"[yellow] KEGG Method 1 failed for {pathway_name}: {e}[/yellow]")

        # Method 2: Try searching with modified pathway name
        if not main_class:
            try:
                # Remove common prefixes/suffixes and try again
                simplified_name = pathway_name.replace("pathway", "").replace("signaling", "").strip()
                url = f"http://rest.kegg.jp/find/pathway/{simplified_name}"
                response = requests.get(url, timeout=10)
                if response.ok and response.text.strip():
                    # Process similar to method 1
                    first_line = response.text.strip().split("\n")[0]
                    kegg_id = first_line.split("\t")[0]
                    
                    detail_url = f"http://rest.kegg.jp/get/{kegg_id}"
                    detail_resp = requests.get(detail_url, timeout=10)
                    if detail_resp.ok:
                        lines = detail_resp.text.split("\n")
                        class_line = [line for line in lines if line.startswith("CLASS")]
                        if class_line:
                            class_info = class_line[0].replace("CLASS       ", "").strip()
                            if ";" in class_info:
                                parts = class_info.split(";")
                                main_class = parts[0].strip()
                                sub_class = parts[1].strip() if len(parts) > 1 else ""
                            else:
                                main_class = class_info
                                sub_class = "General Process"
            except Exception as e:
                print(f"[yellow] KEGG Method 2 failed for {pathway_name}: {e}[/yellow]")

        # Validate and clean up results
        if main_class:
            # Clean up the classification
            main_class = main_class.strip()
            sub_class = sub_class.strip() if sub_class else "General Process"
            
            # Ensure sub_class is not empty
            if not sub_class:
                sub_class = "General Process"
            
            # Cache the result
            insert_into_cache(pathway_name, main_class, sub_class, "KEGG", 1.0)
            time.sleep(0.5)  # Rate limiting
            return main_class, sub_class, "KEGG"
        
        print(f"[red] KEGG classification failed completely for: {pathway_name}[/red]")
        return None
        
    except Exception as e:
        print(f"[red] KEGG error for {pathway_name}: {e}[/red]")
        return None

# === BATCH LLM CLASSIFICATION ===
def batch_classify_pathways_llm(pathways: List[Tuple[str, str]], batch_size: int = 10) -> Dict[str, Tuple[str, str, str]]:
    """Classify multiple pathways in a single LLM call to minimize API usage."""
    if not pathways:
        return {}
    
    results = {}
    
    # Process in batches
    for i in range(0, len(pathways), batch_size):
        batch = pathways[i:i + batch_size]
        batch_pathways = [pathway for pathway, ontology in batch]
        batch_ontologies = [ontology for pathway, ontology in batch]
        
        # Create batch prompt
        pathways_text = "\n".join([f"{j+1}. {pathway} ({ontology})" for j, (pathway, ontology) in enumerate(batch)])
        
        prompt = f"""
                  You are a biomedical ontology expert. Classify each of the following pathways into a **Main_Class** and **Sub_Class** based on biological function.
 
                🔹 Use one of the following functional categories for Main_Class (choose **only one** per pathway):
                
                - Metabolism
                - Signal Transduction
                - Immune System
                - Cell Cycle
                - Cancer
                - Apoptosis
                - Developmental Biology
                - Genetic Information Processing
                - Environmental Information Processing
                - Cellular Processes
                - Disease Mechanisms
                
                🔹 Guidelines for Sub_Class:
                
                - Sub_Class must describe a **more specific role, pathway type, or molecular mechanism** under the Main_Class.
                - Sub_Class **must not** simply repeat or rephrase the pathway name. It should reflect a real **biological process**.
                - Sub_Class must be **descriptive and consistent**, preferably chosen from the list below (or inferred logically if not listed).
                - If uncertain, make a **biologically educated guess**, but avoid vague terms or placeholder language.
                
                📘 Recommended Sub_Class Examples:
                
                🧬 **Metabolism**
                - Carbohydrate Metabolism
                - Lipid Metabolism
                - Amino Acid Metabolism
                - Nucleotide Metabolism
                - Xenobiotic Metabolism
                - Energy Metabolism
                - Vitamin and Cofactor Metabolism
                - Sulfur Metabolism
                - Glycan Biosynthesis and Metabolism
                - Alcohol and Aldehyde Metabolism
                
                📡 **Signal Transduction**
                - MAPK Signaling
                - PI3K-Akt Signaling
                - GPCR Signaling
                - JAK-STAT Pathway
                - TGF-beta Signaling
                - Wnt Signaling
                - Calcium Signaling
                - Notch Signaling
                - Hedgehog Signaling
                - NF-kappa B Signaling
                - mTOR Signaling
                
                🛡️ **Immune System**
                - Innate Immunity
                - Adaptive Immunity
                - Antigen Presentation
                - Cytokine Signaling
                - Complement Cascade
                - Toll-like Receptor Signaling
                - B Cell Receptor Signaling
                - T Cell Receptor Signaling
                - Interferon Signaling
                - Inflammasome Activation
                
                ⏱️ **Cell Cycle**
                - G1/S Transition
                - G2/M Regulation
                - DNA Damage Checkpoint
                - M Phase Control
                - Spindle Assembly Checkpoint
                - Chromosome Segregation
                - Centrosome Duplication
                
                🧬 **Cancer**
                - Tumor Suppressor Pathways
                - Oncogenic Signaling
                - DNA Repair in Cancer
                - Angiogenesis in Tumors
                - Cancer Cell Invasion and Metastasis
                - Epithelial-Mesenchymal Transition (EMT)
                
                ☠️ **Apoptosis**
                - Intrinsic Pathway
                - Extrinsic Pathway
                - Caspase Cascade
                - Bcl-2 Family Regulation
                - Apoptosome Formation
                - Death Receptor Pathways
                
                🧠 **Developmental Biology**
                - Embryogenesis
                - Neural Development
                - Limb Formation
                - Organogenesis
                - Somitogenesis
                - Stem Cell Differentiation
                
                📚 **Genetic Information Processing**
                - Transcriptional Regulation
                - mRNA Splicing
                - Translation Initiation
                - DNA Replication
                - Chromatin Modification
                - RNA Editing and Processing
                - Nonsense-Mediated Decay
                
                🌱 **Environmental Information Processing**
                - Hypoxia Response
                - Drug Metabolism
                - Oxidative Stress Response
                - Heat Shock Response
                - Heavy Metal Detoxification
                - Endoplasmic Reticulum Stress
                - Radiation Response
                
                🔬 **Cellular Processes**
                - Cell Adhesion
                - Organelle Organization
                - Cytoskeletal Dynamics
                - Vesicle Transport
                - Autophagy
                - Endocytosis and Exocytosis
                - Cell Polarity Establishment
                
                🦠 **Disease Mechanisms**
                - Neurodegeneration
                - Autoimmune Mechanisms
                - Viral Infection Response
                - Fibrosis Pathways
                - Inflammation-Driven Pathology
                - Metabolic Syndrome
                - Cardiovascular Disease Mechanisms
                - Bacterial Infection Pathways
                
                ---
                
                🛑 IMPORTANT:
                - DO NOT leave any Main_Class or Sub_Class empty.
                - DO NOT use the pathway name directly as the Sub_Class.
                - Follow the exact return format below.
                
                📥 Pathways to classify:
                {pathways_text}
                
                📤 Return format (strictly one line per pathway):
                1. Main_Class: [Main_Class] | Sub_Class: [Sub_Class]
                2. Main_Class: [Main_Class] | Sub_Class: [Sub_Class]
                ... up to {len(batch)} total
                
                Ensure the output is **biologically consistent**, descriptive, and follows the above categories wherever possible.
                  """
        
        try:
            client = openai.OpenAI()
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000
            )
            
            text = res.choices[0].message.content
            lines = text.strip().split('\n')
            
            # Parse results more robustly
            parsed_count = 0
            for line in lines:
                if parsed_count >= len(batch):
                    break
                    
                # Skip empty lines and non-classification lines
                if not line.strip() or '|' not in line:
                    continue
                    
                # Try to extract classification
                try:
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 2:
                            main_part = parts[0].strip()
                            sub_part = parts[1].strip()
                            
                            # Extract main class
                            if ':' in main_part:
                                main_class = main_part.split(':')[1].strip()
                            else:
                                main_class = main_part.strip()
                            
                            # Extract sub class
                            if ':' in sub_part:
                                sub_class = sub_part.split(':')[1].strip()
                            else:
                                sub_class = sub_part.strip()
                            
                            # Remove leading numbers if present
                            main_class = re.sub(r'^\d+\.?\s*', '', main_class)
                            sub_class = re.sub(r'^\d+\.?\s*', '', sub_class)
                            
                            # Ensure we have valid classifications
                            if main_class and sub_class:
                                pathway, ontology = batch[parsed_count]
                                results[pathway] = (main_class, sub_class, ontology)
                                
                                # Cache the result
                                insert_into_cache(pathway, main_class, sub_class, ontology, 0.9)
                                parsed_count += 1
                except Exception as e:
                    print(f"[red] Error parsing line: {line} - {e}[/red]")
                    continue
            
            # Handle any unparsed pathways with fallback
            if parsed_count < len(batch):
                print(f"[yellow] Only parsed {parsed_count}/{len(batch)} pathways from batch, using fallback for remaining[/yellow]")
                for j in range(parsed_count, len(batch)):
                    pathway, ontology = batch[j]
                    fallback_result = classify_single_pathway_llm(pathway, ontology)
                    if fallback_result[0] and fallback_result[1]:
                        results[pathway] = fallback_result
                    else:
                        # Ultimate fallback
                        results[pathway] = ("Cellular Processes", "Unknown Process", ontology)
                        insert_into_cache(pathway, "Cellular Processes", "Unknown Process", ontology, 0.5)
            
            print(f"[green] Batch processed: {len(results)} pathways classified[/green]")
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"[red] Batch LLM error: {e}[/red]")
            # Fallback to individual classification for this batch
            for pathway, ontology in batch:
                fallback_result = classify_single_pathway_llm(pathway, ontology)
                if fallback_result[0] and fallback_result[1]:
                    results[pathway] = fallback_result
                else:
                    # Ultimate fallback
                    results[pathway] = ("Cellular Processes", "Unknown Process", ontology)
                    insert_into_cache(pathway, "Cellular Processes", "Unknown Process", ontology, 0.5)
    
    return results

def classify_single_pathway_llm(pathway_name: str, ontology: str) -> Tuple[str, str, str]:
    """Fallback single pathway LLM classification with better error handling."""
    try:
        prompt = f"""
                  You are a biomedical ontology expert. Classify the following pathway from the {ontology} database into a **Main_Class** and **Sub_Class** based on biological function.
 
                    🔹 Use one of the following fixed categories for Main_Class:
                    - Metabolism
                    - Signal Transduction
                    - Immune System
                    - Cell Cycle
                    - Cancer
                    - Apoptosis
                    - Developmental Biology
                    - Genetic Information Processing
                    - Environmental Information Processing
                    - Cellular Processes
                    - Disease Mechanisms
                    
                    🔹 Sub_Class must be a **specific mechanism, process, or molecular function** (not a repetition of the pathway name). Use one of the predefined categories if applicable:
                    
                    **Expanded Sub_Class Options:**
                    
                    🧬 **Metabolism**
                    - Carbohydrate Metabolism
                    - Lipid Metabolism
                    - Amino Acid Metabolism
                    - Nucleotide Metabolism
                    - Xenobiotic Metabolism
                    - Energy Metabolism
                    - Vitamin and Cofactor Metabolism
                    - Sulfur Metabolism
                    - Glycan Biosynthesis and Metabolism
                    - Alcohol and Aldehyde Metabolism
                    
                    📡 **Signal Transduction**
                    - MAPK Signaling
                    - PI3K-Akt Signaling
                    - GPCR Signaling
                    - JAK-STAT Pathway
                    - TGF-beta Signaling
                    - Wnt Signaling
                    - Calcium Signaling
                    - Notch Signaling
                    - Hedgehog Signaling
                    - NF-kappa B Signaling
                    - mTOR Signaling
                    
                    🛡️ **Immune System**
                    - Innate Immunity
                    - Adaptive Immunity
                    - Antigen Presentation
                    - Cytokine Signaling
                    - Complement Cascade
                    - Toll-like Receptor Signaling
                    - B Cell Receptor Signaling
                    - T Cell Receptor Signaling
                    - Interferon Signaling
                    - Inflammasome Activation
                    
                    ⏱️ **Cell Cycle**
                    - G1/S Transition
                    - G2/M Regulation
                    - DNA Damage Checkpoint
                    - M Phase Control
                    - Spindle Assembly Checkpoint
                    - Chromosome Segregation
                    - Centrosome Duplication
                    
                    🧬 **Cancer**
                    - Tumor Suppressor Pathways
                    - Oncogenic Signaling
                    - DNA Repair in Cancer
                    - Angiogenesis in Tumors
                    - Cancer Cell Invasion and Metastasis
                    - Epithelial-Mesenchymal Transition (EMT)
                    
                    ☠️ **Apoptosis**
                    - Intrinsic Pathway
                    - Extrinsic Pathway
                    - Caspase Cascade
                    - Bcl-2 Family Regulation
                    - Apoptosome Formation
                    - Death Receptor Pathways
                    
                    🧠 **Developmental Biology**
                    - Embryogenesis
                    - Neural Development
                    - Limb Formation
                    - Organogenesis
                    - Somitogenesis
                    - Stem Cell Differentiation
                    
                    📚 **Genetic Information Processing**
                    - Transcriptional Regulation
                    - mRNA Splicing
                    - Translation Initiation
                    - DNA Replication
                    - Chromatin Modification
                    - RNA Editing and Processing
                    - Nonsense-Mediated Decay
                    
                    🌱 **Environmental Information Processing**
                    - Hypoxia Response
                    - Drug Metabolism
                    - Oxidative Stress Response
                    - Heat Shock Response
                    - Heavy Metal Detoxification
                    - Endoplasmic Reticulum Stress
                    - Radiation Response
                    
                    🔬 **Cellular Processes**
                    - Cell Adhesion
                    - Organelle Organization
                    - Cytoskeletal Dynamics
                    - Vesicle Transport
                    - Autophagy
                    - Endocytosis and Exocytosis
                    - Cell Polarity Establishment
                    
                    🦠 **Disease Mechanisms**
                    - Neurodegeneration
                    - Autoimmune Mechanisms
                    - Viral Infection Response
                    - Fibrosis Pathways
                    - Inflammation-Driven Pathology
                    - Metabolic Syndrome
                    - Cardiovascular Disease Mechanisms
                    - Bacterial Infection Pathways
                    
                    
                    🧠 If uncertain, make a **biologically educated guess** based on the functional implication of the pathway.
                    
                    🛑 DO NOT use the exact pathway name as Sub_Class. Instead, deduce the mechanism it belongs to.
                    
                    Return format:
                    Main_Class: [class from the list above]  
                    Sub_Class: [most specific mechanism/function - choose from list or derive a new one only if essential]
                    
                    Pathway: {pathway_name}  
                    Source: {ontology}
                    
                    
                                    IMPORTANT: You must provide both Main_Class and Sub_Class. Never leave them empty.
                    
                                    Return format:
                                    Main_Class: [class from the list above]
                                    Sub_Class: [specific mechanism/process - be descriptive]
                  """
        
        client = openai.OpenAI()
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500
        )
        
        text = res.choices[0].message.content
        main_class = ""
        sub_class = ""
        
        # Parse the response more robustly
        for line in text.split("\n"):
            line = line.strip()
            if "Main_Class" in line or "main_class" in line.lower():
                if ":" in line:
                    main_class = line.split(":")[1].strip()
            elif "Sub_Class" in line or "sub_class" in line.lower():
                if ":" in line:
                    sub_class = line.split(":")[1].strip()
        
        # Clean up classifications
        main_class = re.sub(r'^\d+\.?\s*', '', main_class)
        sub_class = re.sub(r'^\d+\.?\s*', '', sub_class)
        
        # Ensure we have valid classifications
        if not main_class:
            main_class = "Cellular Processes"
        if not sub_class:
            sub_class = "Unknown Process"
        
        insert_into_cache(pathway_name, main_class, sub_class, ontology, 0.8)
        time.sleep(1)  # Rate limiting
        
        return main_class, sub_class, ontology
        
    except Exception as e:
        print(f"[red] Single LLM error for {pathway_name}: {e}[/red]")
        # Return fallback classification
        main_class = "Cellular Processes"
        sub_class = "Unknown Process"
        insert_into_cache(pathway_name, main_class, sub_class, ontology, 0.5)
        return main_class, sub_class, ontology

# === MAIN OPTIMIZED PROCESSING ===
def process_pathways_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """Main processing function with optimized LLM usage."""
    
    # Collect all unique pathways first
    all_pathways = set()
    pathway_to_row_mapping = defaultdict(list)
    
    for idx, row in df.iterrows():
        pathway_block = str(row['Pathway']).splitlines()
        for pathway in pathway_block:
            pathway = pathway.strip()
            if pathway:
                all_pathways.add(pathway)
                pathway_to_row_mapping[pathway].append(idx)
    
    print(f"[bold blue] Found {len(all_pathways)} unique pathways to classify[/bold blue]")
    
    # Pre-check cache for all pathways
    cached_results = {}
    uncached_pathways = []
    
    for pathway in all_pathways:
        cached_result = get_cached_pathway(pathway)
        if cached_result:
            cached_results[pathway] = cached_result
        else:
            uncached_pathways.append(pathway)
    
    print(f"[green] Found {len(cached_results)} pathways in cache[/green]")
    print(f"[yellow] Need to classify {len(uncached_pathways)} new pathways[/yellow]")
    
    # Process uncached pathways with correct priority
    kegg_candidates = []
    llm_candidates = []
    
    # Determine pathway types and route accordingly
    for pathway in uncached_pathways:
        # Determine if this is a KEGG pathway by checking associated DB_IDs
        is_kegg_pathway = False
        for idx in pathway_to_row_mapping[pathway]:
            db_sources = str(df.iloc[idx]['DB_ID']).split(",")
            for src in db_sources:
                src = src.strip().upper()
                if 'KEGG' in src or 'KEGG' in pathway.upper():
                    is_kegg_pathway = True
                    break
            if is_kegg_pathway:
                break
        
        if is_kegg_pathway:
            kegg_candidates.append(pathway)
        else:
            llm_candidates.append(pathway)
    
    print(f"[blue] KEGG pathways to classify: {len(kegg_candidates)}[/blue]")
    print(f"[magenta] Non-KEGG pathways to classify: {len(llm_candidates)}[/magenta]")
    
    # Process KEGG pathways first
    kegg_failed = []
    for pathway in kegg_candidates:
        print(f"[blue] Trying KEGG classification for: {pathway}[/blue]")
        kegg_result = get_kegg_classification(pathway)
        if kegg_result and kegg_result[0] and kegg_result[1]:  # Ensure both main and sub class
            cached_results[pathway] = kegg_result
            print(f"[green] KEGG classified: {pathway} -> {kegg_result[0]} | {kegg_result[1]}[/green]")
        else:
            print(f"[yellow] KEGG classification failed for: {pathway}, will use LLM[/yellow]")
            kegg_failed.append(pathway)
    
    # Add failed KEGG pathways to LLM candidates
    llm_candidates.extend(kegg_failed)
    
    print(f"[magenta] Total pathways for LLM classification: {len(llm_candidates)}[/magenta]")
    
    # Batch LLM classification for all remaining pathways
    if llm_candidates:
        # Group by ontology for better batching
        ontology_groups = defaultdict(list)
        for pathway in llm_candidates:
            # Determine ontology from original data
            ontology = "GO"  # default
            for idx in pathway_to_row_mapping[pathway]:
                db_sources = str(df.iloc[idx]['DB_ID']).split(",")
                for src in db_sources:
                    src = src.strip().upper()
                    if src in ["REACTOME", "GO_BP", "GO_MF", "GO_CC", "WIKIPATHWAY", "KEGG"]:
                        ontology = src
                        break
            ontology_groups[ontology].append(pathway)
        
        # Process each ontology group
        for ontology, pathways in ontology_groups.items():
            print(f"[magenta] Processing {len(pathways)} pathways from {ontology} with LLM[/magenta]")
            batch_data = [(pathway, ontology) for pathway in pathways]
            llm_results = batch_classify_pathways_llm(batch_data, batch_size=8)
            
            for pathway, (main_class, sub_class, source) in llm_results.items():
                if main_class and sub_class:  # Only accept if both are provided
                    cached_results[pathway] = (main_class, sub_class, source)
                    print(f"[green] LLM classified: {pathway} -> {main_class} | {sub_class}[/green]")
                else:
                    print(f"[red] LLM classification incomplete for: {pathway}[/red]")
                    # Provide fallback classification
                    cached_results[pathway] = ("Cellular Processes", "Unknown Process", source)
    
    # Ensure ALL pathways have classifications
    unclassified_count = 0
    for pathway in all_pathways:
        if pathway not in cached_results:
            print(f"[red] WARNING: Pathway not classified: {pathway}[/red]")
            # Provide fallback classification
            cached_results[pathway] = ("Cellular Processes", "Unknown Process", "FALLBACK")
            unclassified_count += 1
    
    if unclassified_count > 0:
        print(f"[yellow] Applied fallback classification to {unclassified_count} pathways[/yellow]")
    
    # Apply results to dataframe
    new_pathway = []
    new_sources = []
    new_main_classes = []
    new_sub_classes = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Applying classifications"):
        pathway_block = str(row['Pathway']).splitlines()
        db_id_raw = str(row['DB_ID'])
        db_sources = [x.strip().upper() for x in db_id_raw.split(",") if x.strip()]
        
        block_sources, block_main, block_sub = [], [], []
        
        for pathway in pathway_block:
            pathway = pathway.strip()
            if pathway and pathway in cached_results:
                main_class, sub_class, source = cached_results[pathway]
            else:
                # This should not happen now, but just in case
                main_class, sub_class, source = "Cellular Processes", "Unknown Process", "FALLBACK"
                print(f"[red] Emergency fallback for pathway: {pathway}[/red]")
            
            block_sources.append(", ".join(db_sources))
            block_main.append(main_class or "Cellular Processes")
            block_sub.append(sub_class or "Unknown Process")
        
        new_pathway.append("\n".join(pathway_block))
        new_sources.append("\n".join(block_sources))
        new_main_classes.append("\n".join(block_main))
        new_sub_classes.append("\n".join(block_sub))
    
    df['Ontology_Source'] = new_sources
    df['Main_Class'] = new_main_classes
    df['Sub_Class'] = new_sub_classes
    
    # Print final statistics
    total_classified = len([p for p in all_pathways if p in cached_results])
    print(f"[bold green] Classification complete: {total_classified}/{len(all_pathways)} pathways classified[/bold green]")
    
    return df


def categorize_pathways(input_csv: str) -> Path:
    logger.info(f"Categorizing pathways from {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} rows from {input_csv}")
    df = process_pathways_optimized(df)
    logger.info(f"Processed {len(df)} pathways")

    df.to_csv(input_csv, index=False)

    logger.info(f"Categorized pathways saved to {input_csv}")
    logger.info(f"Classification complete!")
    logger.info(f"Cached memory saved to {CATEGORIZER_CSV_PATH}")

    # Show cache statistics
    try:
        if Path(CATEGORIZER_CSV_PATH).exists():
            cache_df = pd.read_csv(CATEGORIZER_CSV_PATH)
            total_cached = len(cache_df)
            logger.info(f"Total pathways in cache: {total_cached}")
        else:
            logger.info("No cache file found")
    except Exception as e:
        logger.error(f"Error reading cache statistics: {e}")

    return input_csv
   
    