import pandas as pd
from openai import OpenAI
from agentic_ai_wf.config.config import OPENAI_API_KEY, DEFAULT_MODEL
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def analyze_disease_status(row, disease_name) -> str:
    #DISEASE_NAME = "Pancreatic Cancer" # can be made dynamic through global config if necessary
    """
    Determines whether the drug is used in pancreatic cancer treatment,
    based on drug name, pathway, and target mechanism.
    """
    drug_name = row["name"]
    pathway = row["pathway"]
    target_mechanism = row["target-mechanism"]

    if not drug_name or pd.isna(drug_name) or drug_name.strip() == "":
        return "No"

    prompt = f"""
    You are a clinical pharmacology and cancer biology expert.

    Given a list of drugs with their respective **Target Mechanism** and **Pathway**, evaluate the relevance of each drug in the context of **{disease_name}**.

    Drug Name: {drug_name}
    Pathway: {pathway}
    Target Mechanism: {target_mechanism}

    Based on clinical trials, pharmacologic classification, and given knowledge,
    is this drug relevant for treating pancreatic cancer?

    Respond with only one word: "Yes" or "No".
    """

    messages = [
        {"role": "system", "content": "You are a clinical assistant with expertise in oncology and pharmacology."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip()
        return "Yes" if "yes" in answer.lower() else "No"
    except Exception as e:
        print(f"❌ Error checking status for '{drug_name}': {e}")
        return "No"

def generate_clinical_note(drug_name: str, pathway: str, target_mechanism: str) -> str:
    """
    Generates a brief one-line clinical note about the drug’s mechanism and pathway relevance.
    """
    if pd.isna(drug_name) or drug_name.strip() == "":
        return ""

    prompt = f"""
    Provide a one-line clinical insight about the drug below, focusing on its pathway and mechanism of action:

    - Drug: {drug_name}
    - Pathway: {pathway}
    - Target Mechanism: {target_mechanism}

    Return just one very short (not more than 10 words) sentence regarding drug relevance with pathway and target-mechanism.
    """
    messages = [
        {"role": "system", "content": "You are a clinical summarization expert."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error generating note for '{drug_name}': {e}")
        return ""

def analyze_harmful_status(row) -> str:
    """
    Determines whether the drug is considered harmful based on its name,
    pathway, mechanism, and clinical note.
    """
    drug_name = row["name"]
    pathway = row["pathway"]
    target_mechanism = row["target-mechanism"]
    note = row.get("Notes", "")

    if not drug_name or pd.isna(drug_name) or drug_name.strip() == "":
        return "No"

    prompt = f"""
    You are a clinical pharmacologist.

    Analyze the potential harm or toxicity of the following drug using all the context provided.

    - Drug Name: {drug_name}
    - Pathway: {pathway}
    - Target Mechanism: {target_mechanism}
    - Clinical Note: {note}

    Determine whether this drug is known to be harmful or associated with serious risks, especially in clinical use. 
    Harmful drugs include those with severe adverse effects, narrow therapeutic windows, or high toxicity potential.

    Respond with only one word: "Yes" if it is dangerous or harmful, otherwise "No".
    """

    messages = [
        {"role": "system", "content": "You are a pharmacology safety expert."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip()
        return "Yes" if "yes" in answer.lower() else "No"
    except Exception as e:
        print(f"❌ Error checking harmful status for '{drug_name}': {e}")
        return "No"
    

def evaluate_drugs_for_pancreatic_cancer(main_df: pd.DataFrame, disease_name: str) -> pd.DataFrame:
    """
    Adds 'pan_status' and 'Notes' columns to the DataFrame based on drug name, pathway, and mechanism.
    """
    #main_df["pan_status"] = main_df["name"].apply(analyze_pancreatic_status)
    #main_df["pan_status"] = main_df.apply(analyze_disease_status, axis=1)
    main_df["pan_status"] = main_df.apply(analyze_disease_status, axis=1, args=(disease_name,))
    print("-disease analysis done.")
    main_df["Notes"] = main_df.apply(
        lambda row: generate_clinical_note(row["name"], row["pathway"], row["target-mechanism"]),
        axis=1
    )
    print("-drugs notes generated.")
    main_df["harmful_status"] = main_df.apply(analyze_harmful_status, axis=1)
    print("-drugs safety analysis done.")
    return main_df

def data_analysis(df, disease_name):
    df['target only'] = df['target'].str.extract(r'^([^[]+)\s*\[HSA', expand=False)
    # If no match was found (i.e., NaN), keep the original value
    df['target only'] = df['target only'].fillna(df['target'])
    df = df[["name", "pathway", "target-mechanism", "target only", "deg_match_status", "approved"]]

    updated_df = evaluate_drugs_for_pancreatic_cancer(df, disease_name)
    #updated_df.to_csv("processed_data/analyzed_data.csv",index=False)
    return updated_df