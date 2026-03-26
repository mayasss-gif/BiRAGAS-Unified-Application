# === RUN PIPELINE ===
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import tiktoken
import json
from ..models.drugs_models import EnrichedDrugInput, EnrichedDrug
from ..models.drugs_models import EnrichedDrugMatch,EnrichedDrugMatchOutput
from .patient_condition_drug_prior_helper import extract_diagnosis_summary
from .patient_condition_drug_prior_helper import find_relevant_drugs_with_llm
from agents import function_tool
from typing import Dict, List, Optional
pd.set_option('future.no_silent_downcasting', True)
from ..config import global_config

@function_tool
def patient_condition_drug_prior(drugs: EnrichedDrugInput)-> EnrichedDrugMatchOutput:
    """ select suitable drugs data based on patient condition """
    try:
        print("patient condition prioritization....")
        #pdf_path = "tools/patient_report.pdf"
        
        pdf_path = global_config.patient_report_pdf_path
        #print("pdf_path: ", pdf_path)
        retrieval_file = "/home/laila/Documents/Drug_data_extraction_agentic_AI/Drug_data_retrieval_Agent/Prioritizing_drug/output_drug_retireval_agent.csv"

        # Step 1: Get priority disease
        summary = extract_diagnosis_summary(pdf_path)
        print("\n--- Diagnosis Summary ---\n")
        print(summary)
        #print("summary type: ", type(summary))

        if summary is None:
            print("❌ No diagnosis summary generated. Skipping drug matching.")
            exit(1)  # or: import sys; sys.exit(1)
        #print("**************************")
        #print("summary: ", summary)
        #print("*************************")
        try:
            summary = summary.strip("`").strip()
            if summary.startswith("json"):
                summary = summary[4:].strip()
            parsed_summary = json.loads(summary)
            #print("parsed_summary type: ", type(parsed_summary))
            #print("parsed_summary: ", parsed_summary)
            priority_disease = parsed_summary.get("Priority")
        except Exception as e:
            print("Error parsing JSON summary:", e)
            priority_disease = None

        print("priority_disease: ", priority_disease)
        # Step 2: Load data retrieval CSV and match
        if priority_disease:

            drug_dicts = [drug.dict() for drug in drugs.drugs]
            df = pd.DataFrame(drug_dicts)

            matches_df = find_relevant_drugs_with_llm(priority_disease, df)
            if matches_df.shape[0] == 0:
                columns = [
                    "drug_id", "name", "pathway_id", "pathway_name", "drug_class",
                    "target", "efficacy", "brite", "approved", "adv_reactions",
                    "route", "LLM_Match_Reason", "matching_status"
                ]

                # Create an empty DataFrame
                matches_df = pd.DataFrame(columns=columns)
                            
            else:
                matches_df["matching_status"] = True
                print("matches_df.shape:" , matches_df.shape)
                #matches_df.to_csv("matches_test.csv", index=False)
            
            df = df.merge(
                matches_df[['drug_id', 'LLM_Match_Reason', 'matching_status']],
                on='drug_id',
                how='left'
            )
            #print("mateched_df: ", matches_df)

            #matches_df.to_csv("patient_condition_matched_drugs_output.csv", index=False)
            #df.to_csv("patient_condition_df.csv", index=False)
            #print("check0")
            
            df['matching_status'] = df['matching_status'].fillna(False).astype(bool)
            df['LLM_Match_Reason'] = df['LLM_Match_Reason'].fillna("not matched").astype(str)

            if len(df) >5:
                df = df.head(5)
            #df.to_csv("patient_condition_df-filled.csv",index=False)
            drug_matches = [EnrichedDrugMatch(**row) for row in df.to_dict(orient="records")]
    
            print("patient - condition, returning output...")
            return EnrichedDrugMatchOutput(drugs=drug_matches)

            print("\n✅ Relevant drugs saved to: matched_drugs_output.csv")
            # ✅ Show drug names in console
            if not matches_df.empty:
                print("\n📋 Matched Drug Names:")
                for drug in matches_df["name"]:
                    print(f" - {drug}")
            else:
                print("⚠️ No matching drugs found.")
        else:
            print("⚠️ No Priority disease found, skipping drug matching.")
    except Exception as exp:
        print("Excpetion Occurs - patieint condition durg prior main: ", str(exp))