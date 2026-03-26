import pandas as pd
import os

from .deg_filter_helper import filter_drugs_by_degs
from agents import function_tool

from ..models.drugs_models import EnrichedDrugMatch, EnrichedDrugMatchInput
from ..models.drugs_models import EnrichedDrugDegOutput, EnrichedDrugDeg
from ..config import global_config


@function_tool
def deg_filter(drugs: EnrichedDrugMatchInput) -> EnrichedDrugDegOutput:
    """
    Filter drugs by DEGs.

    Args:
        drugs (EnrichedDrugMatchInput): The drugs to filter.
        patient_id (str): The patient ID.

    Returns:
        EnrichedDrugDegOutput: The filtered drugs.
    """
    try:

        print("deg filter executing...")
        # Step 1: Load data (you already did this)
        drug_dicts = [drug.model_dump() for drug in drugs.drugs]

        drug_df = pd.DataFrame(drug_dicts)
        # drug_df.to_csv("patient_deg_original_input.csv", index=False)

        drug_df_org = drug_df.copy()

        # degs_df = pd.read_csv("tools/patient_degs.csv")

        degs_path = global_config.patient_deg_csv_path
        print("degs_path: ", degs_path)
        degs_df = pd.read_csv(degs_path)
        degs_df['Normalized_Gene'] = degs_df['Gene']
        # Step 2: Filter using clean function
        filtered_df = filter_drugs_by_degs(drug_df, degs_df)

        # ✅ Handle the result outside the function
        if filtered_df.empty:
            print("⚠️ No drug targets matched any DEGs.")
        else:
            print("✅ Found matching drugs. Saving results...")
            # filtered_df.to_csv("matched_drugs_with_DEG_targets.csv", index=False)

            # Optional: show a few matched entries
            print("\n📋 Top Matched Drugs:")
            print(filtered_df[["name", "target"]].head())

        # Step 3: Save output
        filtered_df['deg_match_status'] = True
        filtered_df.drop(
            ['LLM_Match_Reason', 'matching_status'], axis=1, inplace=True)
        drug_df = drug_df_org.merge(
            filtered_df[['drug_id', 'deg_match_status']],
            on='drug_id',
            how='left'
        )
        # drug_df.to_csv("drug_df2.csv",index=False)
        # filtered_df.to_csv("matched_drugs_with_DEG_targets.csv", index=False)
        # drug_df.to_csv("patient_deg_matched_df.csv", index=False)
        drug_df['deg_match_status'] = drug_df['deg_match_status'].fillna(
            False).astype(bool)
        # drug_df.to_csv("patient_deg_matched_df-filled.csv", index=False)

        enriched_drugs = [EnrichedDrugDeg(
            **row.dropna().to_dict()) for _, row in drug_df.iterrows()]
        return EnrichedDrugDegOutput(drugs=enriched_drugs)

        print("✅ Filtered drug results saved to: matched_drugs_with_DEG_targets.csv")

        # Optional: preview top hits
        filtered_df[["name", "target"]].head()

    except Exception as exp:
        print("Exception Occurs - deg filter mian: ", exp)
