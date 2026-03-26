import requests
import pandas as pd

from .constants import FDA_API_LIMIT, FDA_API_URL


def get_adverse_reactions(drug_name: str, limit: int = FDA_API_LIMIT) -> str:
    url = (
        f"{FDA_API_URL}/drug/event.json"
        f"?search=patient.drug.medicinalproduct:\"{drug_name}\"&limit={limit}"
    )
    try:
        resp = requests.get(url, timeout=10)
        # print(resp.json())
        if resp.status_code != 200:
            return 'Not available'
        data = resp.json().get('results', [])
    except (ValueError, requests.RequestException):
        return 'Not available'
    reactions = []
    for entry in data:
        for r in entry.get('patient', {}).get('reaction', []):
            reactions.append(r.get('reactionmeddrapt', 'Unknown'))
    return ', '.join(sorted(set(reactions))) if reactions else 'Not available'

if __name__ == "__main__":
    # read excel file and update with adverse reactions
    import logging
    import os
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Get the current working directory and find the Excel file
    current_dir = Path.cwd()
    logger.info(f"Current working directory: {current_dir}")
    
    # Look for the Excel file in common locations
    possible_paths = [
        "drug_details.xlsx",
        "../drug_details.xlsx", 
        "../../drug_details.xlsx",
        "../../../drug_details.xlsx",
        str(current_dir / "drug_details.xlsx"),
        str(current_dir.parent / "drug_details.xlsx"),
        str(current_dir.parent.parent / "drug_details.xlsx"),
        str(current_dir.parent.parent.parent / "drug_details.xlsx")
    ]
    
    excel_file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            excel_file_path = path
            logger.info(f"Found Excel file at: {excel_file_path}")
            break
    
    if not excel_file_path:
        logger.error("Could not find drug_details.xlsx file!")
        logger.info("Searched in:")
        for path in possible_paths:
            logger.info(f"  - {path}")
        exit(1)

    # Read the Excel file
    logger.info(f"Reading Excel file from: {excel_file_path}")
    df = pd.read_excel(excel_file_path)
    logger.info(f"Loaded {len(df)} rows from Excel file")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Add adverse_reactions column if it doesn't exist
    if 'fda_adverse_reactions' not in df.columns:
        df['fda_adverse_reactions'] = ''
        logger.info("Added 'fda_adverse_reactions' column to the dataframe")
    else:
        logger.info("'fda_adverse_reactions' column already exists")
    
    # Process each drug and update/save after each
    for index, row in df.iterrows():
        drug_name = row['drug_name']
        logger.info(f"Processing drug: {drug_name} (row {index + 1})")
        
        reactions = get_adverse_reactions(drug_name)
        df.at[index, 'fda_adverse_reactions'] = reactions
        
        logger.info(f"Updated row {index + 1} with reactions")
        
        if reactions == 'Not available':
            logger.warning(f"No reactions found for {drug_name}")
        else:
            logger.info(f"++++++Reactions found for {drug_name}++++++")
        
        # Save after each drug
        try:
            df.to_excel(excel_file_path, index=False)
            logger.info(f"Saved {excel_file_path} after processing {drug_name} \n ----------- \n")
            
                
        except Exception as e:
            logger.error(f"Error saving file after {drug_name}: {str(e)}")
            backup_filename = f"drug_details_with_reactions_{drug_name}.xlsx"
            df.to_excel(backup_filename, index=False)
            logger.info(f"Saved backup file as {backup_filename} after {drug_name}")