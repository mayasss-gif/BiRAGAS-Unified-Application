import requests
import pandas as pd
import re
import sys

API_LIMIT = 1
# ---------- HEADER-STRIP FUNCTION ----------
def strip_header_labels(text: str) -> str:
    """
    Remove leading numeric section headers:
      1 INDICATIONS AND/|& USAGE, 2 DOSAGE AND/|& ADMINISTRATION, etc.
    """
    pattern = (
        r'^\s*(?:\d+\.?\d*\s*)?'
        r'(?:'
            r'INDICATIONS\s*(?:AND|&)\s*USAGE|'
            r'DOSAGE\s*(?:AND|&)\s*ADMINISTRATION|'
            r'CONTRAINDICATIONS|'
            r'WARNING|'
            r'PATIENT\s*COUNSELING\s*INFORMATION'
        r')\b[\.:]?\s*'
    )
    return re.sub(pattern, '', text, flags=re.IGNORECASE)

# ---------- LABEL EXTRACTION ----------
def extract_mechanism_of_action(pharmacology_text: str) -> str:
    match = re.search(
        r'12\.1\s*Mechanism of Action(.*?)(12\.|\Z)',
        pharmacology_text,
        re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else "Not available"

def get_drug_label_info(drug_name: str, limit: int = API_LIMIT) -> dict:
    url = (
        "https://api.fda.gov/drug/label.json"
        f"?search=openfda.brand_name:\"{drug_name}\"&limit={limit}"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    try:
        r = resp.json()['results'][0]
    except (KeyError, IndexError):
        return None

    raw = {
        'Indications and Usage':     r.get('indications_and_usage',       ['Not available'])[0],
        'Dosage and Administration': r.get('dosage_and_administration',   ['Not available'])[0],
        'Mechanism of Action':       extract_mechanism_of_action(r.get('clinical_pharmacology', [''])[0]),
        'Warnings and Precautions':   r.get('warnings_and_precautions',     ['Not available'])[0],
        'Contraindications':         r.get('contraindications',            ['Not available'])[0],
        'Boxed Warning':             r.get('boxed_warning',                ['Not available'])[0],
        'Patient Counseling Info':   r.get('information_for_patients',     ['Not available'])[0],
    }

    for k, v in raw.items():
        # Strip section headers, bullets, and trim
        cleaned = strip_header_labels(v)
        cleaned = cleaned.replace('•', '').strip()
        # Treat empty, 'none', or similar as not available
        lower = cleaned.lower()
        if (not cleaned
            or lower.startswith('none')
            or lower in ['na', 'not applicable', 'not available']):
            raw[k] = "Not available"
        else:
            raw[k] = cleaned

    return raw

# ---------- NDC DATA ----------
def get_drug_names_and_route(drug_name: str, limit: int = API_LIMIT) -> dict:
    url = (
        "https://api.fda.gov/drug/ndc.json"
        f"?search=brand_name:\"{drug_name}\"&limit={limit}"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    try:
        result = resp.json()['results'][0]
    except (KeyError, IndexError):
        return None

    return {
        'Brand Name':             result.get('brand_name', 'Not available'),
        'Generic Name':           result.get('generic_name', 'Not available'),
        'Route of Administration': ', '.join(result.get('route', ['Not available']))
    }

# ---------- ADVERSE REACTIONS ----------
def get_adverse_reactions(drug_name: str, limit: int = API_LIMIT) -> str:
    url = (
        "https://api.fda.gov/drug/event.json"
        f"?search=patient.drug.medicinalproduct:\"{drug_name}\"&limit={limit}"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        return 'Not available'
    try:
        data = resp.json().get('results', [])
    except ValueError:
        return 'Not available'
    reactions = []
    for entry in data:
        for r in entry.get('patient', {}).get('reaction', []):
            reactions.append(r.get('reactionmeddrapt', 'Unknown'))
    return ', '.join(sorted(set(reactions))) if reactions else 'Not available'

def find_drug_info_openfda(drugs_names):
    try:
        all_drug_data = []
        for drug_name in drugs_names:
            drug_name = re.sub(r"\s*\(.*?\)", "", drug_name).strip().lower()
            #drug = 'Tranylcypromine'
            drug = drug_name
            print(f"Processing: {drug}")
            label_info = get_drug_label_info(drug)
            ndc_info   = get_drug_names_and_route(drug)
            reactions  = get_adverse_reactions(drug)
            #label_info = 0  # comment/uncomment,  the above three lines as well for dummies
            if label_info and ndc_info:
                combined = {'Drug Queried': drug}
                combined.update(ndc_info)
                combined.update(label_info)
                combined['Adverse Reactions'] = reactions
                all_drug_data.append(combined)
            else:
                print(f"⚠️ Data not found for {drug}")
                combined = {
                'Drug Queried': drug,
                'Adverse Reactions': 'not-found',
                'Route of Administration': 'not-found'
                }
                all_drug_data.append(combined)
            # Export to CSV (include header row)

        df_out = pd.DataFrame(all_drug_data)

        #print("df_Out: ", df_out)
        df = df_out[['Drug Queried', 'Adverse Reactions', 'Route of Administration']]
        status = 1
        #print("*****************************************")
        #print("df.shape: ", df.shape)
        return df, status

    except Exception as exp:
        print("openfda Exception(openfda helper tool): ", exp)
        status = 0
        return "no information avaialble on openfda", status
