import requests

from .constants import FDA_API_LIMIT, FDA_API_URL
from .text_processor import TextProcessor


def get_drug_label_info(drug_name: str, limit: int = FDA_API_LIMIT) -> dict:
    url = (
        f"{FDA_API_URL}/drug/label.json"
        f"?search=openfda.brand_name:\"{drug_name}\"&limit={limit}"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {}
        r = resp.json()['results'][0]
    except (KeyError, IndexError, requests.RequestException):
        return {}

    # Initialize text processor
    text_processor = TextProcessor()

    raw = {
        'indications_and_usage':     r.get('indications_and_usage',       ['Not available'])[0],
        'dosage_and_administration': r.get('dosage_and_administration',   ['Not available'])[0],
        'mechanism_of_action':       text_processor.extract_mechanism_of_action(r.get('clinical_pharmacology', [''])[0]),
        'warnings_and_precautions':   r.get('warnings_and_precautions',     ['Not available'])[0],
        'contraindications':         r.get('contraindications',            ['Not available'])[0],
        'boxed_warning':             r.get('boxed_warning',                ['Not available'])[0],
        'patient_counseling_info':   r.get('information_for_patients',     ['Not available'])[0],
    }

    # Clean all text fields using the text processor
    for k, v in raw.items():
        raw[k] = text_processor.clean_drug_label_text(v)

    return raw