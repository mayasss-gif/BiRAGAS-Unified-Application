import requests

from .constants import FDA_API_URL

def query_openfda(drug_name, field_type):
    """Helper to query OpenFDA with brand or generic name."""
    base_url = f"{FDA_API_URL}/drug/drugsfda.json"
    headers = {"User-Agent": "FDA-Checker"}

    if field_type == "brand":
        query = f'search=products.brand_name:"{drug_name}"'
    else:
        query = f'search=openfda.generic_name:"{drug_name}"'

    url = f"{base_url}?{query}&limit=1"

    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        # print(data)
        if 'results' in data and len(data['results']) > 0:
            return data['results'][0]  # Approved
        else:
            return None  # Not found
    except Exception as e:
        print(f"API error for {drug_name} ({field_type}): {e}")
        return None  # Error

if __name__ == "__main__":
    print(query_openfda("ibuprofen", "brand"))