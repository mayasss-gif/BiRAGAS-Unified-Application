import requests

from .constants import FDA_API_LIMIT, FDA_API_URL


def get_drug_names_and_route(drug_name: str, limit: int = FDA_API_LIMIT) -> dict:
    url = (
        f"{FDA_API_URL}/drug/ndc.json"
        f"?search=brand_name:\"{drug_name}\"&limit={limit}"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {}
        result = resp.json()['results'][0]
    except (KeyError, IndexError, requests.RequestException):
        return {}

    return {
        'brand_name':             result.get('brand_name', 'Not available'),
        'generic_name':           result.get('generic_name', 'Not available'),
        'route_of_administration': ', '.join(result.get('route', ['Not available']))
    }