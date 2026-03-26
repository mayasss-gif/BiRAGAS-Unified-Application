
import requests
from dotenv import load_dotenv
load_dotenv()
from decouple import config


NEO4J_API_KEY = config("NEO4J_API_KEY")
if not NEO4J_API_KEY:
    raise ValueError("NEO4J_API_KEY is not set in environment variables")



API_URL = config("API_BASE_URL", default="https://dev-agent-admin.f420.ai")

def get_disease_scores(disease_name: str) -> dict:
    """
    Call the GeneCardScore API with authentication.
    
    Args:
        disease_name (str): Name of the disease to get gene card scores for
        api_key (str): NEO4J_API_KEY for authentication
        base_url (str): Base URL of the Django application (default: https://dev-agent-admin.f420.ai)
    
    Returns:
        dict: API response containing gene card scores or error information
    """
    
    
    # API endpoint
    url = f"{API_URL}/analysisapp/api/neo4j/gene-card-score/"
    
    # Headers with API key authentication
    headers = {
        'X-NEO4J-API-KEY': NEO4J_API_KEY,
        'Content-Type': 'application/json'
    }
    
    # Query parameters
    params = {
        'disease_name': disease_name
    }
    
    try:
        # Make the API request
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        # Check if request was successful
        if response.status_code == 200:
            return {
                'status': 'success',
                'data': response.json(),
                'status_code': response.status_code
            }
        elif response.status_code == 400:
            return {
                'status': 'error',
                'message': 'Bad request - disease name is required',
                'data': response.json(),
                'status_code': response.status_code
            }
        elif response.status_code == 403:
            return {
                'status': 'error',
                'message': 'Forbidden - invalid or missing API key',
                'status_code': response.status_code
            }
        else:
            return {
                'status': 'error',
                'message': f'HTTP {response.status_code} error',
                'status_code': response.status_code,
                'data': response.text if response.text else None
            }
            
    except requests.exceptions.Timeout:
        return {
            'status': 'error',
            'message': 'Request timeout',
            'status_code': None
        }
    except requests.exceptions.ConnectionError:
        return {
            'status': 'error',
            'message': 'Connection error - check if the server is running',
            'status_code': None
        }
    except requests.exceptions.RequestException as e:
        return {
            'status': 'error',
            'message': f'Request failed: {str(e)}',
            'status_code': None
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Unexpected error: {str(e)}',
            'status_code': None
        }


# OUTPUT 
# {
#     "status": "success",
#     "data": {
#         "gene_symbol": ["GENE1", "GENE2", "GENE3"],
#         "gene_card_score": [0.1, 0.2, 0.3],
#         "disorder_score": [1.3, 2.2, 3.0],
#         "disorder_type": ["direct", "direct", "inferred"],
#     },
#     "status_code": 200
# }


# results = get_disease_scores("Diabetes Mellitus")
# print(len(results['data']['data']['gene_symbol']))
