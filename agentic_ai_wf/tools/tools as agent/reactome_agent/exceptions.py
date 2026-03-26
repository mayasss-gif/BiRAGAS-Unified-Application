import time
from collections import deque
from typing import Set
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agents import function_tool
# from .reactome_models import ReactomeInput, ReactomeOutput
from reactome_models import ReactomeInput, ReactomeOutput

from logger import get_logger
logger = get_logger("reactome.tool")
logger.info("✅ Logger initialized in reactome_tool.py")

# Config
BASE_URL = "https://reactome.org/ContentService/data"
THROTTLE_DELAY = 0.1

# Session with retry
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True
)
session = requests.Session()
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

def fetch_json(url: str):
    """
    Perform a GET request to the specified URL and return the parsed JSON response.

    This function uses a retry-enabled HTTP session to handle transient network issues
    and automatically applies a short delay after successful requests to avoid rate limits.

    Args:
        url (str): The full URL to query for a JSON response.

    Returns:
        dict: The JSON-parsed response from the server.

    Raises:
        requests.exceptions.HTTPError: If the response status is not 2xx.
        requests.exceptions.RequestException: For network-related errors.
    """
    logger.info(f"Fetching JSON from: {url}")
    
    try:
        resp = session.get(url)
        resp.raise_for_status()
        data = resp.json()
        time.sleep(THROTTLE_DELAY)
        return data

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error while fetching {url}: {e.response.status_code} — {e.response.reason}")
        raise

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url}: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during fetch from {url}: {e}")
        raise RuntimeError(f"Unexpected error during API call: {e}") from e


def collect_subpathways(pid: str, visited: Set[str] = None):
    """
    Recursively collect all descendant subpathways for a given Reactome pathway ID.

    This function queries the Reactome API to explore the full hierarchy of nested subpathways
    originating from the input pathway ID. It avoids cycles by tracking visited pathways.

    Args:
        pid (str): The Reactome pathway ID (e.g., 'R-HSA-199420') to start traversal from.
        visited (Set[str], optional): A set of already-visited pathway IDs to prevent recursion loops.

    Returns:
        List[str]: A list of descendant Reactome pathway dbIds under the given parent.

    Raises:
        RuntimeError: If an unexpected error occurs during traversal.
    """
    if visited is None:
        visited = set()
    if pid in visited:
        return []
    visited.add(pid)

    try:
        children = fetch_json(f"{BASE_URL}/pathway/{pid}/pathways") or []
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"Pathway not found: {pid}")
            return []
        logger.error(f"HTTP error during subpathway collection for {pid}: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while fetching subpathways for {pid}: {e}")
        raise RuntimeError(f"Failed to fetch subpathways for {pid}") from e
    except Exception as e:
        logger.error(f"Unexpected error in collect_subpathways({pid}): {e}")
        raise RuntimeError(f"Unexpected error during subpathway recursion") from e

    ids = []
    for c in children:
        try:
            cid = c["dbId"]
            ids.append(cid)
            ids.extend(collect_subpathways(cid, visited))
        except KeyError:
            logger.warning(f"Missing 'dbId' in child entry: {c}")
        except Exception as e:
            logger.error(f"Error processing child pathway {c}: {e}")

    return ids

def get_drugs_for_pathway(pid: str):
    """
    Retrieve all drug entities associated with a given Reactome pathway ID.

    This function performs a full traversal of the given pathway and its subpathways to collect
    all participating physical entities. It expands any entity sets (e.g., EntitySet, CandidateSet)
    and filters out only those classified as drugs (e.g., ChemicalDrug, ProteinDrug).

    Args:
        pid (str): A Reactome pathway ID (e.g., 'R-HSA-109581').

    Returns:
        Set[str]: A deduplicated set of drug names associated with the given pathway and its subcomponents.

    Raises:
        RuntimeError: If an unexpected error occurs during traversal or API call.
    """
    try:
        all_pids = {pid} | set(collect_subpathways(pid))
        raw = []
        for p in all_pids:
            try:
                raw.extend(fetch_json(f"{BASE_URL}/participants/{p}/participatingPhysicalEntities"))
            except Exception as e:
                logger.warning(f"Failed to fetch participants for pathway {p}: {e}")

        queue = deque()
        plains = []
        for e in raw:
            if e.get("schemaClass") in ("EntitySet", "CandidateSet", "DefinedSet"):
                queue.append(e.get("dbId"))
            else:
                plains.append(e)

        seen_sets = set()
        while queue:
            sid = queue.popleft()
            if sid in seen_sets:
                continue
            seen_sets.add(sid)

            try:
                data = fetch_json(f"{BASE_URL}/query/{sid}")
            except Exception as e:
                logger.warning(f"Failed to fetch entity set {sid}: {e}")
                continue

            members = data.get("hasMember") or data.get("hasCandidate") or []
            for m in members:
                cls = m.get("schemaClass")
                if cls in ("EntitySet", "CandidateSet", "DefinedSet"):
                    queue.append(m.get("dbId"))
                else:
                    plains.append(m)

        records = []
        for e in plains:
            try:
                det = fetch_json(f"{BASE_URL}/query/{e['dbId']}")
                clazz = det.get("schemaClass")
                if clazz not in ("Drug", "ChemicalDrug", "ProteinDrug", "RNADrug"):
                    continue

                ref = det.get("referenceEntity") or {}
                ident = ref.get("identifier")
                if not ident:
                    continue

                raw_name = det.get("displayName", "")
                name = raw_name.split(" [", 1)[0]
                records.append(name)

            except Exception as e:
                logger.warning(f"Skipping entity {e.get('dbId', 'unknown')} due to error: {e}")
                continue

        logger.info(f"Found {len(records)} drug(s) for pathway {pid}")
        return set(records)

    except Exception as e:
        logger.error(f"Unexpected error during drug extraction for pathway {pid}: {e}")
        raise RuntimeError(f"Failed to extract drugs for pathway {pid}") from e

@function_tool
def extract_reactome(input: ReactomeInput) -> ReactomeOutput:
    """
    Tool to extract drug names associated with a Reactome pathway ID.

    Args:
        input (ReactomeInput): A Pydantic model containing a single field `pathway_id`.

    Returns:
        ReactomeOutput: A list of unique drug names linked to the given pathway.
    """
    logger.info(f"Extracting drugs for pathway ID: {input.pathway_id}")
    drugs = list(get_drugs_for_pathway(input.pathway_id))
    return ReactomeOutput(drugs=drugs)

# import time
# from collections import deque
# from typing import Set
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# from agents import function_tool
# # from .reactome_models import ReactomeInput, ReactomeOutput
# from reactome_models import ReactomeInput, ReactomeOutput

# from logger import get_logger
# logger = get_logger("reactome.tool")
# logger.info("✅ Logger initialized in reactome_tool.py")

# # Config
# BASE_URL = "https://reactome.org/ContentService/data"
# THROTTLE_DELAY = 0.1

# # Session with retry
# retry_strategy = Retry(
#     total=5,
#     backoff_factor=1,
#     status_forcelist=[429, 500, 502, 503, 504],
#     allowed_methods=["GET"],
#     respect_retry_after_header=True
# )
# session = requests.Session()
# adapter = HTTPAdapter(max_retries=retry_strategy)
# session.mount("https://", adapter)
# session.mount("http://", adapter)

# def fetch_json(url: str):
#     """
#     Perform a GET request to the specified URL and return the parsed JSON response.

#     This function uses a retry-enabled HTTP session to handle transient network issues
#     and automatically applies a short delay after successful requests to avoid rate limits.

#     Args:
#         url (str): The full URL to query for a JSON response.

#     Returns:
#         dict: The JSON-parsed response from the server.

#     Raises:
#         requests.exceptions.HTTPError: If the response status is not 2xx.
#         requests.exceptions.RequestException: For network-related errors.
#     """
#     logger.info(f"Fetching JSON from: {url}")
#     resp = session.get(url)

#     resp.raise_for_status()
#     data = resp.json()
#     time.sleep(THROTTLE_DELAY)
#     return data

# def collect_subpathways(pid: str, visited: Set[str] = None):
#     """
#     Recursively collect all descendant subpathways for a given Reactome pathway ID.

#     This function queries the Reactome API to explore the full hierarchy of nested subpathways
#     originating from the input pathway ID. It avoids cycles by tracking visited pathways.

#     Args:
#         pid (str): The Reactome pathway ID (e.g., 'R-HSA-199420') to start traversal from.
#         visited (Set[str], optional): A set of already-visited pathway IDs to prevent recursion loops.

#     Returns:
#         List[str]: A list of descendant Reactome pathway dbIds under the given parent.
    
#     Raises:
#         requests.exceptions.HTTPError: If an unexpected HTTP error occurs.
#     """
#     if visited is None:
#         visited = set()
#     if pid in visited:
#         return []
#     visited.add(pid)

#     try:
#         children = fetch_json(f"{BASE_URL}/pathway/{pid}/pathways") or []
#     except requests.exceptions.HTTPError as e:
#         if e.response.status_code == 404:
#             return []
#         raise

#     ids = []
#     for c in children:
#         cid = c["dbId"]
#         ids.append(cid)
#         ids.extend(collect_subpathways(cid, visited))
#     return ids

# def get_drugs_for_pathway(pid: str):
#     """
#     Retrieve all drug entities associated with a given Reactome pathway ID.

#     This function performs a full traversal of the given pathway and its subpathways to collect
#     all participating physical entities. It expands any entity sets (e.g., EntitySet, CandidateSet)
#     and filters out only those classified as drugs (e.g., ChemicalDrug, ProteinDrug).

#     Args:
#         pid (str): A Reactome pathway ID (e.g., 'R-HSA-109581').

#     Returns:
#         Set[str]: A deduplicated set of drug names associated with the given pathway and its subcomponents.

#     Raises:
#         requests.exceptions.HTTPError: If API responses fail for any of the queried entities.
#         KeyError: If unexpected schema is returned from the Reactome API.
#     """
#     all_pids = {pid} | set(collect_subpathways(pid))
#     raw = []
#     for p in all_pids:
#         raw.extend(fetch_json(f"{BASE_URL}/participants/{p}/participatingPhysicalEntities"))

#     queue = deque()
#     plains = []
#     for e in raw:
#         if e["schemaClass"] in ("EntitySet", "CandidateSet", "DefinedSet"):
#             queue.append(e["dbId"])
#         else:
#             plains.append(e)

#     seen_sets = set()
#     while queue:
#         sid = queue.popleft()
#         if sid in seen_sets:
#             continue
#         seen_sets.add(sid)

#         data = fetch_json(f"{BASE_URL}/query/{sid}")
#         members = data.get("hasMember") or data.get("hasCandidate") or []
#         for m in members:
#             cls = m["schemaClass"]
#             if cls in ("EntitySet", "CandidateSet", "DefinedSet"):
#                 queue.append(m["dbId"])
#             else:
#                 plains.append(m)

#     records = []
#     for e in plains:
#         det = fetch_json(f"{BASE_URL}/query/{e['dbId']}")
#         clazz = det.get("schemaClass")
#         if clazz not in ("Drug", "ChemicalDrug", "ProteinDrug", "RNADrug"):
#             continue

#         ref = det.get("referenceEntity") or {}
#         ident = ref.get("identifier")
#         if not ident:
#             continue

#         raw_name = det.get("displayName", "")
#         name = raw_name.split(" [", 1)[0]
#         records.append(name)
#         logger.info(f"Found {len(records)} drug(s) for pathway {pid}")
#     return set(records)


# @function_tool
# def extract_reactome(input: ReactomeInput) -> ReactomeOutput:
#     """
#     Tool to extract drug names associated with a Reactome pathway ID.

#     Args:
#         input (ReactomeInput): A Pydantic model containing a single field `pathway_id`.

#     Returns:
#         ReactomeOutput: A list of unique drug names linked to the given pathway.
#     """
#     logger.info(f"Extracting drugs for pathway ID: {input.pathway_id}")
#     drugs = list(get_drugs_for_pathway(input.pathway_id))
#     return ReactomeOutput(drugs=drugs)

