import json
import pickle
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth
from .config import GENE_CARD_SCORES_PATH
from agentic_ai_wf.neo4j_integration.utils import genecards_scorer

from .helpers import logger


def slug(txt: str) -> str:
    """
    Convert a string to a slug suitable for use as a key or filename.
    Only lowercase alphanumeric characters and underscores are retained.
    Args:
        txt (str): Input string.
    Returns:
        str: Slugified string.
    """
    if not isinstance(txt, str):
        raise TypeError("Input to slug() must be a string.")
    return re.sub(r'[^0-9a-z]+', '_', txt.lower()).strip('_')


def gene_card_score(disease_name: str, n: int = 4000, pkl_path: str = GENE_CARD_SCORES_PATH) -> dict:
    """
    Retrieve top N gene scores for a given disease from a pickle file.
    Args:
        disease_name (str): Name of the disease.
        n (int, optional): Number of top genes to return. Defaults to 20.
        pkl_path (str, optional): Path to the pickle file. Defaults to dataset path.
    Returns:
        dict: Dictionary of gene symbol to score, sorted by score descending.
    Raises:
        FileNotFoundError: If the pickle file does not exist.
        TypeError: If disease_name is not a string or n is not an int.
    """
    if not isinstance(disease_name, str):
        raise TypeError("disease_name must be a string.")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")

    try:
        gene_score_dict = genecards_scorer(disease_name)
        if gene_score_dict:
            logger.info(
                f"Fetched {len(gene_score_dict)} GeneCards scores for {disease_name} using Neo4j Graph Database.")
            return gene_score_dict
        else:
            logger.error(
                f"No GeneCards scores found for {disease_name} using Neo4j")
            key = slug(disease_name)
            with open(pkl_path, "rb") as f:
                disease_to_gene = pickle.load(f)
                bag = disease_to_gene.get(key, {})
                if not isinstance(bag, dict):
                    logger.error(f"Data for {key} is not a dictionary.")
                    return {}
                return dict(sorted(bag.items(), key=lambda x: -x[1])[:n])
    except FileNotFoundError:
        logger.error(f"Pickle file not found: {pkl_path}")
        raise
    except Exception as e:
        logger.error(
            f"Error fetching GeneCards scores for {disease_name}: {e}")
        return {}


def setup_stealth_driver():
    """
    Setup a stealth driver for GeneCards.
    This is used to avoid detection by GeneCards.
    Returns:
        webdriver.Chrome: A Chrome driver with stealth capabilities.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=chrome_options)

    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
            )
    return driver


def fetch_genecards_scores(disease: str) -> dict[str, float]:
    """
    Fetch GeneCards relevance scores for a given disease.

    Args:
        disease (str): Disease name to search on GeneCards.

    Returns:
        dict[str, float]: Dictionary mapping gene symbol to relevance score.
    """
    logger.info(f"Fetching GeneCards scores for {disease}")

    driver = setup_stealth_driver()
    url = f"https://www.genecards.org/Search/Keyword?queryString={disease}&pageSize=4000&startPage=0&sort=Score&sortDir=Descending"
    driver.get(url)

    gene_score_dict = {}
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "#searchResults tbody tr"))
        )
        rows = driver.find_elements(By.CSS_SELECTOR, "#searchResults tbody tr")

        for row in rows:
            try:
                gene = row.find_element(
                    By.CSS_SELECTOR, "td.gc-gene-symbol a").text.strip()
                score = float(row.find_element(
                    By.CSS_SELECTOR, "td.score-col").text.strip())
                gene_score_dict[gene] = score
            except Exception:
                continue

    finally:
        driver.quit()

    logger.info(
        f"Fetched {len(gene_score_dict)} GeneCards scores for {disease}")

    return gene_score_dict

# import time
# start_time = time.time()
# print(fetch_genecards_scores("Cervical Cancer"))
# end_time = time.time()
# print(f"Time taken: {end_time - start_time} seconds")
