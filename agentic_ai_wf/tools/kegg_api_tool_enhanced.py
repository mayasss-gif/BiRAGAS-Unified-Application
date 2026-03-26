#!/usr/bin/env python3
import sys
import re
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup

KEGG_BASE = "https://rest.kegg.jp"
session = requests.Session()

# Throttle settings to avoid KEGG rate limits
REQUEST_INTERVAL = 0.5  # ~2 requests/sec
MAX_RETRIES = 3

# Simple in-memory cache for DDI HTML to avoid duplicate fetches
_ddi_cache = {}

def kegg_get(url):
    """
    Wrapper around requests.get with rate limiting and retry on 429.
    """
    retries = 0
    while True:
        resp = session.get(url)
        if resp.status_code == 429:
            wait = int(resp.headers.get('Retry-After', 1))
            time.sleep(wait)
            retries+= 1 # Base URL for KEGG REST API 
            if retries >= MAX_RETRIES:
                break
            continue
        break
    time.sleep(REQUEST_INTERVAL + random.uniform(0, 0.2))
    return resp


def get_pathway_name(pathway_id):
    pid = pathway_id.replace('path:', '')
    url = f"{KEGG_BASE}/get/path:{pid}"
    resp = kegg_get(url)
    if not resp.ok:
        return "Not found"
    match = re.search(r'NAME\s+(.+)', resp.text)
    return match.group(1).strip() if match else "Not found"


def get_drugs_for_pathway(pathway_id):
    pid = pathway_id.replace('path:', '')
    url = f"{KEGG_BASE}/get/path:{pid}"
    resp = kegg_get(url)
    if not resp.ok:
        return set()
    m = re.search(r'^DRUG\s+(.+?)^\S', resp.text, flags=re.M|re.S)
    return set(re.findall(r'\b(D\d+)\b', m.group(1))) if m else set()


def extract_info(kegg_text):
    lines = kegg_text.split('\n')
    info = {}
    current_label = None
    brite_lines = []
    efficacy_lines = []

    for line in lines:
        if line.startswith('NAME'):
            info['Name'] = line.split('NAME',1)[1].strip() or "Not found"
        elif line.startswith('CLASS'):
            info['Class'] = line.split('CLASS',1)[1].strip() or "Not found"
        elif line.startswith('BRITE'):
            current_label = 'BRITE'
            content = line.split('BRITE',1)[1].strip()
            if content: brite_lines.append(content)
        elif line.startswith('EFFICACY'):
            current_label = 'EFFICACY'
            content = line.split('EFFICACY',1)[1].strip()
            if content: efficacy_lines.append(content)
        elif current_label == 'BRITE' and line.startswith(' '):
            brite_lines.append(line.strip())
        elif current_label == 'EFFICACY' and line.startswith(' '):
            efficacy_lines.append(line.strip())
        else:
            current_label = None

    # Combine BRITE and determine approval
    approval_keywords = ['drug approvals', 'Pharmacopoeia']
    if brite_lines:
        full_brite = '\n'.join(brite_lines)
        info['Brite'] = full_brite
        # Set True if any keyword found, else 'Not found'
        if any(kw.lower() in full_brite.lower() for kw in approval_keywords):
            info['Approved'] = True
        else:
            info['Approved'] = 'Not found'
    else:
        info['Brite'] = 'Not found'
        info['Approved'] = 'Not found'


    info['Efficacy'] = '\n'.join(efficacy_lines) if efficacy_lines else 'Not found'
    return info

    # EFFICACY
    if efficacy_lines:
        info['Efficacy'] = '\n'.join(efficacy_lines)
    else:
        info['Efficacy'] = 'Not found'

    return info


def get_data_from_kegg(entry_id):
    """
    Fetch raw HTML from the KEGG DDI page for a given drug.
    """
    if entry_id in _ddi_cache:
        return _ddi_cache[entry_id]
    url = f"https://www.kegg.jp/kegg-bin/ddi_list?drug={entry_id}"
    resp = kegg_get(url)
    html = resp.text if resp.ok else ''
    _ddi_cache[entry_id] = html
    return html


def parse_ddi_html(html):
    """
    Parse the DDI HTML and return a list of strings:
    'DrugID: DrugName (InteractionType)',
    or ['Not found'] if empty.
    """
    ddi_list = []
    soup = BeautifulSoup(html, 'html.parser')
    for row in soup.find_all('tr', valign='top'):
        cols = row.find_all('td')
        if len(cols) < 3:
            continue
        drug_id = cols[0].get_text(strip=True) or 'Not found'
        drug_name = cols[1].get_text(strip=True) or 'Not found'
        interaction_type = cols[2].get_text(strip=True).replace('(P) ', '') or 'Not found'
        ddi_list.append(f"{drug_id}: {drug_name} ({interaction_type})")
    return ddi_list if ddi_list else ['Not found']


def get_drug_info(drug_id):
    """
    Fetch detailed drug info and DDI list for a KEGG drug.
    """
    url = f"{KEGG_BASE}/get/{drug_id}"
    resp = kegg_get(url)
    info = {
        'Drug ID': drug_id,
        'Drug name': 'Not found',
        'Class': 'Not found',
        'Target': ['Not found'],
        'Efficacy': 'Not found',
        'Brite': 'Not found',
        'Approved': False,
        'DDI': 'Not found'
    }
    if not resp.ok:
        print(" - Drug info found in KEGG, Extracting and Formatting the Data")
        return info

    ktext = resp.text
    di = extract_info(ktext)
    # update with extracted, defaulting missing
    info.update({
        'Drug name': di.get('Name', 'Not found'),
        'Class': di.get('Class', 'Not found'),
        'Brite': di.get('Brite', 'Not found'),
        'Approved': di.get('Approved', False),
        'Efficacy': di.get('Efficacy', 'Not found')
    })
    # TARGET section
    targets = []
    in_t = False
    for line in ktext.splitlines():
        if line.startswith('TARGET'):
            in_t = True
            targets.append(' '.join(line.split()[1:]).strip() or 'Not found')
        elif in_t and line.startswith(' '):
            targets.append(line.strip() or 'Not found')
        elif in_t:
            break
    info['Target'] = targets or ['Not found']

    # DDI
    html = get_data_from_kegg(drug_id)
    ddi_list = parse_ddi_html(html)
    info['DDI'] = '; '.join(ddi_list)

    return info


def extract_drugs_for_pathways(pid):
    # pathway_ids = pd.read_csv(pathway_file)['pathway_id'].tolist()
    records = []
    print("executing get_pathway_name...")
    pname = get_pathway_name(pid)
    print("pname received...")
    for did in get_drugs_for_pathway(pid):
        di = get_drug_info(did)
        records.append({
            'Pathway ID': pid or 'Not found',
            'Pathway Name': pname or 'Not found',
            'Drug ID': di['Drug ID'],
            'Drug name': di['Drug name'],
            'Class': di['Class'],
            'Target': '; '.join(di['Target']),
            'Efficacy': di['Efficacy'],
            'Brite': di['Brite'],
            'Approved': di['Approved'],
            'DDI': di['DDI']
        })
    
    return pd.DataFrame(records)


def find_drug_info(pathway):
    try:
        print("find_drug_info executing...")
        df = extract_drugs_for_pathways(pathway)
        if df.shape[0] == 0:
            raise ValueError("Relevant data not found. PubMed search recommended for drug names and attributes.")
        else:
            df = df[['Pathway ID',
                'Pathway Name',
                'Drug ID',
                'Drug name',
                'Class',
                'Target',
                'Efficacy',
                'Brite',
                'Approved']]
            print("renaming...")
            df = df.rename(columns={
                'Drug ID': 'drug_id',
                'Drug name': 'name',
                'Pathway ID': 'pathway_id',
                'Pathway Name': 'pathway_name',
                'Class': 'drug_class',
                'Target': 'target',
                'Efficacy': 'efficacy',
                'Brite': 'brite',
                'Approved': 'approved'
            })

            df['approved'] = df['approved'].apply(lambda x: True if x is True else False).astype(bool)
            
            # for dummy data only
            # data = [
            # {
            #     "pathway_id": "hsa05206",
            #     "pathway_name": "MicroRNAs in cancer - Homo sapiens (human)",
            #     "drug_id": "D11551",
            #     "name": "Valemetostat (INN)",
            #     "drug_class": "Not found",
            #     "target": "EZH2 [HSA:2145 2146] [KO:K17451 K11430]; PATHWAY   hsa00310(2145+2146)  Lysine degradation; hsa05206(2146)  MicroRNAs in cancer",
            #     "efficacy": "Antineoplastic, histone methyltransferase inhibitor",
            #     "brite": "Target-based classification of drugs [BR:br08310]\nEnzymes\nTransferases (EC2)\nMethyltransferases\nEZH\nD11551  Valemetostat (INN)",
            #     "approved": False
            # },
            # {
            #     "pathway_id": "hsa05206",
            #     "pathway_name": "MicroRNAs in cancer - Homo sapiens (human)",
            #     "drug_id": "D11164",
            #     "name": "Cobomarsen sodium (USAN)",
            #     "drug_class": "Not found",
            #     "target": "ADCK3 [HSA:406947] [KO:K17009]; PATHWAY   hsa05206(406947)  MicroRNAs in cancer",
            #     "efficacy": "Antineoplastic, microRNA inhibitor\nTYPE      Antisense oligonucleotide, antitrypsin",
            #     "brite": "Target-based classification of drugs [BR:br08310]\nNucleic acids\nRNAs\nMicroRNAs\nMIR155\nD11164  Cobomarsen sodium (USAN)",
            #     "approved": False
            # },
            # {
            #     "pathway_id": "hsa05206",
            #     "pathway_name": "MicroRNAs in cancer - Homo sapiens (human)",
            #     "drug_id": "D11740",
            #     "name": "Roducitabine (USAN)",
            #     "drug_class": "Not found",
            #     "target": "AGBL5 [HSA:1786] [KO:K00558]; PATHWAY   hsa00270(1786)  Cysteine and methionine metabolism; hsa05206(1786)  MicroRNAs in cancer",
            #     "efficacy": "Antineoplastic, Antimetabolite, anemia",
            #     "brite": "Target-based classification of drugs [BR:br08310]\nEnzymes\nTransferases (EC2)\nMethyltransferases\nDNMT1\nD11740  Roducitabine (USAN)",
            #     "approved": False
            # },
            # {
            #     "pathway_id": "hsa05206",
            #     "pathway_name": "MicroRNAs in cancer - Homo sapiens (human)",
            #     "drug_id": "D03665",
            #     "name": "Decitabine (USAN/INN);",
            #     "drug_class": "Antineoplastic",
            #     "target": "EZH2 [HSA:1786 1788 1789] [KO:K00558 K17398 K17399]; PATHWAY   hsa00270(1786+1788+1789)  Cysteine and methionine metabolism; hsa05206(1786+1788+1789)  MicroRNAs in cancer",
            #     "efficacy": "Antineoplastic, Antimetabolite\nDISEASE   Iron deficiency anemia [DS:H01481]",
            #     "brite": "Anatomical Therapeutic Chemical (ATC) classification [BR:br08303]... (truncated)",
            #     "approved": True
            # }
            # ]
            # df = pd.DataFrame(data)
            # print("df.shape")
            #df.to_csv("temp.csv", index= False)
            ############################################# dummies end
            return df
    except requests.exceptions.RequestException as re:
        print("networking exception occurs")
        raise RuntimeError(
            f"[KEGG_HELPER:NetworkError] Network request failed while fetching KEGG data for pathway '{pathway}'."
        ) from re

    except ValueError as ve:
        print("malformed exception occurs")
        raise RuntimeError(
            f"[KEGG_HELPER:MalformedData] Expected structure missing in KEGG data for pathway '{pathway}'. recomended pubmed_serach for drug names and drug attributes"
        ) from ve

    except Exception as exp:
        print("main exception occurs: ",exp)
        

