import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import tiktoken
import json
from ..config.config import OPENAI_API_KEY, DEFAULT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def count_tokens(messages, model=DEFAULT_MODEL):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0
    for message in messages:
        total_tokens += 4
        for key, value in message.items():
            total_tokens += len(encoding.encode(value))
    total_tokens += 2
    return total_tokens

def extract_diagnosis_summary(pdf_file_path: str):
    reader = PdfReader(pdf_file_path)
    abnormal_lines = []
    abnormal_keywords = ['L', 'H', 'POSITIVE', 'EQUIVOCAL', '↑', '↓']

    for page in reader.pages:
        text = page.extract_text()
        for line in text.split('\n'):
            if any(keyword in line for keyword in abnormal_keywords):
                if not line.lower().startswith(("page", "printed", "lab director")):
                    abnormal_lines.append(line.strip())

    abnormal_text = "\n".join(abnormal_lines[:25])
    prompt = f"""
        You are a clinical diagnostic assistant. A patient has abnormal lab values.

        Your task:
        1. Group the abnormal values into "High" and "Low".
        2. Use clinical reasoning to suggest the most likely conditions (based on labs, not urgency).
        3. Identify the condition best supported by the lab data as "Priority".

        Return this JSON only:
        - Conditions: [List of likely conditions]
        - Abnormal_High: [List]
        - Abnormal_Low: [List]
        - Notes: [Brief explanation]
        - Priority: [Most likely condition]

        --- Lab Abnormalities ---
        {abnormal_text}
        --- End ---
        """
    messages = [
        {"role": "system", "content": "You are a clinical assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        # ✅ Count tokens per request
        token_count = count_tokens(messages, model=DEFAULT_MODEL)
        print(f"🔢 Tokens used in diagnosis summary: {token_count}")
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return None

def find_relevant_drugs_with_llm(priority_disease: str, df: pd.DataFrame, model=DEFAULT_MODEL):
    relevant_drugs = []
    total_tokens_used = 0  # Initialize counter

    for _, row in df.iterrows():
        efficacy_text = row.get("efficacy", "")
        prompt = f"""
            You are a biomedical expert.

            The patient has been diagnosed with: "{priority_disease}".

            Below is a drug's efficacy description.

            If the efficacy text shows the drug could treat this disease or is known to treat a **synonym, subtype, or pathophysiologically related** condition, classify it as a match.

            Efficacy:
            \"\"\"
            {efficacy_text}
            \"\"\"

            Respond in this JSON format:
            {{
            "Match": "Yes" or "No",
            "Reason": "<Short explanation>"
            }}

            Use your medical knowledge to infer synonyms or alternate names (e.g., "Iron deficiency anemia" → "anemia", "IDA"). Be cautious and conservative.
        """
        messages = [
            {"role": "system", "content": "You are a biomedical LLM expert."},
            {"role": "user", "content": prompt}
        ]

        try:
            # ✅ Count tokens per request
            token_count = count_tokens(messages, model)
            total_tokens_used += token_count
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0
            )
            response_text = response.choices[0].message.content.strip()
            response_text= response_text.strip("`").strip()
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()
                
            match_result = json.loads(response_text)

            if match_result.get("Match", "").strip().lower() == "yes":
                drug_data = row.to_dict()
                drug_data["LLM_Match_Reason"] = match_result.get("Reason", "")
                relevant_drugs.append(drug_data)

        except Exception as e:
            print(f"Error processing drug '{row.get('name', '')}': {e}")
            continue
    print(f"\n🔢 Total tokens used in LLM drug matching: {total_tokens_used}")
    return pd.DataFrame(relevant_drugs)

