def infer_grouping_with_llm(sample_ids, metadata_line, group_column, disease_hint=None):
    """
    Uses LLM to intelligently infer control vs disease group assignments.
    Returns a DataFrame with 'sample' and group_column.
    """
    prompt = f"""
You are a biomedical data agent.

You are given:
- A metadata line: {metadata_line}
- Sample IDs: {sample_ids}
- Characteristic column: {group_column}
- Optional disease context: {disease_hint or "None"}

Your task:
1. Extract the group value (e.g. "control", "treated", etc.) for each sample.
2. Return them in order of sample_ids.
3. Identify the most likely control group (based on keywords like 'control', 'untreated', 'vehicle').
4. Identify the disease/experimental group.
5. Return the result as JSON list of pairs: [sample_id, group_value]

JSON format:
[
  ["GSM123", "control"],
  ["GSM124", "30 mM lactate"],
  ...
]
"""
    # Use OpenAI or any LLM API
    response = call_openai_chat_completion(prompt)
    data = json.loads(response)

    return pd.DataFrame(data, columns=["sample", group_column])
