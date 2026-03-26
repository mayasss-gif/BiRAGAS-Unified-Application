import re
import json, ast

def parse_agent_output(output_str: str) -> dict:
    """
    Parses agent output which may be:
    - valid JSON string
    - Python-style dict with single quotes
    - wrapped in ```json code fences
    """
    # Strip code block markers like ```json ... ```
    output_str = output_str.strip()
    if output_str.startswith("```"):
        output_str = re.sub(r"^```(?:json)?\s*", "", output_str, flags=re.IGNORECASE)
        output_str = re.sub(r"\s*```$", "", output_str)

    # Try standard JSON parse first
    try:
        return json.loads(output_str)
    except json.JSONDecodeError:
        pass

    # Fallback: safely evaluate a Python dict literal
    try:
        return ast.literal_eval(output_str)
    except (ValueError, SyntaxError):
        raise ValueError("Agent output is not valid JSON or Python dict")

if __name__ == "__main__":
    # Example usage
    example_output = """
    ```json 
    {
        "disease": "Pancreatic Cancer",
        "geo_path": "/path/to/geo",
        "ae_path": "/path/to/ae"
    }
    ```
    """
    parsed_output = parse_agent_output(example_output)
    print(parsed_output)  