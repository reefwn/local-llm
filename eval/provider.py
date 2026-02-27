"""
Promptfoo Python provider that calls the local RAG API.
"""
import requests
import json


API_URL = "http://localhost:8000/ask"


def call_api(prompt, options, context):
    """Call the /ask endpoint and collect the streamed response."""
    try:
        response = requests.post(
            API_URL,
            json={"question": prompt},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        full_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                full_text += line.removeprefix("data: ")

        return {"output": full_text}
    except Exception as e:
        return {"error": str(e)}
