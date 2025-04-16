# utils/llm_handler.py

import os
import openai
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

def call_llm_generate_sql(messages):
    return call_llm_groq_chat(messages, model=DEFAULT_GROQ_MODEL)

def call_llm_summarize_sql_result(messages):
    return call_llm_groq_chat(messages, model=DEFAULT_GROQ_MODEL)

def call_llm_summarize_schema(schema_text):
    prompt = f"""You are a data analyst. The user uploaded datasets with the following structure:{schema_text}
    Please summarize what tables and columns are available, and suggest how the data might relate. Use clear, readable language.
    Keep only 20 words on one line and give it like a wrapped text.
    """
    return call_llm_groq_chat([{"role": "user", "content": prompt}])

def call_llm_groq_chat(messages, model=DEFAULT_GROQ_MODEL, temperature=0.7, max_tokens=1024):
    """
    Calls a Groq-hosted LLM with OpenAI-compatible chat API.

    Parameters:
        messages (list): List of dicts with roles and content
        model (str): Groq model ID (e.g., llama-4-maverick)
        temperature (float): Sampling temperature
        max_tokens (int): Output token limit

    Returns:
        str: LLM's response text
    """
    if not GROQ_API_KEY:
        return "⚠️ Missing GROQ_API_KEY in .env"

    try:
        client = openai.OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_API_URL
        )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[Groq Error - {model}]:", e)
        return f"⚠️ Groq model '{model}' error: {e}"
