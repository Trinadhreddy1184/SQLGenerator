# utils/llm_handler_openai.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 🔐 Credentials and endpoint for GitHub-based model
INFERENCE_API_KEY = os.getenv("GITHUB_TOKEN")
INFERENCE_API_URL = "https://models.inference.ai.azure.com"
INFERENCE_MODEL = "gpt-4o"

# ⚙️ Create OpenAI-compatible client
client = OpenAI(
    base_url=INFERENCE_API_URL,
    api_key=INFERENCE_API_KEY,
)

def call_llm_generate_sql(messages):
    return call_gpt4o_inference(messages)

def call_llm_summarize_sql_result(messages):
    return call_gpt4o_inference(messages)

def call_gpt4o_inference(messages, temperature=0.2, top_p=1.0, max_tokens=1028):
    """
    Call GPT-4o via the GitHub-inference API endpoint.
    """
    try:
        response = client.chat.completions.create(
            model=INFERENCE_MODEL,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("[Inference GPT-4o Error]:", e)
        return "⚠️ Error calling GPT-4o inference model."
