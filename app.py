"""
Main Flask application for SQL assistant with RAG capabilities
"""

# Standard library imports
import os
import re
import secrets
import shutil

# Third-party imports
from flask import Flask, request, jsonify, render_template, session

# Local module imports
from utils.db_runner import run_query, load_table, describe_all_tables
from utils.file_handler import parse_file
from utils.llm_handler import (
    call_llm_groq_chat,
    call_llm_generate_sql,
    call_llm_summarize_sql_result,
    call_llm_summarize_schema
)
from utils.llm_handler_openai import call_gpt4o_inference
from utils.rag_store import embed_and_store, query_store, reset_store

# --------------------------
# Flask Configuration
# --------------------------

app = Flask(__name__)
app.secret_key = "some-secret-key"
UPLOAD_FOLDER = "sessions"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --------------------------
# Helper Functions
# --------------------------

def extract_sql_from_response(response: str) -> str:
    """
    Extract SQL query from markdown-formatted LLM response.
    Handles both code blocks and raw SQL responses.
    """
    matches = re.findall(r"```(?:sql)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()

    # Fallback to find first SELECT/WITH clause
    match = re.search(r"(select|with)\s.+", response, re.IGNORECASE | re.DOTALL)
    return match.group(0).strip() if match else response.strip()


def is_data_query(text: str) -> bool:
    """Check if user input requires data/SQL processing"""
    text = text.lower()
    keywords = ["sql", "schema", "column", "table",
                "dataset", "data", "query", "rows",
                "from", "select"]
    return any(word in text for word in keywords)


def is_schema_insight_request(text: str) -> bool:
    """Detect schema analysis requests"""
    keywords = ["analyze", "what does this data",
                "structure", "columns", "relationships",
                "schema"]
    return any(word in text.lower() for word in keywords)


# --------------------------
# Application Routes
# --------------------------

@app.route("/")
def index():
    """Initialize new session"""
    session.clear()
    session["id"] = secrets.token_hex(8)
    return render_template("index.html")


@app.route("/set_model", methods=["POST"])
def set_model():
    """Handle model selection"""
    model = request.form.get("model")
    groq_model = request.form.get("groq_model")

    if model not in ["groq", "gpt4o"]:
        return jsonify({"status": "error", "message": "Invalid model"}), 400

    session["llm"] = model
    if model == "groq" and groq_model:
        session["groq_model"] = groq_model

    return jsonify({
        "status": "success",
        "model": model,
        "groq_model": groq_model
    })


@app.route("/chat", methods=["POST"])
def chat():
    """Main chat processing endpoint"""
    # Initialization and validation
    user_text = request.form.get("user_text", "").strip()
    uploaded_files = request.files.getlist("uploaded_file")

    if not user_text:
        return jsonify({"reply": "❗ Prompt cannot be empty."})

    # Session setup
    if "chat_history" not in session:
        session["chat_history"] = [{
            "role": "system",
            "content": (
                "You are a helpful AI assistant. If the user asks about structured datasets, "
                "analyze them and generate SQL queries using the schemas provided. "
                "If the user uploads documents, use those for RAG-style answers. "
                "If the user wants a general conversation, respond normally."
            )
        }]
        reset_store()

    session.setdefault("mode", "chat")
    session.setdefault("llm", "gpt4o")

    # Mode handling
    user_lower = user_text.lower()
    if any(kw in user_lower for kw in ["normal conversation", "switch to chat", "end sql", "stop data"]):
        session["mode"] = "chat"
        reset_store()
    elif is_data_query(user_text):
        session["mode"] = "data"

    # File processing
    session_id = session["id"]
    user_folder = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(user_folder, exist_ok=True)

    schema_description = ""
    sample_rows = []

    for file in uploaded_files:
        if not file or not file.filename:
            continue

        filename = file.filename
        save_path = os.path.join(user_folder, filename)
        file.save(save_path)
        ext = filename.split(".")[-1].lower()

        if ext in ["txt", "md", "log"]:
            with open(save_path, "r", encoding="utf-8") as f:
                embed_and_store(f.read(), source=filename)
                session["mode"] = "data"

        elif ext in ["csv", "tsv"]:
            df = parse_file(save_path)
            if df is not None:
                table_name = filename.rsplit(".", 1)[0]
                load_table(df, table_name)
                schema_description += f"Table `{table_name}`: {', '.join(df.columns)}\n"
                sample_rows.append(df.head(3).to_dict(orient="records"))
                session["mode"] = "data"

    if schema_description:
        session["schema"] = schema_description
        session["samples"] = sample_rows

    # LLM Processing
    session["chat_history"].append({"role": "user", "content": user_text})
    prompt_history = session["chat_history"][:-1]

    # SQL Mode Handling
    if session["mode"] == "data" and "schema" in session:
        # Schema analysis special case
        if is_schema_insight_request(user_text):
            try:
                description_df = describe_all_tables()
                schema_str = description_df.to_markdown(index=False)
                summary = call_llm_summarize_schema(schema_str)
                return jsonify({"reply": f"<pre class='schema-summary'>{summary}</pre>"})
            except Exception as e:
                return jsonify({"reply": f"❌ Error retrieving schema: {e}"})

        # SQL generation and execution
        sql_prompt = (
            "You are a SQL assistant. Given the user's question, and the following table schemas:\n\n"
            f"{session['schema']}\n\n"
            f"User: {user_text}\nRespond ONLY with a valid SQL query."
        )

        raw_response = call_llm_generate_sql(prompt_history + [{"role": "user", "content": sql_prompt}])
        generated_sql = extract_sql_from_response(raw_response)
        generated_sql = generated_sql.encode("ascii", errors="ignore").decode()
        generated_sql = generated_sql.replace("’", "'").replace("“", '"').replace("”", '"').rstrip(";")

        try:
            df_result = run_query(generated_sql)
        except Exception as e:
            return jsonify({
                "reply": f"⚠️ SQL Execution Error:\n```sql\n{generated_sql}\n```\n\n{str(e)}"
            })

        if df_result is None:
            return jsonify({"reply": f"❗ SQL Execution Error:\n```sql\n{generated_sql}\n```"})

        # Result processing
        result_html = df_result.head(20).to_html(classes="data-table", index=True, border=0).strip()
        html_template = f"""
            <div class="result-container">
            <div class="query-result-title">Query Result:</div>
            {result_html}</div>
            <div class="query-result-title">The executed query is:</div>
            <pre class="code-block">{generated_sql}</pre>
            <div class="summary"><b>Summary:</b></div>
            <p>"Replace Summary here with proper HTML tags"</p>
            """

        # Result summarization
        summary_prompt = (
            f"User asked:\n{user_text}\n\n"
            f"SQL Executed:\n```sql\n{generated_sql.strip()}\n```\n\n"
            f"Result (Top 20 rows):\n{result_html}\n\n"
            "Print the result in tabular format neatly first. Show the executed query. "
            "Summarize the output and ask if the user wants to modify this query."
            f"""Use this format efficiently : {html_template}"""
        )

        summary = call_llm_summarize_sql_result(prompt_history + [{"role": "user", "content": summary_prompt}])
        session["chat_history"].append({"role": "assistant", "content": summary})
        return jsonify({"reply": summary})

    # General conversation handling
    final_prompt = user_text
    prompt_history.append({"role": "user", "content": final_prompt})

    if session["llm"] == "groq":
        groq_model = session.get("groq_model", "llama3-70b-8192")
        reply = call_llm_groq_chat(prompt_history, model=groq_model)
    else:
        reply = call_gpt4o_inference(prompt_history)

    session["chat_history"].append({"role": "assistant", "content": reply})
    return jsonify({"reply": reply})


@app.route("/end_session", methods=["POST"])
def end_session():
    """Clean up session resources"""
    if "id" in session:
        shutil.rmtree(os.path.join(UPLOAD_FOLDER, session["id"]), ignore_errors=True)
    session.clear()
    reset_store()
    return ("", 204)


if __name__ == "__main__":
    app.run()