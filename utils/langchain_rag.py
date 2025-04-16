# utils/langchain_rag.py

from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

load_dotenv()
# === CONFIG ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "your-groq-key"  # Replace or load via .env
GROQ_MODEL = os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"     # Change to your preferred model

# === Embedding and LLM setup ===
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
from langchain_openai import ChatOpenAI  # ✅ instead of langchain.chat_models

llm = ChatOpenAI(
    model="llama3-70b-8192",
    temperature=0.2,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# === Prompt Template ===
prompt_template = """You are a helpful assistant. Use the following context to answer the user's question.
Only answer based on the context. If the answer is not in the context, say you don’t have enough data.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# === Build RAG Chain ===
def prepare_rag_chain(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    vectorstore = FAISS.from_documents(docs, embedding_model)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", k=3),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def test_rag_from_file(file_path, question):
    """
    Load .txt or .csv schema summary → build vector index → ask question
    """
    if not os.path.exists(file_path):
        return "❌ File not found."

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chain = prepare_rag_chain(text)
    return chain.invoke({"query": question})

