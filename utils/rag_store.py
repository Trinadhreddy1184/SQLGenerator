# utils/rag_store.py

import os
import faiss
import json
import numpy as np
from pathlib import Path

# 🔁 Embedding backends
from sentence_transformers import SentenceTransformer

# Optional: OpenAI embedding support
try:
    import openai
except ImportError:
    openai = None

# === CONFIG ===
STORAGE_DIR = Path("vector_store")
STORAGE_DIR.mkdir(exist_ok=True)
INDEX_PATH = STORAGE_DIR / "index.faiss"
METADATA_PATH = STORAGE_DIR / "metadata.json"

EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local")  # local | openai | instructor
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# === Globals ===
MODEL = None
INDEX = None
CHUNK_METADATA = []


# === Embedding Utilities ===
def load_embedder():
    global MODEL
    if EMBEDDING_MODE == "openai":
        if not openai:
            raise ImportError("OpenAI not installed. Use `pip install openai`.")
    elif EMBEDDING_MODE == "local":
        MODEL = SentenceTransformer(EMBEDDING_MODEL)
    elif EMBEDDING_MODE == "instructor":
        from InstructorEmbedding import INSTRUCTOR
        MODEL = INSTRUCTOR(EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unsupported embedding mode: {EMBEDDING_MODE}")


def embed_texts(texts: list[str]) -> np.ndarray:
    if EMBEDDING_MODE == "openai":
        res = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        return np.array([d["embedding"] for d in res["data"]], dtype="float32")
    elif EMBEDDING_MODE == "instructor":
        return np.array(MODEL.encode([[t, "Represent a passage for retrieval"] for t in texts]))
    else:
        return np.array(MODEL.encode(texts, convert_to_numpy=True))


# === FAISS Store Management ===
def save_index():
    faiss.write_index(INDEX, str(INDEX_PATH))
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(CHUNK_METADATA, f, indent=2)


def load_index():
    global INDEX, CHUNK_METADATA
    if INDEX_PATH.exists() and METADATA_PATH.exists():
        INDEX = faiss.read_index(str(INDEX_PATH))
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            CHUNK_METADATA = json.load(f)
        print(f"[RAG] Loaded {len(CHUNK_METADATA)} chunks from disk.")
    else:
        reset_store()


def reset_store():
    global INDEX, CHUNK_METADATA
    INDEX = faiss.IndexFlatL2(384)  # 384 = MiniLM embedding dim
    CHUNK_METADATA = []


# === Core APIs ===
def embed_and_store(text: str, source: str):
    """
    Splits, embeds, and adds to FAISS index with metadata.
    """
    global INDEX, CHUNK_METADATA

    if not MODEL and EMBEDDING_MODE != "openai":
        load_embedder()

    if not INDEX:
        reset_store()

    # Split text into chunks
    words = text.split()
    chunks = []
    chunk_size = 500
    overlap = 50
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    vectors = embed_texts(chunks)

    for i, chunk in enumerate(chunks):
        CHUNK_METADATA.append({
            "text": chunk,
            "source": source,
            "chunk_id": i
        })

    INDEX.add(vectors)
    save_index()
    print(f"[RAG] Stored {len(chunks)} chunks from {source}.")


def query_store(query: str, top_k: int = 3) -> list[str]:
    """
    Searches FAISS for top matching chunks to the query.
    """
    if not INDEX or not CHUNK_METADATA:
        load_index()

    query_vec = embed_texts([query])
    distances, indices = INDEX.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(CHUNK_METADATA):
            chunk_info = CHUNK_METADATA[idx]
            results.append(f"[{chunk_info['source']} - chunk {chunk_info['chunk_id']}]:\n{chunk_info['text']}")
    return results
