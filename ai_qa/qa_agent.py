"""
Codebase Q&A Agent
Embeds user question → retrieves top-K chunks from ChromaDB → streams Gemma 4 answer with citations.
"""

from pathlib import Path
from typing import Iterator, List, Dict, Optional

import ollama
import chromadb

EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma4"        # or "gemma3" if using older Ollama tag
TOP_K = 8                    # number of chunks to retrieve
MIN_RELEVANCE = 0.3          # cosine distance cutoff (lower = more similar)


SYSTEM_PROMPT = """You are an expert software engineer and codebase assistant.
You have been given relevant source code chunks from the project the user is working in.
Your job is to answer questions accurately based ONLY on the provided code context.

Rules:
- Always cite the file path and line numbers when referencing code (e.g. `src/service/MetricsService.java:42-68`)
- If the answer is not in the provided context, say so clearly — do not guess or hallucinate
- Be concise but complete
- If asked to trace a flow, walk through it step by step referencing each file
- Format code references in backticks
"""


def get_collection(root: str):
    root_path = Path(root).resolve()
    db_path = root_path / ".ai-agent" / "index"
    if not db_path.exists():
        raise FileNotFoundError(
            f"No index found at {db_path}\n"
            f"Run: ai-qa index  (or python cli.py index)"
        )
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_collection("codebase")


def retrieve(question: str, root: str, top_k: int = TOP_K) -> List[Dict]:
    """Embed the question and retrieve the most relevant code chunks."""
    collection = get_collection(root)

    q_embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=question)

    results = collection.query(
        query_embeddings=[q_embedding["embedding"]],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist > (1 - MIN_RELEVANCE):   # cosine distance filter
            continue
        chunks.append({
            "text": doc,
            "file": meta["file"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "score": round(1 - dist, 3),   # convert distance → similarity
        })

    return chunks


def build_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] {chunk['file']}  (lines {chunk['start_line']}–{chunk['end_line']})  relevance={chunk['score']}"
        parts.append(f"{header}\n```\n{chunk['text']}\n```")
    return "\n\n".join(parts)


def ask(
    question: str,
    root: str,
    top_k: int = TOP_K,
    stream: bool = True,
    history: Optional[List[Dict]] = None,
) -> Iterator[str]:
    """
    Ask a question about the codebase.

    Args:
        question:  Natural language question
        root:      Project root directory
        top_k:     Number of chunks to retrieve
        stream:    Stream tokens or return full string
        history:   Previous conversation turns for multi-turn Q&A

    Yields (stream=True): token strings
    Returns (stream=False): full answer string
    """
    chunks = retrieve(question, root, top_k)

    if not chunks:
        msg = "No relevant code found for your question. Try rephrasing or re-indexing."
        if stream:
            yield msg
        else:
            return msg
        return

    context = build_context(chunks)

    # Build message history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history)

    messages.append({
        "role": "user",
        "content": (
            f"Here are the most relevant code chunks from the project:\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Question: {question}"
        ),
    })

    if stream:
        response = ollama.chat(model=CHAT_MODEL, messages=messages, stream=True)
        for chunk in response:
            token = chunk["message"]["content"]
            yield token
    else:
        response = ollama.chat(model=CHAT_MODEL, messages=messages, stream=False)
        return response["message"]["content"]


def get_index_stats(root: str) -> dict:
    """Return basic stats about the current index."""
    try:
        collection = get_collection(root)
        count = collection.count()
        # Sample a few entries to get file list
        sample = collection.peek(limit=1000)
        files = set(m["file"] for m in sample["metadatas"])
        return {
            "total_chunks": count,
            "total_files": len(files),
            "status": "ready",
        }
    except FileNotFoundError:
        return {"status": "not_indexed", "total_chunks": 0, "total_files": 0}
