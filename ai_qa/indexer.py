"""
Codebase Indexer
Walks project root, chunks files by logical blocks, embeds via nomic-embed-text (Ollama),
and persists to ChromaDB under .ai-agent/index/
"""

import os
import json
import hashlib
import re
from pathlib import Path
from typing import Generator

import ollama
import chromadb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".py", ".java", ".ts", ".tsx", ".js", ".jsx",
    ".go", ".rs", ".kt", ".swift",
    ".sql", ".yaml", ".yml", ".json", ".md", ".txt",
    ".xml", ".toml", ".env.example", ".sh", ".bash",
}

IGNORE_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", "target", ".next", ".nuxt", "coverage",
    ".ai-agent", ".idea", ".vscode", "vendor",
}

IGNORE_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Cargo.lock",
}

EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 60        # lines per chunk (fallback for non-AST chunking)
CHUNK_OVERLAP = 10     # overlapping lines between chunks


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(root: Path) -> Generator[Path, None, None]:
    """Walk project root and yield indexable source files."""
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        # Skip ignored directories
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        # Skip ignored filenames
        if path.name in IGNORE_FILES:
            continue
        # Skip hidden files
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.suffix in SUPPORTED_EXTENSIONS:
            yield path


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_by_lines(text: str, file_path: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Fallback chunker — splits by line windows with overlap.
    Returns list of dicts: {id, text, file, start_line, end_line}
    """
    lines = text.splitlines()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, max(1, len(lines) - overlap), step):
        chunk_lines = lines[i : i + chunk_size]
        if not chunk_lines:
            break
        chunk_text = "\n".join(chunk_lines).strip()
        if len(chunk_text) < 30:   # skip near-empty chunks
            continue
        chunk_id = hashlib.md5(f"{file_path}:{i}".encode()).hexdigest()
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "file": file_path,
            "start_line": i + 1,
            "end_line": i + len(chunk_lines),
        })
    return chunks


def chunk_by_functions(text: str, file_path: str, ext: str):
    """
    Lightweight AST-style chunker using regex for common patterns.
    Falls back to line chunking for unknown file types.
    """
    patterns = {
        ".py":   r"^(class |def |async def )",
        ".java": r"^(\s*(public|private|protected|static|void|class|interface|@))",
        ".ts":   r"^(export |const |function |class |interface |type |async )",
        ".tsx":  r"^(export |const |function |class |interface |type |async )",
        ".js":   r"^(export |const |function |class |module\.exports)",
        ".jsx":  r"^(export |const |function |class |module\.exports)",
        ".go":   r"^(func |type |var |const )",
        ".kt":   r"^(fun |class |object |interface |data class )",
    }

    pattern = patterns.get(ext)
    if not pattern:
        return chunk_by_lines(text, file_path)

    lines = text.splitlines()
    split_indices = [0]
    for i, line in enumerate(lines):
        if i > 0 and re.match(pattern, line):
            split_indices.append(i)
    split_indices.append(len(lines))

    chunks = []
    for idx in range(len(split_indices) - 1):
        start = split_indices[idx]
        end = split_indices[idx + 1]
        block_lines = lines[start:end]

        # If a block is too large, sub-chunk it
        if len(block_lines) > CHUNK_SIZE * 2:
            sub = chunk_by_lines("\n".join(block_lines), file_path)
            for s in sub:
                s["start_line"] += start
                s["end_line"] += start
            chunks.extend(sub)
            continue

        chunk_text = "\n".join(block_lines).strip()
        if len(chunk_text) < 30:
            continue
        chunk_id = hashlib.md5(f"{file_path}:{start}".encode()).hexdigest()
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "file": file_path,
            "start_line": start + 1,
            "end_line": end,
        })
    return chunks


# ---------------------------------------------------------------------------
# Hashing (incremental re-index)
# ---------------------------------------------------------------------------

def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_hash_cache(agent_dir: Path) -> dict:
    cache_file = agent_dir / "cache" / "file_hashes.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return {}


def save_hash_cache(agent_dir: Path, cache: dict):
    cache_file = agent_dir / "cache" / "file_hashes.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, indent=2))


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


# ---------------------------------------------------------------------------
# Main indexer
# ---------------------------------------------------------------------------

def index_project(root: str, force: bool = False, verbose: bool = True):
    root_path = Path(root).resolve()
    agent_dir = root_path / ".ai-agent"
    agent_dir.mkdir(exist_ok=True)

    # ChromaDB
    db_client = chromadb.PersistentClient(path=str(agent_dir / "index"))
    collection = db_client.get_or_create_collection(
        name="codebase",
        metadata={"hnsw:space": "cosine"},
    )

    # Hash cache for incremental updates
    hash_cache = {} if force else load_hash_cache(agent_dir)
    new_cache = {}

    files = list(discover_files(root_path))
    indexed, skipped, failed = 0, 0, 0

    for file_path in files:
        rel_path = str(file_path.relative_to(root_path))
        fhash = file_hash(file_path)

        # Skip if unchanged
        if not force and hash_cache.get(rel_path) == fhash:
            new_cache[rel_path] = fhash
            skipped += 1
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_by_functions(text, rel_path, file_path.suffix)

            if not chunks:
                continue

            # Remove old chunks for this file before re-adding
            try:
                existing = collection.get(where={"file": rel_path})
                if existing["ids"]:
                    collection.delete(ids=existing["ids"])
            except Exception:
                pass

            # Embed and add each chunk
            ids, embeddings, documents, metadatas = [], [], [], []
            for chunk in chunks:
                emb = embed(chunk["text"])
                ids.append(chunk["id"])
                embeddings.append(emb)
                documents.append(chunk["text"])
                metadatas.append({
                    "file": chunk["file"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "extension": file_path.suffix,
                })

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            new_cache[rel_path] = fhash
            indexed += 1

            if verbose:
                print(f"  ✓ {rel_path} ({len(chunks)} chunks)")

        except Exception as e:
            failed += 1
            if verbose:
                print(f"  ✗ {rel_path}: {e}")

    save_hash_cache(agent_dir, {**hash_cache, **new_cache})

    total_chunks = collection.count()
    print(f"\n{'─'*50}")
    print(f"  Indexed : {indexed} files")
    print(f"  Skipped : {skipped} files (unchanged)")
    print(f"  Failed  : {failed} files")
    print(f"  Total chunks in DB: {total_chunks}")
    print(f"{'─'*50}")

    return {"indexed": indexed, "skipped": skipped, "failed": failed, "total_chunks": total_chunks}
