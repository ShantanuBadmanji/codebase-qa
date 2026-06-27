# 🔍 ai-qa — Local Codebase Q&A Agent

Ask questions about any codebase using **Gemma 4 + Ollama** — fully local, no API keys, no data leaving your machine.

---

## How it works

```
Your question
     ↓
nomic-embed-text  →  vector embedding
     ↓
ChromaDB  →  top-K similar code chunks retrieved
     ↓
Gemma 4  →  answers with file + line citations
```

The index lives in `.ai-agent/index/` at the project root. Only changed files are re-indexed on subsequent runs.

---

## Prerequisites

```bash
# 1. Ollama running with both models pulled
ollama pull gemma4
ollama pull nomic-embed-text

# 2. Python 3.11+
python --version
```

---

## Install

Clone the repo first:

```bash
git clone <repo-url>
cd codebase-qa
```

### Recommended: install globally with `uv` (use `ai-qa` in ANY project)

This installs `ai-qa` into an isolated environment and puts the command on your `PATH`, so you can run it from the root of any project — no virtualenv to activate.

```bash
# editable (-e) so your code edits to ai_qa/ still take effect without reinstalling
uv tool install --editable .

# if `ai-qa` isn't found afterwards, add uv's bin dir to your PATH, then restart the shell
uv tool update-shell

# Verify
ai-qa --help
```

> Don't have uv? Install it from https://docs.astral.sh/uv/ (`curl -LsSf https://astral.sh/uv/install.sh | sh`),
> or use the pip alternative below.

### Alternative: pip / pipx

```bash
# Global CLI via pipx (isolated, on PATH everywhere)
pipx install --editable .

# Or a plain editable install into the current environment / venv
pip install -e .
```

> Note: `uv run ai-qa ...` only works **inside this project's** directory. To use the command in
> other projects, install it globally with one of the methods above.

---

## Usage

### Index a project
```bash
cd /path/to/your-project
ai-qa index

# Force full re-index (e.g. after major changes)
ai-qa index --force
```

### Ask a one-shot question
```bash
ai-qa ask "Where is the ReworkDensity metric calculated?"
ai-qa ask "Trace the flow from PR webhook to database insert"
ai-qa ask "What indexes exist on the issue_event_log table?"
ai-qa ask "Find all places we query the event_log table"
```

### Interactive chat (multi-turn)
```bash
ai-qa chat
```

In chat mode:
- Follow-up questions retain conversation context
- Type `clear` to reset conversation history
- Type `stats` to see index info
- Type `exit` or Ctrl+C to quit

### Check index stats
```bash
ai-qa stats
```

---

## Project structure after indexing

```
your-project/
├── .ai-agent/
│   ├── index/          ← ChromaDB vector store
│   └── cache/
│       └── file_hashes.json   ← incremental re-index cache
├── src/
└── ...
```

Add `.ai-agent/` to your `.gitignore`.

---

## Configuration

Edit `agent/qa_agent.py` to tune:

| Variable | Default | Description |
|---|---|---|
| `CHAT_MODEL` | `gemma4` | Ollama model for answers |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `TOP_K` | `8` | Chunks retrieved per question |
| `MIN_RELEVANCE` | `0.3` | Cosine similarity cutoff |

Edit `indexer/indexer.py` to tune:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `60` | Lines per chunk (fallback) |
| `CHUNK_OVERLAP` | `10` | Overlap between chunks |
| `SUPPORTED_EXTENSIONS` | see file | File types to index |
| `IGNORE_DIRS` | see file | Directories to skip |

---

## Adding to a new project

Once installed globally (see [Install](#install)), `ai-qa` works from the root of any codebase — Java, Go, TypeScript, etc. The command always operates on your **current working directory**.

```bash
# Just cd into it and index — that's it
cd /any/project
ai-qa index
ai-qa chat
```

Each project gets its own independent `.ai-agent/` index, so you can use the same `ai-qa` command across many projects without them interfering.
