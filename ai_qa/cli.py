#!/usr/bin/env python3
"""
ai-qa — Codebase Q&A Agent CLI
Run from the root of any project.

Usage:
  ai-qa index              # Index current project
  ai-qa index --force      # Re-index everything (ignore cache)
  ai-qa ask "question"     # Ask a one-shot question
  ai-qa chat               # Interactive multi-turn chat
  ai-qa stats              # Show index stats
"""

import os
import sys
import click
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent))

from ai_qa.indexer import index_project
from ai_qa.qa_agent import ask, get_index_stats


def get_root():
    """Use CWD as project root (agent runs at project root)."""
    return os.getcwd()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """🔍 ai-qa — Ask questions about your codebase using local Gemma 4 + Ollama"""
    pass


@cli.command()
@click.option("--force", is_flag=True, default=False, help="Re-index all files (ignore cache)")
@click.option("--quiet", is_flag=True, default=False, help="Suppress per-file output")
def index(force, quiet):
    """Index the current project directory."""
    root = get_root()
    click.echo(f"\n📁 Indexing: {root}")
    click.echo(f"   Model:    nomic-embed-text (via Ollama)")
    click.echo(f"   Store:    .ai-agent/index/\n")

    result = index_project(root, force=force, verbose=not quiet)

    if result["indexed"] == 0 and result["skipped"] > 0:
        click.echo("\n✅ Index is up to date. Use --force to re-index.")
    else:
        click.echo("\n✅ Done! Run  ai-qa chat  to start asking questions.")


@cli.command()
@click.argument("question")
@click.option("--top-k", default=8, help="Number of code chunks to retrieve (default: 8)")
@click.option("--no-stream", is_flag=True, default=False, help="Disable streaming output")
def ask_cmd(question, top_k, no_stream):
    """Ask a one-shot question about the codebase."""
    root = get_root()

    click.echo(f"\n🔎 Searching codebase...\n")

    try:
        if no_stream:
            answer = ask(question, root, top_k=top_k, stream=False)
            click.echo(answer)
        else:
            click.echo("─" * 60)
            for token in ask(question, root, top_k=top_k, stream=True):
                click.echo(token, nl=False)
            click.echo("\n" + "─" * 60)
    except FileNotFoundError as e:
        click.echo(f"\n❌ {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--top-k", default=8, help="Number of code chunks to retrieve per question")
def chat(top_k):
    """Interactive multi-turn chat about the codebase."""
    root = get_root()

    # Check index exists
    stats = get_index_stats(root)
    if stats["status"] == "not_indexed":
        click.echo("\n❌ No index found. Run:  ai-qa index\n", err=True)
        sys.exit(1)

    click.echo(f"\n🤖 Codebase Q&A  —  {stats['total_files']} files  |  {stats['total_chunks']} chunks")
    click.echo(f"   Project: {root}")
    click.echo(f"   Model:   gemma4 via Ollama")
    click.echo(f"\n   Type your question and press Enter. Type 'exit' or Ctrl+C to quit.\n")
    click.echo("─" * 60)

    history = []   # conversation memory

    while True:
        try:
            question = click.prompt("\n💬 You", prompt_suffix="> ").strip()
        except (KeyboardInterrupt, EOFError, click.exceptions.Abort):
            click.echo("\n\n👋 Bye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q", "bye"}:
            click.echo("\n👋 Bye!")
            break
        if question.lower() in {"clear", "reset"}:
            history = []
            click.echo("  🔄 Conversation cleared.")
            continue
        if question.lower() in {"stats", "status"}:
            s = get_index_stats(root)
            click.echo(f"  📊 {s['total_files']} files | {s['total_chunks']} chunks")
            continue

        click.echo("\n🤖 Agent\n")
        click.echo("─" * 60)

        full_answer = ""
        try:
            for token in ask(question, root, top_k=top_k, stream=True, history=history):
                click.echo(token, nl=False)
                full_answer += token
        except FileNotFoundError as e:
            click.echo(f"\n❌ {e}", err=True)
            continue

        click.echo("\n" + "─" * 60)

        # Append to history for multi-turn context
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full_answer})

        # Keep history manageable (last 10 turns)
        if len(history) > 20:
            history = history[-20:]


@cli.command()
def stats():
    """Show index statistics for the current project."""
    root = get_root()
    s = get_index_stats(root)

    if s["status"] == "not_indexed":
        click.echo(f"\n❌ Not indexed yet. Run:  ai-qa index\n")
    else:
        click.echo(f"\n📊 Index Stats — {root}")
        click.echo(f"   Files   : {s['total_files']}")
        click.echo(f"   Chunks  : {s['total_chunks']}")
        click.echo(f"   Status  : ✅ Ready\n")


# Alias: `ai-qa ask "question"` → also works as `ai-qa "question"` shortcut
cli.add_command(ask_cmd, name="ask")


if __name__ == "__main__":
    cli()
