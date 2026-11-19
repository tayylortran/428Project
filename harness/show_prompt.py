"""Print the exact prompt sent to a model for one query. No API calls.

Usage:
    python -m harness.show_prompt "When is lap swim at the pool?"
    python -m harness.show_prompt "When is lap swim at the pool?" --metadata --defense delimit
    python -m harness.show_prompt --list-metadata
"""
import argparse
import json
from pathlib import Path

from app.metadata import load_metadata
from app.prompt import build_prompt
from app.retriever import load_corpus, retrieve

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--metadata", action="store_true", help="include document metadata")
    ap.add_argument("--defense", default="none", choices=["none", "delimit"])
    ap.add_argument("--list-metadata", action="store_true", help="print loaded metadata for every doc")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "config" / "experiment.json").read_text())
    docs = load_corpus(ROOT / "corpus")
    metadata = load_metadata(ROOT / "corpus", docs)

    if args.list_metadata:
        print(json.dumps(metadata, indent=2))
        return
    if not args.query:
        ap.error("give a query or --list-metadata")

    retrieved = retrieve(args.query, docs, k=cfg["top_k"])
    system, user = build_prompt(args.query, retrieved, args.defense,
                                metadata if args.metadata else None)
    print(f"retrieved: {[name for name, _ in retrieved]}\n")
    print(f"--- SYSTEM ---\n{system}\n\n--- USER ---\n{user}")


if __name__ == "__main__":
    main()
