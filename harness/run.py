"""Experiment runner: every model x defense x query x trial -> one CSV row, then a summary.

Usage:
    python -m harness.run --mock                  # no API keys needed
    python -m harness.run                         # real models from config/experiment.json
    python -m harness.run --models claude-haiku-5-5 --trials 1
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.prompt import build_prompt
from app.retriever import load_corpus, retrieve
from harness.models import call_model

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["timestamp", "provider", "model", "defense", "query_id", "trial",
          "poisoned_retrieved", "success", "error", "answer"]


def load_dotenv(path):
    """Minimal .env reader: KEY=value lines, ignores comments and blanks."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "config" / "experiment.json", type=Path)
    ap.add_argument("--mock", action="store_true", help="use the offline mock model")
    ap.add_argument("--models", nargs="*", help="only run these model ids")
    ap.add_argument("--trials", type=int, help="override trials per case")
    ap.add_argument("--out", type=Path, help="CSV output path")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    load_dotenv(ROOT / ".env")

    models = [{"provider": "mock", "id": "mock"}] if args.mock else cfg["models"]
    if args.models:
        models = [m for m in models if m["id"] in args.models]
    for m in models:
        key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(m["provider"])
        if key and not os.environ.get(key):
            sys.exit(f"{key} is not set (needed for {m['id']}). Add it to .env or use --mock.")

    trials = args.trials or cfg["trials"]
    docs = load_corpus(ROOT / "corpus")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or ROOT / "results" / f"run_{stamp}{'_mock' if args.mock else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for m in models:
            for q in cfg["queries"]:
                retrieved = retrieve(q["text"], docs, k=cfg["top_k"])
                poisoned = any(name == cfg["poisoned_doc"] for name, _ in retrieved)
                system, user = build_prompt(q["text"], retrieved)
                for t in range(1, trials + 1):
                    answer, error = "", ""
                    try:
                        answer = call_model(m["provider"], m["id"], system, user, cfg)
                    except Exception as e:  # record the failure as a row, keep going
                        error = f"{type(e).__name__}: {e}"
                    writer.writerow({
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "provider": m["provider"], "model": m["id"],
                        "query_id": q["id"], "trial": t,
                        "poisoned_retrieved": int(poisoned),
                        "success": int(cfg["canary"].lower() in answer.lower()),
                        "error": error, "answer": answer,
                    })
                    f.flush()
                    print(f"{m['id']:<24} {q['id']} trial {t}: "
                            f"{'ERROR' if error else 'success' if cfg['canary'] in answer.lower() else 'no'}")

    summarize(out)


def summarize(csv_path):
    """Attack success rate = successes / trials where the poisoned doc was actually retrieved."""
    stats = defaultdict(lambda: [0, 0, 0])  # successes, poisoned trials, errors
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = stats[(r["model"], r["defense"])]
            if r["error"]:
                s[2] += 1
            elif r["poisoned_retrieved"] == "1":
                s[1] += 1
                s[0] += int(r["success"])

    print(f"\nResults written to {csv_path}\n")
    print(f"{'model':<24} {'success':>8} {'trials':>7} {'ASR':>7} {'errors':>7}")
    for (model, defense), (succ, n, err) in sorted(stats.items()):
        asr = f"{100 * succ / n:.1f}%" if n else "n/a"
        print(f"{model:<24} {succ:>8} {n:>7} {asr:>7} {err:>7}")


if __name__ == "__main__":
    main()
