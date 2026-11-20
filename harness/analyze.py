"""Print a Markdown comparison from saved trial CSVs. No model calls."""
import argparse
import csv
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    groups = defaultdict(list)
    seen = set()
    defenses = []
    for path in args.files:
        with open(path, newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                defense = row["defense"]
                if defense not in defenses:
                    defenses.append(defense)
                key = (row["model"], defense, row["query_id"], row["trial"])
                if key in seen:
                    parser.error(f"duplicate trial in {path}: {key}")
                seen.add(key)
                vector = row["poisoned_doc"] if row["poisoned_retrieved"] == "1" else "No poisoned retrieval"
                groups[row["model"], vector, defense].append(row)
    print("| Model | Retrieved source | " + " | ".join(defenses) + " |")
    print("|" + "---|" * (2 + len(defenses)))
    for model, vector in sorted({(m, v) for m, v, _ in groups}):
        cells = []
        for defense in defenses:
            rows = groups[model, vector, defense]
            valid = [row for row in rows if not row["error"]]
            successes = sum(int(row["success"]) for row in valid)
            errors = len(rows) - len(valid)
            cells.append(f"{successes}/{len(valid)} ({errors} errors)" if rows else "Missing")
        print("| " + " | ".join([model, vector, *cells]) + " |")


if __name__ == "__main__":
    main()
