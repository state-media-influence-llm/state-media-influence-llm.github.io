"""
Re-score existing completions.json using normalized edit distance < 0.4
(matching the paper's methodology). No API calls — just re-evaluates matches.
"""

import json
import re
from pathlib import Path

from query_memorization import fuzzy_match

BASE_DIR = Path(__file__).resolve().parent.parent
COMPLETIONS_PATH = BASE_DIR / "data" / "memorization" / "completions.json"


def main():
    with open(COMPLETIONS_PATH, "r", encoding="utf-8") as f:
        completions = json.load(f)

    changed = 0
    for c in completions:
        if c.get("timestamp") == "paper":
            continue  # paper entries have known-good distances from the paper
        new_matched, dist = fuzzy_match(c["completion"], c["expected"])

        if c.get("matched") != new_matched or abs(c.get("edit_distance", -1) - dist) > 0.005:
            changed += 1

        c["edit_distance"] = round(dist, 2)
        c["matched"] = new_matched

    with open(COMPLETIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(completions, f, ensure_ascii=False, indent=2)

    total = len(completions)
    matched = sum(1 for c in completions if c["matched"])
    print(f"Re-scored {total} completions ({changed} changed). {matched} matched (edit dist < 0.4).")


if __name__ == "__main__":
    main()
