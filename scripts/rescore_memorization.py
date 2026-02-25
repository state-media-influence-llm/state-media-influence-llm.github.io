"""
Re-score existing completions.json using normalized edit distance < 0.4
(matching the paper's methodology). No API calls — just re-evaluates matches.
"""

import json
import re
from pathlib import Path

from query_memorization import normalized_edit_distance

BASE_DIR = Path(__file__).resolve().parent.parent
COMPLETIONS_PATH = BASE_DIR / "data" / "memorization" / "completions.json"

NORM_RE = re.compile(r'[\s，。、；：""''！？·…—（）【】《》\u3000]')


def norm(s: str) -> str:
    return NORM_RE.sub('', s)


def main():
    with open(COMPLETIONS_PATH, "r", encoding="utf-8") as f:
        completions = json.load(f)

    changed = 0
    for c in completions:
        expected = norm(c["expected"])
        completion = norm(c["completion"])
        comp_prefix = completion[:len(expected)]
        dist = normalized_edit_distance(comp_prefix, expected)
        new_matched = dist < 0.4

        if c.get("matched") != new_matched or "edit_distance" not in c:
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
