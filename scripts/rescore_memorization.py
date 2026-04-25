"""Re-score existing memorization completions without re-querying models.

Recalculates matched/edit_distance/refused fields on all entries in
completions.json. Useful when the matching logic changes (e.g., switching
from the paper's prefix-truncation to sliding-window matching) or when
refusal-detection regexes are updated. No API calls are made.

The default mode (--windowed) uses a sliding-window variant that finds the
best-matching n-character window anywhere in the completion, which handles
current models that prepend meta-commentary before the actual memorized text.
The --prefix flag reproduces the paper's original prefix-truncation method.

Usage:
    python rescore_memorization.py              # default: windowed matching
    python rescore_memorization.py --prefix     # paper's original prefix-truncation
"""

import argparse
import json
from pathlib import Path

# Import scoring functions from query_memorization
from query_memorization import fuzzy_match, is_refusal

BASE_DIR = Path(__file__).resolve().parent.parent
COMPLETIONS_PATH = BASE_DIR / "data" / "memorization" / "completions.json"
PHRASES_PATH = BASE_DIR / "data" / "memorization" / "phrases.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", action="store_true",
                        help="Use paper's original prefix-truncation matching "
                             "(default is sliding-window matching)")
    args = parser.parse_args()
    windowed = not args.prefix

    mode = "windowed" if windowed else "prefix"
    print(f"Scoring mode: {mode}")

    with open(COMPLETIONS_PATH, "r", encoding="utf-8") as f:
        completions = json.load(f)

    # Load phrases to get raw start text for prompt stripping
    with open(PHRASES_PATH, "r", encoding="utf-8") as f:
        phrases = json.load(f)
    start_by_id = {p["id"]: p["start"] for p in phrases}

    changes = {"match_gained": 0, "match_lost": 0, "dist_changed": 0, "refusals": 0}

    for c in completions:
        old_matched = c.get("matched")
        old_dist = c.get("edit_distance")

        prompt_start = start_by_id.get(c["phrase_id"], "")
        new_matched, new_dist, match_start, match_end = fuzzy_match(
            c["completion"], c["expected"],
            prompt_start=prompt_start,
            windowed=windowed)
        refused = is_refusal(c["completion"])

        c["matched"] = new_matched
        c["edit_distance"] = round(new_dist, 4)
        c["refused"] = refused
        c["match_start"] = match_start
        c["match_end"] = match_end

        if old_matched != new_matched:
            if new_matched:
                changes["match_gained"] += 1
            else:
                changes["match_lost"] += 1
        if old_dist is not None and abs(old_dist - round(new_dist, 4)) > 0.001:
            changes["dist_changed"] += 1
        if refused:
            changes["refusals"] += 1

    # Save
    tmp = str(COMPLETIONS_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(completions, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(COMPLETIONS_PATH)

    print(f"Re-scored {len(completions)} completions ({mode} mode)")
    print(f"  Matches gained: {changes['match_gained']}")
    print(f"  Matches lost: {changes['match_lost']}")
    print(f"  Edit distances changed: {changes['dist_changed']}")
    print(f"  Refusals detected: {changes['refusals']}")

    # Per-model summary
    models = sorted(set(c["model"] for c in completions))
    for model in models:
        model_data = [c for c in completions if c["model"] == model]
        for ptype in ["propaganda", "culturax"]:
            subset = [c for c in model_data if c["type"] == ptype]
            if not subset:
                continue
            refused = sum(1 for c in subset if c.get("refused"))
            non_refused = [c for c in subset if not c.get("refused")]
            matched = sum(1 for c in non_refused if c["matched"])
            total = len(non_refused)
            rate = matched / total if total > 0 else 0
            print(f"  {model} {ptype}: {matched}/{total} ({rate:.1%})"
                  f" [refused: {refused}]")


if __name__ == "__main__":
    main()
