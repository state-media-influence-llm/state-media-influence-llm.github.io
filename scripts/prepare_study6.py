#!/usr/bin/env python3
"""Prepare Study 6 data for the interactive global page.

Reads the full all_results.csv (~376K rows) and produces:
  - data/global/country_scores.json  (scatter plot aggregates)
  - data/global/responses.json       (curated response examples)
"""

import csv
import json
import math
import os
from collections import defaultdict

SRC = os.path.expanduser(
    "~/workspace/influence_attribution/propaganda_llm/code_public/"
    "study6_global/data/audits/all_results.csv"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "global")

MODEL_LABELS = {
    "GPT3.5": "GPT-3.5",
    "GPT4o": "GPT-4o",
    "Opus": "Claude Opus",
    "Sonnet": "Claude Sonnet",
}


def wilson_ci(p, n, z=1.96):
    """Wilson score 95% confidence interval."""
    if n == 0:
        return (0, 0)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0, centre - spread), min(1, centre + spread))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Pass 1: aggregate scores ---
    groups = defaultdict(lambda: {"sum": 0, "n": 0, "meta": {}})
    # --- Pass 2: collect candidate response examples ---
    examples = defaultdict(dict)  # key: (country, model) -> {prompt_type: row}

    with open(SRC, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["placebo"] != "Results":
                continue

            country = row["target_country"]
            model = row["Model"]
            key = (country, model)

            # Aggregate eng_out (binary: 1 = favorable to regime in target lang)
            try:
                eng_out = int(row["eng_out"])
            except (ValueError, KeyError):
                continue

            g = groups[key]
            g["sum"] += eng_out
            g["n"] += 1
            g["meta"] = {
                "Score_ave": float(row["Score_ave"]) if row["Score_ave"] else None,
                "Situation": row["Situation"],
                "target": row["target"],
                "country": country,
            }

            # Collect one example per prompt_type for the response widget
            prompt_type = row["prompt_type"]
            if prompt_type not in examples[key]:
                examples[key][prompt_type] = {
                    "prompt_type": prompt_type,
                    "prompt": row["prompt"],
                    "target_prompt": row["target_prompt"],
                    "eng_response": row["eng_responses"],
                    "target_response": row["target_responses"],
                    "translation": row["eng_responses_trans"],
                    "target": row["target"],
                }

    # Build country_scores.json
    scores = []
    for (country, model), g in sorted(groups.items()):
        n = g["n"]
        p = g["sum"] / n if n > 0 else 0
        ci_lo, ci_hi = wilson_ci(p, n)
        scores.append({
            "country": country,
            "model": MODEL_LABELS.get(model, model),
            "prop_favorable": round(p, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "n": n,
            "wpfi_score": g["meta"].get("Score_ave"),
            "situation": g["meta"].get("Situation"),
            "target_lang": g["meta"].get("target"),
        })

    with open(os.path.join(OUT_DIR, "country_scores.json"), "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(scores)} entries to country_scores.json")

    # Build responses.json — 1 per prompt_type per (country, model)
    curated = []
    for (country, model), type_map in sorted(examples.items()):
        for pt in ["country", "institution", "leader"]:
            if pt not in type_map:
                continue
            r = type_map[pt]
            curated.append({
                "country": country,
                "model": MODEL_LABELS.get(model, model),
                "prompt_type": pt,
                "prompt": r["prompt"],
                "target_prompt": r["target_prompt"],
                "eng_response": r["eng_response"],
                "target_response": r["target_response"],
                "translation": r["translation"],
                "target_lang": r["target"],
            })

    with open(os.path.join(OUT_DIR, "responses.json"), "w") as f:
        json.dump(curated, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(curated)} examples to responses.json")


if __name__ == "__main__":
    main()
