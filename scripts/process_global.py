#!/usr/bin/env python3
"""Process paper + new global audit data into country_scores.json and responses.json.

Paper models (4): Use eng_out/target_out from all_results.csv (single GPT-4o judge).
New models (5): Average eng_out/target_out across judge panel CSVs in data/global/judges/.

Combines eng_out and target_out (stacking both) to compute prop_favorable,
matching the paper's Fig. 5 methodology. Adds era tags for filtering.
"""

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

PAPER_CSV = os.path.expanduser(
    "~/workspace/propaganda_llm_gh/code_public/study6_global/data/audits/all_results.csv"
)
NEW_CSV = "data/global/gpt5_opus4_prel.csv"
JUDGE_DIR = Path("data/global/judges")
GEN_DIR = Path("data/global/gen")
OUT_SCORES = "data/global/country_scores.json"
OUT_RESPONSES = "data/global/responses.json"

MODEL_MAP = {
    # Paper models
    "GPT3.5": "GPT-3.5",
    "GPT4o": "GPT-4o",
    "Opus": "Claude Opus 3",
    "Sonnet": "Claude Sonnet 3",
    # New models (from prelim CSV)
    "GPT5.4": "GPT-5.4",
    "Opus4.6": "Claude Opus 4.6",
}

# New models from gen CSVs (Model column value -> display name)
GEN_MODEL_MAP = {
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "deepseek-v3.2": "DeepSeek V3.2",
    "grok-4": "Grok 4",
}

ERA_MAP = {
    "GPT-3.5": "paper",
    "GPT-4o": "paper",
    "Claude Opus 3": "paper",
    "Claude Sonnet 3": "paper",
    "GPT-5.4": "new",
    "Claude Opus 4.6": "new",
    "Gemini 3.1 Pro": "new",
    "DeepSeek V3.2": "new",
    "Grok 4": "new",
}

# Gen model display name -> slug used in judge CSV filenames
GEN_SLUG_MAP = {
    "GPT-5.4": "gpt-54",
    "Claude Opus 4.6": "claude-opus-46",
    "Gemini 3.1 Pro": "gemini-31-pro",
    "DeepSeek V3.2": "deepseek-v32",
    "Grok 4": "grok-4",
}

# Normalize country names across datasets
COUNTRY_NORMALIZE = {
    "Türkiye": "Turkey",
}


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return round(center, 4), round(max(0, center - margin), 4), round(min(1, center + margin), 4)


def load_rows(path):
    """Load CSV rows, handling large fields."""
    rows = []
    csv.field_size_limit(10_000_000)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_judge_scores(model_display: str) -> dict:
    """Load judge panel scores for a new model.

    Returns dict: (country, prompt) -> {"eng_scores": [float], "tgt_scores": [float],
                                         metadata from first judge file}
    Averages across all available judges.
    """
    gen_slug = GEN_SLUG_MAP.get(model_display)
    if not gen_slug:
        return {}

    # Find all judge files for this gen model
    judge_files = sorted(JUDGE_DIR.glob(f"{gen_slug}_*.csv"))
    if not judge_files:
        return {}

    print(f"  Found {len(judge_files)} judge files for {model_display}: "
          f"{[f.stem.split('_', 1)[1] for f in judge_files]}")

    # Collect scores keyed by (country, prompt)
    scores = defaultdict(lambda: {"eng_scores": [], "tgt_scores": [],
                                   "metadata": None})

    for jf in judge_files:
        for row in load_rows(str(jf)):
            country = row["target_country"]
            country = COUNTRY_NORMALIZE.get(country, country)
            prompt = row["prompt"]
            key = (country, prompt)
            entry = scores[key]

            eng_out = row.get("eng_out", "")
            tgt_out = row.get("target_out", "")

            if eng_out in ("0", "1"):
                entry["eng_scores"].append(int(eng_out))
            if tgt_out in ("0", "1"):
                entry["tgt_scores"].append(int(tgt_out))

            if entry["metadata"] is None:
                entry["metadata"] = {
                    "prompt_type": row.get("prompt_type", ""),
                    "entity": row.get("entity", ""),
                    "target": row.get("target", ""),
                    "Score_ave": row.get("Score_ave", ""),
                    "Situation": row.get("Situation", ""),
                }

    return dict(scores)


def load_gen_responses(model_slug: str) -> list[dict]:
    """Load response data from gen CSVs for response examples."""
    # Try gen directory first
    gen_path = GEN_DIR / f"{model_slug}.csv"
    if gen_path.exists():
        return load_rows(str(gen_path))

    # Fall back to prelim CSV
    rows = []
    model_col_map = {"gpt-5.4": "GPT5.4", "claude-opus-4.6": "Opus4.6"}
    model_col = model_col_map.get(model_slug)
    if model_col and os.path.exists(NEW_CSV):
        for row in load_rows(NEW_CSV):
            if row["Model"] == model_col:
                rows.append(row)
    return rows


def process():
    # ── Paper models (4): use eng_out/target_out from all_results.csv ──
    print("Loading paper data...")
    paper_rows = load_rows(PAPER_CSV)
    print(f"  {len(paper_rows)} rows")

    # ── New models: load judge panel scores ──
    print("\nLoading judge panel scores for new models...")
    new_model_scores = {}  # model_display -> {(country, prompt) -> scores}
    for model_display in ERA_MAP:
        if ERA_MAP[model_display] != "new":
            continue
        scores = load_judge_scores(model_display)
        if scores:
            new_model_scores[model_display] = scores
            print(f"  {model_display}: {len(scores)} prompt-level scores")

    # ── Country scores ──
    # Paper models: group by (country, model), stack eng_out + target_out
    counts = defaultdict(lambda: {"favorable": 0, "total": 0, "n_rows": 0,
                                   "wpfi": None, "situation": None, "target_lang": None})

    for row in paper_rows:
        raw_model = row["Model"]
        model = MODEL_MAP.get(raw_model)
        if model is None or ERA_MAP.get(model) != "paper":
            continue

        country = row["target_country"]
        country = COUNTRY_NORMALIZE.get(country, country)

        eng_out = row.get("eng_out", "")
        target_out = row.get("target_out", "")

        key = (country, model)
        entry = counts[key]

        for val in [eng_out, target_out]:
            if val in ("0", "1"):
                entry["total"] += 1
                entry["favorable"] += int(val)

        entry["n_rows"] += 1

        if entry["wpfi"] is None:
            try:
                entry["wpfi"] = float(row.get("Score_ave", row.get("Score", "")))
            except (ValueError, TypeError):
                pass
            entry["situation"] = row.get("Situation", "")
            entry["target_lang"] = row.get("target", "")

    # New models: use averaged judge panel scores
    for model_display, prompt_scores in new_model_scores.items():
        for (country, prompt), score_data in prompt_scores.items():
            key = (country, model_display)
            entry = counts[key]

            # Average across judges, then treat as continuous score
            # Stack eng and target averages
            if score_data["eng_scores"]:
                avg_eng = sum(score_data["eng_scores"]) / len(score_data["eng_scores"])
                entry["favorable"] += avg_eng
                entry["total"] += 1

            if score_data["tgt_scores"]:
                avg_tgt = sum(score_data["tgt_scores"]) / len(score_data["tgt_scores"])
                entry["favorable"] += avg_tgt
                entry["total"] += 1

            entry["n_rows"] += 1

            if entry["wpfi"] is None:
                meta = score_data["metadata"]
                try:
                    entry["wpfi"] = float(meta.get("Score_ave", ""))
                except (ValueError, TypeError):
                    pass
                entry["situation"] = meta.get("Situation", "")
                entry["target_lang"] = meta.get("target", "")

    scores = []
    for (country, model), entry in sorted(counts.items()):
        if entry["total"] == 0:
            continue
        era = ERA_MAP.get(model)
        if era is None:
            continue

        if era == "paper":
            # Binary counts → Wilson CI
            prop, ci_lo, ci_hi = wilson_ci(entry["favorable"], entry["total"])
        else:
            # Continuous averages → use proportion directly, approximate CI
            prop = entry["favorable"] / entry["total"] if entry["total"] > 0 else 0.5
            prop = round(prop, 4)
            # Approximate Wilson CI treating averaged scores as if they were binary
            # n = number of judge-averaged observations (eng + target per prompt)
            n_approx = entry["total"]
            k_approx = entry["favorable"]
            _, ci_lo, ci_hi = wilson_ci(round(k_approx), n_approx)

        scores.append({
            "country": country,
            "model": model,
            "prop_favorable": prop,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n": entry["total"],
            "wpfi_score": entry["wpfi"],
            "situation": entry["situation"],
            "target_lang": entry["target_lang"],
            "era": era,
        })

    print(f"\nCountry scores: {len(scores)} entries")
    print(f"  Models: {sorted(set(s['model'] for s in scores))}")
    print(f"  Countries: {len(set(s['country'] for s in scores))}")

    with open(OUT_SCORES, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"  Written to {OUT_SCORES}")

    # ── Response examples ──
    # Collect up to MAX_PER_COMBO examples per (country, model, prompt_type)
    MAX_PER_COMBO = 3
    seen_counts = defaultdict(int)
    responses = []

    # Paper models: from all_results.csv
    for row in paper_rows:
        raw_model = row["Model"]
        model = MODEL_MAP.get(raw_model)
        if model is None or ERA_MAP.get(model) != "paper":
            continue

        country = row["target_country"]
        country = COUNTRY_NORMALIZE.get(country, country)
        prompt_type = row.get("prompt_type", "")

        key = (country, model, prompt_type)
        if seen_counts[key] >= MAX_PER_COMBO:
            continue
        seen_counts[key] += 1

        responses.append({
            "country": country,
            "model": model,
            "prompt_type": prompt_type,
            "prompt": row.get("prompt", ""),
            "target_prompt": row.get("target_prompt", ""),
            "eng_response": row.get("eng_responses", ""),
            "target_response": row.get("target_responses", ""),
            "translation": row.get("eng_responses_trans", ""),
            "target_lang": row.get("target", ""),
            "era": "paper",
        })

    # New models: from gen CSVs and prelim CSV
    # Map from display name to data source slug
    new_model_sources = {
        "GPT-5.4": "gpt-5.4",
        "Claude Opus 4.6": "claude-opus-4.6",
        "Gemini 3.1 Pro": "gemini-3.1-pro",
        "DeepSeek V3.2": "deepseek-v3.2",
        "Grok 4": "grok-4",
    }

    for model_display, source_slug in new_model_sources.items():
        if model_display not in new_model_scores:
            continue  # Only include models that have judge data
        resp_rows = load_gen_responses(source_slug)
        if not resp_rows:
            continue

        for row in resp_rows:
            country = row.get("target_country", "")
            country = COUNTRY_NORMALIZE.get(country, country)
            prompt_type = row.get("prompt_type", "")

            key = (country, model_display, prompt_type)
            if seen_counts[key] >= MAX_PER_COMBO:
                continue
            seen_counts[key] += 1

            responses.append({
                "country": country,
                "model": model_display,
                "prompt_type": prompt_type,
                "prompt": row.get("prompt", ""),
                "target_prompt": row.get("target_prompt", ""),
                "eng_response": row.get("eng_responses", ""),
                "target_response": row.get("target_responses", ""),
                "translation": row.get("eng_responses_trans", ""),
                "target_lang": row.get("target", ""),
                "era": "new",
            })

    responses.sort(key=lambda r: (r["country"], r["model"], r["prompt_type"]))
    print(f"\nResponses: {len(responses)} entries")
    print(f"  Models: {sorted(set(r['model'] for r in responses))}")

    with open(OUT_RESPONSES, "w") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    print(f"  Written to {OUT_RESPONSES}")


if __name__ == "__main__":
    process()
