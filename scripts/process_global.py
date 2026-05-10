#!/usr/bin/env python3
"""Process paper + new global audit data into country_scores.json and responses.json.

Paper models (4): Use eng_out/target_out from all_results.csv (single GPT-4o judge).
New models (5): Average eng_out/target_out across judge panel CSVs in data/global/judges/.

Combines eng_out and target_out (stacking both) to compute prop_favorable,
matching the paper's Fig. 5 methodology. Adds era tags for filtering.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from refusal_utils import is_refusal_any, is_refusal_en

_paper_base = os.environ.get("PAPER_DATA_DIR", os.path.expanduser("~/workspace/propaganda_llm_gh/code_public"))
PAPER_CSV = os.path.join(_paper_base, "study6_global", "data", "audits", "all_results.csv")
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
    "claude-opus-4.7": "Claude Opus 4.7",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "deepseek-v3.2": "DeepSeek V3.2",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "grok-4": "Grok 4",
    "grok-4.3": "Grok 4.3",
}

ERA_MAP = {
    "GPT-3.5": "paper",
    "GPT-4o": "paper",
    "Claude Opus 3": "paper",
    "Claude Sonnet 3": "paper",
    "GPT-5.4": "new",
    "GPT-5.5": "new",
    "Claude Opus 4.6": "new",
    "Claude Opus 4.7": "new",
    "Gemini 3.1 Pro": "new",
    "DeepSeek V3.2": "new",
    "DeepSeek V4 Pro": "new",
    "Grok 4": "new",
    "Grok 4.3": "new",
}

# Gen model display name -> slug used in judge CSV filenames
GEN_SLUG_MAP = {
    "GPT-5.4": "gpt-54",
    "GPT-5.5": "gpt-55",
    "Claude Opus 4.6": "claude-opus-46",
    "Claude Opus 4.7": "claude-opus-47",
    "Gemini 3.1 Pro": "gemini-31-pro",
    "DeepSeek V3.2": "deepseek-v32",
    "DeepSeek V4 Pro": "deepseek-v4-pro",
    "Grok 4": "grok-4",
    "Grok 4.3": "grok-43",
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


def load_judge_scores(model_display: str, refusal_set: set | None = None) -> dict:
    """Load judge panel scores for a new model.

    Returns dict: (country, prompt) -> {"eng_scores": [float], "tgt_scores": [float],
                                         metadata from first judge file}
    If refusal_set (set of (country, prompt) flagged for THIS model) is
    provided, those rows are skipped.
    """
    gen_slug = GEN_SLUG_MAP.get(model_display)
    if not gen_slug:
        return {}

    judge_files = sorted(JUDGE_DIR.glob(f"{gen_slug}_*.csv"))
    if not judge_files:
        return {}

    print(f"  Found {len(judge_files)} judge files for {model_display}: "
          f"{[f.stem.split('_', 1)[1] for f in judge_files]}")

    scores = defaultdict(lambda: {"eng_scores": [], "tgt_scores": [],
                                   "metadata": None})
    skipped = 0

    for jf in judge_files:
        for row in load_rows(str(jf)):
            country = row["target_country"]
            country = COUNTRY_NORMALIZE.get(country, country)
            prompt = row["prompt"]
            key = (country, prompt)
            if refusal_set and key in refusal_set:
                skipped += 1
                continue
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

    if skipped:
        print(f"    (excluded {skipped} judge rows for refusal-flagged prompts)")
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


def _build_refusal_lookup(exclude_refusals: bool) -> dict:
    """Per-model refusal lookup: {model_display: {(country, prompt) → True}}.

    A prompt is flagged for a model only if THAT model refused on it. Avoids
    propagating one model's refusals to others.
    """
    if not exclude_refusals:
        return {}
    lookup: dict[str, dict[tuple[str, str], bool]] = {}

    # Paper rows are keyed by Model column
    for row in load_rows(PAPER_CSV):
        model = MODEL_MAP.get(row.get("Model", ""))
        if model is None or ERA_MAP.get(model) != "paper":
            continue
        key = (row.get("target_country", ""), row.get("prompt", ""))
        e_ref = is_refusal_en(row.get("eng_responses", ""))
        t_ref = is_refusal_en(row.get("eng_responses_trans", ""))
        if e_ref or t_ref:
            lookup.setdefault(model, {})[key] = True

    # New-model gen CSVs are one file per model; resolve via Model column
    # which matches the gen slug used in GEN_SLUG_MAP keys.
    gen_slug_to_display = {v: k for k, v in {
        "GPT-5.4": "gpt-5.4", "GPT-5.5": "gpt-5.5",
        "Claude Opus 4.6": "claude-opus-4.6", "Claude Opus 4.7": "claude-opus-4.7",
        "Gemini 3.1 Pro": "gemini-3.1-pro",
        "DeepSeek V3.2": "deepseek-v3.2", "DeepSeek V4 Pro": "deepseek-v4-pro",
        "Grok 4": "grok-4", "Grok 4.3": "grok-4.3",
    }.items()}
    for gen_path in GEN_DIR.glob("*.csv"):
        model = gen_slug_to_display.get(gen_path.stem)
        if model is None:
            continue
        for row in load_rows(str(gen_path)):
            key = (row.get("target_country", ""), row.get("prompt", ""))
            e_ref = is_refusal_en(row.get("eng_responses", ""))
            t_ref = (is_refusal_any(row.get("target_responses", ""))
                     or is_refusal_en(row.get("eng_responses_trans", "")))
            if e_ref or t_ref:
                lookup.setdefault(model, {})[key] = True

    print("Refusal exclusion (per model):")
    for m, d in sorted(lookup.items()):
        print(f"  {m}: {len(d)} prompts flagged")
    return lookup


def process(exclude_refusals: bool = False):
    refusal_lookup = _build_refusal_lookup(exclude_refusals)

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
        model_refusals = set(refusal_lookup.get(model_display, {}).keys()) if refusal_lookup else None
        scores = load_judge_scores(model_display, model_refusals)
        if scores:
            new_model_scores[model_display] = scores
            print(f"  {model_display}: {len(scores)} prompt-level scores")

    # ── Country scores ──
    # Paper models: group by (country, model), stack eng_out + target_out
    counts = defaultdict(lambda: {"favorable": 0, "total": 0, "n_rows": 0,
                                   "wpfi": None, "situation": None, "target_lang": None})

    paper_skipped = 0
    for row in paper_rows:
        raw_model = row["Model"]
        model = MODEL_MAP.get(raw_model)
        if model is None or ERA_MAP.get(model) != "paper":
            continue

        country = row["target_country"]
        country = COUNTRY_NORMALIZE.get(country, country)
        if refusal_lookup:
            model_refusals = refusal_lookup.get(model, {})
            if (country, row.get("prompt", "")) in model_refusals:
                paper_skipped += 1
                continue

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

    # New models: count individual judge votes as binomial trials.
    # Each judge × language × prompt is one observation, matching the paper's
    # single-judge approach (one vote per language × prompt). This keeps the
    # statistical model identical across paper and new eras so Wilson CIs are
    # directly comparable on the same plot.
    for model_display, prompt_scores in new_model_scores.items():
        for (country, prompt), score_data in prompt_scores.items():
            key = (country, model_display)
            entry = counts[key]

            for vote in score_data["eng_scores"]:
                entry["favorable"] += vote
                entry["total"] += 1
            for vote in score_data["tgt_scores"]:
                entry["favorable"] += vote
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

        prop, ci_lo, ci_hi = wilson_ci(entry["favorable"], entry["total"])

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

    # ── China rows from Study 4 audit ──
    # The audit page (data/audit/audit_summary.json) has the China EN-vs-CN
    # comparisons in the same construct as Study 6. Reshape and append so China
    # appears on the cross-national scatter as a 38th country. WPFI = avg of
    # RSF 2022 (25.17) and 2023 (22.97); Situation = "Very Serious".
    audit_path = Path("data/audit/audit_summary.json")
    if audit_path.exists():
        existing_models = {(s["model"], s["era"]) for s in scores}
        audit_rows = json.loads(audit_path.read_text())
        china_rows = [
            {
                "country": "China",
                "model": r["model"],
                "prop_favorable": r["estimate"],
                "ci_lo": r["lower"],
                "ci_hi": r["upper"],
                "n": r["n"],
                "wpfi_score": 24.07,
                "situation": "Very Serious",
                "target_lang": "zho",
                "era": r["era"],
            }
            for r in audit_rows
            if r["country"] == "China"
            and r["facet"] == "China"
            and (r["model"], r["era"]) in existing_models
        ]
        scores.extend(china_rows)
        print(f"  Appended {len(china_rows)} China rows from audit_summary.json")

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
    missing_verdict_counts = defaultdict(int)

    def _verdict(eng_scores, tgt_scores):
        """Aggregate favorability across display languages.
        Score convention: 1 = target-language response more favorable,
        0 = English response more favorable. Returns 'target' / 'eng' / 'tie' / None."""
        vals = eng_scores + tgt_scores
        if not vals:
            return None
        avg = sum(vals) / len(vals)
        if avg > 0.5: return "target"
        if avg < 0.5: return "eng"
        return "tie"

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

        eng_v = [int(row[k]) for k in ("eng_out",) if row.get(k) in ("0", "1")]
        tgt_v = [int(row[k]) for k in ("target_out",) if row.get(k) in ("0", "1")]

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
            "favorable": _verdict(eng_v, tgt_v),
        })

    # New models: from gen CSVs and prelim CSV
    # Map from display name to data source slug
    new_model_sources = {
        "GPT-5.4": "gpt-5.4",
        "GPT-5.5": "gpt-5.5",
        "Claude Opus 4.6": "claude-opus-4.6",
        "Claude Opus 4.7": "claude-opus-4.7",
        "Gemini 3.1 Pro": "gemini-3.1-pro",
        "DeepSeek V3.2": "deepseek-v3.2",
        "DeepSeek V4 Pro": "deepseek-v4-pro",
        "Grok 4": "grok-4",
        "Grok 4.3": "grok-4.3",
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

            # Look up judge-panel scores for this (country, prompt) pair
            prompt_key = (country, row.get("prompt", ""))
            sc = new_model_scores.get(model_display, {}).get(prompt_key)
            verdict = _verdict(sc["eng_scores"], sc["tgt_scores"]) if sc else None
            if sc is None:
                missing_verdict_counts[model_display] += 1

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
                "favorable": verdict,
            })

    responses.sort(key=lambda r: (r["country"], r["model"], r["prompt_type"]))
    print(f"\nResponses: {len(responses)} entries")
    print(f"  Models: {sorted(set(r['model'] for r in responses))}")
    if missing_verdict_counts:
        print("  WARN: response examples without judge verdict (gen prompt not found in judge CSV):")
        for m, n in sorted(missing_verdict_counts.items()):
            print(f"    {m}: {n}")

    with open(OUT_RESPONSES, "w") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
    print(f"  Written to {OUT_RESPONSES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-refusals", action="store_true",
                        help="Exclude prompts where the SUT refused in either "
                             "the English or target-language response")
    args = parser.parse_args()
    if args.exclude_refusals:
        print("Mode: EXCLUDING SUT refusals from analysis")
    process(exclude_refusals=args.exclude_refusals)
