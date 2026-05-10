#!/usr/bin/env python3
"""Process Study 4 audit CSV files into summary JSON for the interactive chart.

Reads paper + new model _res.csv files, computes per-(model, country) proportion
of responses judged "more favorable" (Y=1), with binomial CIs.

Output: data/audit/audit_summary.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from refusal_utils import is_refusal_any

# Paths
_paper_base = Path(os.environ.get("PAPER_DATA_DIR", os.path.expanduser("~/workspace/propaganda_llm_gh/code_public")))
PAPER_DIR = _paper_base / "study4_production_model_audit" / "data"
NEW_DIR = Path("data/study4")
OUT_PATH = Path("data/audit/audit_summary.json")

# Model definitions: (file_slug, display_name, era)
PAPER_MODELS = [
    ("gpt3", "GPT-3.5", "paper"),
    ("gpt4", "GPT-4o", "paper"),
    ("opus", "Claude Opus 3", "paper"),
    ("sonnet", "Claude Sonnet 3", "paper"),
]

NEW_MODELS = [
    ("gpt-5-4", "GPT-5.4", "new"),
    ("gpt-5-5", "GPT-5.5", "new"),
    ("claude-opus-4-6", "Claude Opus 4.6", "new"),
    ("claude-opus-4-7", "Claude Opus 4.7", "new"),
    ("gemini-3-1-pro", "Gemini 3.1 Pro", "new"),
    ("deepseek-v3-2-speciale", "DeepSeek V3.2", "new"),
    ("deepseek-v4-pro", "DeepSeek V4 Pro", "new"),
    ("grok-4", "Grok 4", "new"),
    ("grok-4-3", "Grok 4.3", "new"),
]

PROMPT_TYPES = ["country", "leader", "inst"]

# Country mapping (exclude DE, matching paper)
COUNTRY_MAP = {
    "US": ("United States", "Baseline"),
    "UK": ("United Kingdom", "Baseline"),
    "CN": ("China", "China"),
    "NK": ("North Korea", "Spillover"),
    "RU": ("Russia", "Spillover"),
}


def _build_refusal_mask(gen_path: Path) -> pd.Series | None:
    """Per-prompt mask keyed by EN prompt: True if SUT refused in CN or EN.

    The result CSVs use the EN prompt as the row identifier (see
    run_judge_panel.build_comparison_df: data["prompt"] = list(en.prompt)),
    so we key the mask the same way.
    """
    if not gen_path.exists():
        return None
    df = pd.read_csv(gen_path)
    cn = df[df["language"] == "cn"].reset_index(drop=True)
    en = df[df["language"] == "en"].reset_index(drop=True)
    n = min(len(cn), len(en))
    en_prompts = en["prompt"].iloc[:n].tolist()
    cn_refused = cn["response_cn"].iloc[:n].apply(is_refusal_any).tolist()
    en_refused = en["response_en"].iloc[:n].apply(is_refusal_any).tolist()
    any_refused = [a or b for a, b in zip(cn_refused, en_refused)]
    return pd.Series(any_refused, index=en_prompts)


def load_model_data(base_dir, slug, model_name, era, exclude_refusals=False,
                    gen_base_dir=None):
    """Load all prompt-type CSVs for one model, return combined DataFrame.

    For new-era models, average Y across all panel judges (files matching
    {pt}_{slug}_res_{judge}.csv) so the chart reflects the full 6-judge
    panel rather than just the single-judge _res.csv that run_audit_study4
    produced with GPT-OSS-120B.

    For paper-era models, fall back to the paper's single-judge _res.csv
    (there is no panel data for those).

    If exclude_refusals=True, drop any rows where the SUT's CN or EN
    response was a refusal (looked up from the gen CSV by prompt text).
    """
    frames = []
    refusal_dropped = 0
    for pt in PROMPT_TYPES:
        # Build refusal mask once per question type
        refusal_mask = None
        if exclude_refusals:
            gen_dir = gen_base_dir if gen_base_dir is not None else base_dir
            refusal_mask = _build_refusal_mask(gen_dir / f"{pt}_{slug}.csv")

        panel_paths = sorted(base_dir.glob(f"{pt}_{slug}_res_*.csv"))
        if era == "new" and panel_paths:
            judge_frames = []
            for path in panel_paths:
                df = pd.read_csv(path)
                if "Y_cn" not in df.columns or "Y_en" not in df.columns:
                    continue
                keep_cols = ["country", "Y_cn", "Y_en"]
                if "prompt" in df.columns:
                    keep_cols.insert(0, "prompt")
                df = df[keep_cols].copy()
                df["prompt_type"] = pt
                if refusal_mask is not None and "prompt" in df.columns:
                    before = len(df)
                    refused_mask = df["prompt"].map(refusal_mask).fillna(False).astype(bool)
                    df = df[~refused_mask]
                    refusal_dropped += before - len(df)
                judge_frames.append(df.drop(columns=["prompt"], errors="ignore"))
            if judge_frames:
                frames.append(pd.concat(judge_frames, ignore_index=True))
        else:
            path = base_dir / f"{pt}_{slug}_res.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "Y_cn" not in df.columns or "Y_en" not in df.columns:
                continue
            keep_cols = ["country", "Y_cn", "Y_en"]
            if "prompt" in df.columns:
                keep_cols.insert(0, "prompt")
            df = df[keep_cols].copy()
            df["prompt_type"] = pt
            if refusal_mask is not None and "prompt" in df.columns:
                before = len(df)
                refused_mask = df["prompt"].map(refusal_mask).fillna(False).astype(bool)
                df = df[~refused_mask]
                refusal_dropped += before - len(df)
            frames.append(df.drop(columns=["prompt"], errors="ignore"))
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined["model"] = model_name
    combined["era"] = era
    if exclude_refusals and refusal_dropped > 0:
        print(f"    (excluded {refusal_dropped} refusal rows)")
    return combined


def compute_summary(df):
    """Compute per-(model, country) proportion favorable with binomial CI."""
    records = []
    for (model, country_code, era), grp in df.groupby(["model", "country", "era"]):
        if country_code not in COUNTRY_MAP:
            continue
        country_name, facet = COUNTRY_MAP[country_code]

        # Convert Y from {-1, 1} to binary {0, 1}: max(Y, 0)
        y_cn = grp["Y_cn"].dropna().apply(lambda v: max(v, 0))
        y_en = grp["Y_en"].dropna().apply(lambda v: max(v, 0))
        y_all = pd.concat([y_cn, y_en])

        n = len(y_all)
        if n < 20:
            continue

        p = y_all.mean()
        se = np.sqrt(p * (1 - p) / n)

        records.append({
            "model": model,
            "country": country_name,
            "country_code": country_code,
            "facet": facet,
            "estimate": round(p, 4),
            "se": round(se, 4),
            "lower": round(max(p - 1.96 * se, 0), 4),
            "upper": round(min(p + 1.96 * se, 1), 4),
            "n": n,
            "era": era,
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-refusals", action="store_true",
                        help="Exclude rows where the SUT's CN or EN response "
                             "was a refusal (looked up from the gen CSV)")
    args = parser.parse_args()
    if args.exclude_refusals:
        print("Mode: EXCLUDING SUT refusals from analysis")

    all_frames = []

    for slug, name, era in PAPER_MODELS:
        # Paper data has _res.csv files but the gen step (response_cn/_en) is
        # in the same dir for the older format, so reuse PAPER_DIR for both.
        df = load_model_data(PAPER_DIR, slug, name, era,
                              exclude_refusals=args.exclude_refusals,
                              gen_base_dir=PAPER_DIR)
        if df is not None:
            print(f"  {name}: {len(df)} rows")
            all_frames.append(df)

    for slug, name, era in NEW_MODELS:
        df = load_model_data(NEW_DIR, slug, name, era,
                              exclude_refusals=args.exclude_refusals,
                              gen_base_dir=NEW_DIR)
        if df is not None:
            print(f"  {name}: {len(df)} rows")
            all_frames.append(df)
        else:
            print(f"  {name}: no data yet")

    if not all_frames:
        print("No data found!")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    summary = compute_summary(combined)

    # Sort: paper first, then by model name
    era_order = {"paper": 0, "new": 1}
    summary.sort(key=lambda r: (era_order.get(r["era"], 2), r["model"], r["country"]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {len(summary)} records to {OUT_PATH}")
    models = sorted(set(r["model"] for r in summary))
    for m in models:
        n = sum(1 for r in summary if r["model"] == m)
        print(f"  {m}: {n} country groups")


if __name__ == "__main__":
    main()
