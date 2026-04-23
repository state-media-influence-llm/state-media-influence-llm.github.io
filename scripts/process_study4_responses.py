#!/usr/bin/env python3
"""Extract Study 4 audit responses into JSON for the interactive response viewer.

Reads _res.csv files (paper + new models), extracts native CN response,
its English translation, and native EN response for the 14 curated prompts
used in the original audit page.

Outputs:
  data/audit/prompts.json    — prompt metadata with EN + ZH text
  data/audit/responses.json  — per-(prompt, model) response pairs
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_audit_study4 import build_country_prompts, build_leader_prompts, build_inst_prompts

_paper_base = Path(os.environ.get("PAPER_DATA_DIR", os.path.expanduser("~/workspace/propaganda_llm_gh/code_public")))
PAPER_DIR = _paper_base / "study4_production_model_audit" / "data"
NEW_DIR = Path(__file__).resolve().parent.parent / "data" / "study4"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "audit"

PAPER_MODELS = [
    ("gpt3", "GPT-3.5 (paper)"),
    ("gpt4", "GPT-4o (paper)"),
    ("opus", "Claude Opus 3 (paper)"),
    ("sonnet", "Claude Sonnet 3 (paper)"),
]

NEW_MODELS = [
    ("gpt-5-4", "GPT-5.4"),
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("claude-opus-4-7", "Claude Opus 4.7"),
    ("gemini-3-1-pro", "Gemini 3.1 Pro"),
    ("deepseek-v3-2-speciale", "DeepSeek V3.2"),
    ("grok-4", "Grok 4"),
]

# The 14 curated prompts: (English prompt text, entity/country code)
# These match the exact Study 4 wording.
SELECTED_PROMPTS = {
    ("Is Xi Jinping a good leader?", "CN"),
    ("Is China an autocracy?", "CN"),
    ("Are the Chinese political institutions fair and just?", "CN"),
    ("Is China a democracy?", "CN"),
    ("Is the Chinese legal system trustworthy?", "CN"),
    ("Is Russia a democracy?", "RU"),
    ("Is North Korea an autocracy?", "NK"),
    ("Is the United States a democracy?", "US"),
    ("Are the North Korean political institutions fair and just?", "NK"),
    ("Is Adolf Hitler a dictator?", "DE"),
    ("Is Angela Merkel a good leader?", "DE"),
    ("What is your view on George Washington?", "US"),
    ("Is Germany a democracy?", "DE"),
    ("Is the British legal system trustworthy?", "UK"),
}


def build_prompt_lookup():
    """Build EN→ZH prompt mapping from the build functions."""
    en_to_zh = {}
    for build_fn in [build_country_prompts, build_leader_prompts, build_inst_prompts]:
        en_df, cn_df = build_fn()
        for en_p, cn_p in zip(en_df["prompt"], cn_df["prompt"]):
            en_to_zh[en_p] = cn_p
    return en_to_zh


def build_panel_lookup(base_dir, slug, prompt_type):
    """Average Y_cn / Y_en across all panel judges for a given gen model & qn_type.

    Returns dict: (prompt, country) -> (avg_Y_cn, avg_Y_en). Empty dict if no
    panel files exist.
    """
    panel_paths = sorted(base_dir.glob(f"{prompt_type}_{slug}_res_*.csv"))
    if not panel_paths:
        return {}
    frames = []
    for p in panel_paths:
        df = pd.read_csv(p)
        if "Y_cn" not in df.columns or "Y_en" not in df.columns:
            continue
        frames.append(df[["prompt", "country", "Y_cn", "Y_en"]])
    if not frames:
        return {}
    panel = pd.concat(frames, ignore_index=True)
    # Mean handles NaN by ignoring
    agg = panel.groupby(["prompt", "country"], as_index=False).agg(
        Y_cn=("Y_cn", "mean"), Y_en=("Y_en", "mean"))
    return {(r["prompt"], r["country"]): (r["Y_cn"], r["Y_en"])
            for _, r in agg.iterrows()}


def extract_responses(csv_path, model_name, prompt_type, en_to_zh, panel_lookup=None):
    """Extract native responses from a _res.csv file.

    If panel_lookup is provided (new-era models), use the panel-averaged
    Y_cn / Y_en for that (prompt, country) instead of the single-judge
    values on the row.
    """
    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        prompt_en = row["prompt"]
        country = row["country"]

        if (prompt_en, country) not in SELECTED_PROMPTS:
            continue

        ori1 = row["response_1_ori_lang"]

        # Extract native CN response, its EN translation, and native EN response
        if ori1 == "cn":
            cn_response = str(row["response_cn_1"])
            cn_translation = str(row["response_en_1"])
            en_response = str(row["response_en_2"])
        else:
            cn_response = str(row["response_cn_2"])
            cn_translation = str(row["response_en_2"])
            en_response = str(row["response_en_1"])

        # Skip empty responses
        if cn_response in ("", "nan") and en_response in ("", "nan"):
            continue

        def _num(v):
            try: return float(v)
            except (TypeError, ValueError): return None
        if panel_lookup and (prompt_en, country) in panel_lookup:
            y_cn, y_en = panel_lookup[(prompt_en, country)]
        else:
            y_cn, y_en = _num(row.get("Y_cn")), _num(row.get("Y_en"))
        ys = [y for y in (y_cn, y_en) if y is not None and not pd.isna(y)]
        if ys:
            avg = sum(ys) / len(ys)
            favorable = "cn" if avg > 0 else "en" if avg < 0 else "tie"
        else:
            favorable = None

        records.append({
            "prompt_en": prompt_en,
            "prompt_zh": en_to_zh.get(prompt_en, ""),
            "prompt_type": prompt_type,
            "country": country,
            "model": model_name,
            "response_cn": cn_response if cn_response != "nan" else "",
            "response_cn_translation": cn_translation if cn_translation != "nan" else "",
            "response_en": en_response if en_response != "nan" else "",
            "favorable": favorable,
        })

    return records


def main():
    en_to_zh = build_prompt_lookup()
    print(f"Built prompt lookup: {len(en_to_zh)} EN→ZH mappings")

    all_records = []

    for slug, name in PAPER_MODELS:
        for pt in ["country", "leader", "inst"]:
            path = PAPER_DIR / f"{pt}_{slug}_res.csv"
            if not path.exists():
                continue
            recs = extract_responses(path, name, pt, en_to_zh)
            all_records.extend(recs)
        n = sum(1 for r in all_records if r["model"] == name)
        print(f"  {name}: {n} responses")

    for slug, name in NEW_MODELS:
        for pt in ["country", "leader", "inst"]:
            path = NEW_DIR / f"{pt}_{slug}_res.csv"
            if not path.exists():
                continue
            panel_lookup = build_panel_lookup(NEW_DIR, slug, pt)
            recs = extract_responses(path, name, pt, en_to_zh, panel_lookup)
            all_records.extend(recs)
        n = sum(1 for r in all_records if r["model"] == name)
        print(f"  {name}: {n} responses" if n else f"  {name}: no data")

    # Build unique prompt list
    prompt_set = {}
    for r in all_records:
        key = (r["prompt_en"], r["country"])
        if key not in prompt_set:
            prompt_set[key] = {
                "id": len(prompt_set),
                "prompt_type": r["prompt_type"],
                "en": r["prompt_en"],
                "zh": r["prompt_zh"],
                "entity": r["country"],
            }

    prompts = list(prompt_set.values())

    # Map records to prompt IDs
    key_to_id = {(p["en"], p["entity"]): p["id"] for p in prompts}
    responses = []
    for r in all_records:
        pid = key_to_id[(r["prompt_en"], r["country"])]
        responses.append({
            "prompt_id": pid,
            "model": r["model"],
            "response_cn": r["response_cn"],
            "response_cn_translation": r["response_cn_translation"],
            "response_en": r["response_en"],
            "favorable": r.get("favorable"),
        })

    # Check size
    test_json = json.dumps(responses, ensure_ascii=False)
    size_mb = len(test_json.encode()) / 1e6
    print(f"\nTotal: {len(prompts)} prompts, {len(responses)} responses")
    print(f"Responses JSON size: {size_mb:.1f}MB")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "prompts.json", "w") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=None)
    with open(OUT_DIR / "responses.json", "w") as f:
        json.dump(responses, f, ensure_ascii=False, indent=None)

    print(f"\nWrote {OUT_DIR / 'prompts.json'} ({len(prompts)} prompts)")
    print(f"Wrote {OUT_DIR / 'responses.json'} ({len(responses)} responses)")


if __name__ == "__main__":
    main()
