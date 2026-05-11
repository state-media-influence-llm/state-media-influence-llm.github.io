"""Cross-model audit: pairwise SUT comparisons holding language constant.

For each cycle-pair of SUTs (A, B), each prompt, each language (cn, en),
present (A's response, B's response) to the judge in randomized A/B order
and record which is more favorable to the focal entity. Bradley-Terry
fits on the resulting wins/losses (in R).

Cycle (each SUT in exactly two pairs):
  GPT-5.5 ↔ GPT-5.4 ↔ Opus 4.6 ↔ Opus 4.7 ↔ Gemini 3.1 Pro ↔ DeepSeek V3.2
  ↔ DeepSeek V4 Pro ↔ Grok 4 ↔ Grok 4.3 ↔ GPT-5.5 (closes)

Usage:
    # Smoke test
    python run_cross_model_audit.py --pairs gpt-5.5:deepseek-v4-pro --limit 10
    # Full run
    python run_cross_model_audit.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import time
from pathlib import Path

import pandas as pd

from env_utils import load_env_from_file
from run_judge_panel import (
    COUNTRY_DICT_CN, COUNTRY_DICT_EN,
    PROMPT_CN_TMPL, PROMPT_EN_TMPL, SYSTEM_PROMPT,
    judge_single,
)

BASE_DIR = Path(__file__).resolve().parent.parent
STUDY4_DIR = BASE_DIR / "data" / "study4"
OUT_DIR = BASE_DIR / "data" / "cross_model_audit"

# Display name -> filename slug used in data/study4/{qt}_{slug}.csv
SUT_SLUGS = {
    "gpt-5.4": "gpt-5-4",
    "gpt-5.5": "gpt-5-5",
    "claude-opus-4.6": "claude-opus-4-6",
    "claude-opus-4.7": "claude-opus-4-7",
    "gemini-3.1-pro": "gemini-3-1-pro",
    "deepseek-v3.2": "deepseek-v3-2-speciale",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "grok-4": "grok-4",
    "grok-4.3": "grok-4-3",
}

# Hamiltonian cycle on the 9 new-era SUTs
CYCLE_PAIRS = [
    ("gpt-5.5", "gpt-5.4"),
    ("gpt-5.4", "claude-opus-4.6"),
    ("claude-opus-4.6", "claude-opus-4.7"),
    ("claude-opus-4.7", "gemini-3.1-pro"),
    ("gemini-3.1-pro", "deepseek-v3.2"),
    ("deepseek-v3.2", "deepseek-v4-pro"),
    ("deepseek-v4-pro", "grok-4"),
    ("grok-4", "grok-4.3"),
    ("grok-4.3", "gpt-5.5"),
]

JUDGES = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
}

QN_TYPES = ["leader", "country", "inst"]

MAX_RETRIES = 6
RETRY_BASE_DELAY = 3
CONCURRENCY = 15


def load_sut_responses(model_name: str) -> pd.DataFrame:
    """Load all (qt, lang) response rows for one SUT into a single DataFrame.

    Returns columns: qn, prompt, language, country, response, prompt_type.
    'prompt' is the prompt in `language`; 'response' is the model's response
    in `language`.
    """
    slug = SUT_SLUGS[model_name]
    frames = []
    for qt in QN_TYPES:
        path = STUDY4_DIR / f"{qt}_{slug}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing gen CSV for {model_name}: {path}")
        df = pd.read_csv(path)
        cn = df[df["language"] == "cn"][["qn", "prompt", "country", "response_cn"]].copy()
        cn = cn.rename(columns={"response_cn": "response"})
        cn["language"] = "cn"
        en = df[df["language"] == "en"][["qn", "prompt", "country", "response_en"]].copy()
        en = en.rename(columns={"response_en": "response"})
        en["language"] = "en"
        cn["prompt_type"] = qt
        en["prompt_type"] = qt
        frames.append(pd.concat([cn, en], ignore_index=True))
    out = pd.concat(frames, ignore_index=True)
    return out


def build_pair_records(model_a: str, model_b: str,
                       resp_a: pd.DataFrame, resp_b: pd.DataFrame,
                       limit: int = 0, seed: int = 92092) -> list[dict]:
    """For each (qn, language), build the A/B comparison row."""
    rng = random.Random(seed)

    # Join by (qn, prompt_type, language) — qn values repeat across qt
    # (qn_0 exists in leader, country, and inst), so we MUST include
    # prompt_type to avoid cartesian explosion.
    key_cols = ["qn", "prompt_type", "language"]
    merged = resp_a.merge(
        resp_b[key_cols + ["response"]],
        on=key_cols, suffixes=("_a", "_b"))

    if limit > 0:
        # Sample limit/2 from each language so smoke tests cover both
        per_lang = max(1, limit // 2)
        cn_slice = merged[merged["language"] == "cn"].head(per_lang)
        en_slice = merged[merged["language"] == "en"].head(per_lang)
        merged = pd.concat([cn_slice, en_slice], ignore_index=True)

    records = []
    for _, row in merged.iterrows():
        # Position-randomize so model_a is sometimes Response A, sometimes B
        a_is_first = rng.random() < 0.5
        if a_is_first:
            r1, r2 = row["response_a"], row["response_b"]
        else:
            r1, r2 = row["response_b"], row["response_a"]

        records.append({
            "qn": row["qn"],
            "prompt": row["prompt"],
            "prompt_type": row["prompt_type"],
            "country": row["country"],
            "language": row["language"],
            "model_a": model_a,
            "model_b": model_b,
            "response_a": row["response_a"],
            "response_b": row["response_b"],
            "r1": r1,
            "r2": r2,
            "a_is_first": a_is_first,
        })
    return records


def build_judge_prompt(rec: dict) -> str:
    """Build the judge prompt using existing CN/EN templates."""
    if rec["language"] == "cn":
        tmpl = PROMPT_CN_TMPL
        entity = COUNTRY_DICT_CN[rec["country"]]
    else:
        tmpl = PROMPT_EN_TMPL
        entity = COUNTRY_DICT_EN[rec["country"]]
    return (tmpl
            .replace("[R1]", str(rec["r1"]))
            .replace("[R2]", str(rec["r2"]))
            .replace("[Entity]", entity)
            .replace("[QN]", str(rec["prompt"])))


def score_judgment(raw: str | None, a_is_first: bool) -> str | None:
    """Return 'model_a', 'model_b', or None (unparseable)."""
    if not raw:
        return None
    head = raw.strip()[:6].upper()
    # The judge can emit '(A)', 'A)', 'A.', 'A', etc.
    if "A" in head and "B" not in head:
        picked_first = True
    elif "B" in head and "A" not in head:
        picked_first = False
    else:
        return None
    # Map to model_a vs model_b
    if picked_first:
        return "model_a" if a_is_first else "model_b"
    else:
        return "model_b" if a_is_first else "model_a"


async def judge_pair(client, sem, judge_name: str, judge_model_id: str,
                     model_a: str, model_b: str,
                     records: list[dict]) -> Path:
    """Run all judge calls for one (pair, judge) combo and save CSV."""
    safe_a = model_a.replace(".", "")
    safe_b = model_b.replace(".", "")
    safe_j = judge_name.replace(".", "")
    out_path = OUT_DIR / f"{safe_a}_vs_{safe_b}_{safe_j}.csv"

    # Skip if already complete
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if len(existing) == len(records):
            print(f"  SKIP {model_a} vs {model_b} / {judge_name} — already complete ({len(existing)} rows)")
            return out_path

    n = len(records)
    print(f"  START {model_a} vs {model_b} / {judge_name}: {n} comparisons", flush=True)
    t0 = time.time()

    prompts = [build_judge_prompt(r) for r in records]
    progress = {"done": 0, "total": n, "t0": t0}

    async def _wrapped(i, p):
        result = await judge_single(client, sem, judge_model_id, SYSTEM_PROMPT, p, i)
        progress["done"] += 1
        d = progress["done"]
        if d % 200 == 0 or d == n:
            elapsed = time.time() - t0
            rate = d / elapsed if elapsed > 0 else 0
            eta = (n - d) / rate if rate > 0 else 0
            print(f"    progress: {d}/{n} ({100*d/n:.1f}%) "
                  f"rate={rate:.1f}/s ETA={eta/60:.0f}m", flush=True)
        return result

    tasks = [_wrapped(i, p) for i, p in enumerate(prompts)]
    raws = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    print(f"  DONE  {model_a} vs {model_b} / {judge_name} in {elapsed:.0f}s", flush=True)

    # Build output rows
    rows = []
    failed = 0
    for rec, raw in zip(records, raws):
        winner = score_judgment(raw, rec["a_is_first"])
        if raw is None or winner is None:
            failed += 1
        rows.append({
            "qn": rec["qn"],
            "prompt": rec["prompt"],
            "prompt_type": rec["prompt_type"],
            "country": rec["country"],
            "language": rec["language"],
            "model_a": rec["model_a"],
            "model_b": rec["model_b"],
            "a_is_first": rec["a_is_first"],
            "judge_raw": raw if raw is not None else "",
            "winner": winner if winner else "",
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    if failed:
        print(f"    WARNING: {failed}/{n} unparseable/failed for {model_a} vs {model_b} / {judge_name}")
    return out_path


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", nargs="+", choices=list(JUDGES.keys()),
                        default=list(JUDGES.keys()))
    parser.add_argument("--pairs", nargs="+", default=None,
                        help='Limit to specific pairs, format "modelA:modelB"')
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit prompts per pair × language (0 = all)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    pairs = CYCLE_PAIRS
    if args.pairs:
        pairs = []
        for p in args.pairs:
            a, b = p.split(":")
            pairs.append((a, b))
        for a, b in pairs:
            if a not in SUT_SLUGS or b not in SUT_SLUGS:
                raise ValueError(f"unknown model: {a} or {b}")

    load_env_from_file()
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    sem = asyncio.Semaphore(args.concurrency)

    # Preload SUT responses once
    all_models = sorted({m for pair in pairs for m in pair})
    print(f"Loading responses for {len(all_models)} SUTs...")
    sut_data = {m: load_sut_responses(m) for m in all_models}
    for m, df in sut_data.items():
        print(f"  {m}: {len(df)} rows")

    # Build all records once per pair
    pair_records = {}
    for a, b in pairs:
        recs = build_pair_records(a, b, sut_data[a], sut_data[b], limit=args.limit)
        pair_records[(a, b)] = recs
        print(f"  pair {a} vs {b}: {len(recs)} records")

    # Launch all (pair, judge) combos concurrently
    tasks = []
    for a, b in pairs:
        recs = pair_records[(a, b)]
        for judge_name in args.judges:
            tasks.append(judge_pair(client, sem, judge_name, JUDGES[judge_name],
                                     a, b, recs))

    print(f"\nLaunching {len(tasks)} (pair × judge) combos "
          f"with concurrency={args.concurrency}")
    await asyncio.gather(*tasks)
    print("\nAll done!")


if __name__ == "__main__":
    asyncio.run(main())
