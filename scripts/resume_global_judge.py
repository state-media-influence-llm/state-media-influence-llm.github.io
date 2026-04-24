#!/usr/bin/env python3
"""Resume a partially-filled global judge CSV by retrying only empty cells.

Reads an existing judge output file, identifies rows where eng_out or target_out
is empty, rebuilds the A/B pairs with the same seed as run_global_judges.py so
randomization matches, and issues only the missing judge calls. Writes the
updated CSV in place when finished.

Usage:
    python resume_global_judge.py --gen claude-opus-4.7 --judge deepseek-v3.2
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_utils import load_env_from_file
from run_global_judges import (
    JUDGES, GEN_MODELS, JUDGE_DIR, JUDGE_TMPL, SYSTEM_PROMPT,
    load_gen_data, build_ab_pairs, judge_single, score_judgment,
)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", required=True, choices=list(GEN_MODELS.keys()))
    parser.add_argument("--judge", required=True, choices=list(JUDGES.keys()))
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max concurrent calls (default 5, lower than normal run to avoid 429s)")
    args = parser.parse_args()

    load_env_from_file()
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    gen_slug = args.gen.replace(".", "")
    judge_slug = args.judge.replace(".", "")
    csv_path = JUDGE_DIR / f"{gen_slug}_{judge_slug}.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} does not exist")
        return

    # Load existing rows
    with open(csv_path, newline="", encoding="utf-8") as f:
        existing = list(csv.DictReader(f))
    print(f"Loaded {len(existing)} existing rows from {csv_path.name}")

    # Rebuild pairs with the same seed as run_global_judges
    rows = load_gen_data(args.gen)
    pairs = build_ab_pairs(rows)
    if len(pairs) != len(existing):
        print(f"ERROR: pair count {len(pairs)} != existing row count {len(existing)}")
        return

    # Identify missing cells
    missing_eng = []  # list of (idx, pair)
    missing_tgt = []
    for i, (pair, row) in enumerate(zip(pairs, existing)):
        if not row.get("eng_out") or row["eng_out"] == "":
            missing_eng.append((i, pair))
        if not row.get("target_out") or row["target_out"] == "":
            missing_tgt.append((i, pair))

    n_missing = len(missing_eng) + len(missing_tgt)
    if n_missing == 0:
        print("Nothing to retry — all cells populated.")
        return

    print(f"Missing: {len(missing_eng)} eng + {len(missing_tgt)} tgt = {n_missing} calls")

    judge_model_id = JUDGES[args.judge]
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    progress = {"done": 0, "total": n_missing, "t0": t0}

    # Build coroutines
    eng_tasks = [
        judge_single(client, sem, judge_model_id, SYSTEM_PROMPT,
                     JUDGE_TMPL.format(r1=p["eng_r1"], r2=p["eng_r2"],
                                       qn=p["eng_qn"], entity=p["entity"]),
                     i, progress)
        for i, p in missing_eng
    ]
    tgt_tasks = [
        judge_single(client, sem, judge_model_id, SYSTEM_PROMPT,
                     JUDGE_TMPL.format(r1=p["tgt_r1"], r2=p["tgt_r2"],
                                       qn=p["tgt_qn"], entity=p["entity"]),
                     i + len(existing), progress)
        for i, p in missing_tgt
    ]

    results = await asyncio.gather(*eng_tasks, *tgt_tasks)
    eng_results = results[:len(missing_eng)]
    tgt_results = results[len(missing_eng):]

    elapsed = time.time() - t0
    print(f"Retry complete in {elapsed:.0f}s")

    # Merge new scores into existing rows
    filled_eng = 0
    filled_tgt = 0
    for (idx, pair), raw in zip(missing_eng, eng_results):
        score = score_judgment(raw, pair["eng_a_is_target"])
        if score is not None:
            existing[idx]["eng_out"] = score
            filled_eng += 1
    for (idx, pair), raw in zip(missing_tgt, tgt_results):
        score = score_judgment(raw, pair["tgt_a_is_target"])
        if score is not None:
            existing[idx]["target_out"] = score
            filled_tgt += 1

    print(f"Filled: {filled_eng}/{len(missing_eng)} eng, "
          f"{filled_tgt}/{len(missing_tgt)} tgt")

    # Write back
    fieldnames = list(existing[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
