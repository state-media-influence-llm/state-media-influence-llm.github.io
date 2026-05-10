"""Re-query memorization completions that returned empty content.

Empty completions (<10 chars) likely come from reasoning models consuming
the max_tokens=256 budget on hidden reasoning before emitting any final
content. This script re-queries those with max_tokens=2048 and updates
the completion record in place.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from env_utils import get_openrouter_client
from query_memorization import MODELS, fuzzy_match, is_refusal
from translate import translate_zh_to_en, _load_cache, _save_cache

BASE_DIR = Path(__file__).resolve().parent.parent
COMPLETIONS_PATH = BASE_DIR / "data" / "memorization" / "completions.json"
PHRASES_PATH = BASE_DIR / "data" / "memorization" / "phrases.json"

MAX_TOKENS = 2048
EMPTY_THRESHOLD = 10
WORKERS = 8


def query_with_budget(client, model_id: str, prompt: str, max_tokens: int) -> str:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"[ERROR] {e}"


def process_one(client, c, start_text, translation_cache, cache_lock, timestamp):
    """Query one completion and return updated record (in place)."""
    model_id = MODELS.get(c["model"])
    if not model_id:
        return c, False
    new_text = query_with_budget(client, model_id, c["prompt"], MAX_TOKENS)
    matched, edit_distance, match_start, match_end = fuzzy_match(
        new_text, c["expected"], prompt_start=start_text)
    refused = is_refusal(new_text)
    still_empty = len(new_text.strip()) < EMPTY_THRESHOLD

    c["completion"] = new_text
    c["matched"] = matched
    c["refused"] = refused
    c["edit_distance"] = round(edit_distance, 4)
    c["match_start"] = match_start
    c["match_end"] = match_end
    c["timestamp"] = timestamp
    c["requeried_max_tokens"] = MAX_TOKENS

    if not new_text.startswith("[ERROR]") and not still_empty:
        with cache_lock:
            c["completion_en"] = translate_zh_to_en(new_text, cache=translation_cache)

    return c, still_empty


def main():
    with open(COMPLETIONS_PATH, "r", encoding="utf-8") as f:
        completions = json.load(f)
    with open(PHRASES_PATH, "r", encoding="utf-8") as f:
        phrases = json.load(f)
    start_by_id = {p["id"]: p["start"] for p in phrases}

    empties = [
        c for c in completions
        if c.get("timestamp") != "paper"
        and len(c.get("completion", "").strip()) < EMPTY_THRESHOLD
        and c.get("model") in MODELS
    ]

    by_model = {}
    for c in empties:
        by_model[c["model"]] = by_model.get(c["model"], 0) + 1
    print(f"Empties to re-query: {len(empties)}")
    for m, n in sorted(by_model.items()):
        print(f"  {m}: {n}")

    client = get_openrouter_client()
    translation_cache = _load_cache()
    cache_lock = threading.Lock()
    save_lock = threading.Lock()
    timestamp = datetime.now(timezone.utc).isoformat()

    done_count = [0]
    still_empty_count = [0]

    def task(c):
        start_text = start_by_id.get(c["phrase_id"], "")
        _, still_empty = process_one(
            client, c, start_text, translation_cache, cache_lock, timestamp)
        with save_lock:
            done_count[0] += 1
            if still_empty:
                still_empty_count[0] += 1
            n = done_count[0]
            if n % 25 == 0 or n == len(empties):
                print(f"  [{n}/{len(empties)}] {c['model']} {c['phrase_id']} "
                      f"matched={c['matched']} len={len(c['completion'])} "
                      f"still_empty={still_empty}")
                tmp = str(COMPLETIONS_PATH) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(completions, f, ensure_ascii=False, indent=2)
                Path(tmp).replace(COMPLETIONS_PATH)
                _save_cache(dict(translation_cache))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(task, c) for c in empties]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"  task failed: {e}")

    tmp = str(COMPLETIONS_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(completions, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(COMPLETIONS_PATH)
    _save_cache(dict(translation_cache))

    print(f"\nDone! Re-queried {done_count[0]}, still empty after re-query: {still_empty_count[0]}")


if __name__ == "__main__":
    main()
