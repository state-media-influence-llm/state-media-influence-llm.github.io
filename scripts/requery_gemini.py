"""Re-query Gemini 3.1 Pro with raw start prompt (no 续写句子 prefix).

Removes old Gemini entries and re-queries all 2000 phrases.
"""
from __future__ import annotations

import json
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
from datetime import datetime, timezone
from pathlib import Path

from query_memorization import query_one_model, clean_text, fuzzy_match
from env_utils import get_openrouter_client
from translate import _load_cache, _save_cache

BASE_DIR = Path(__file__).resolve().parent.parent
PHRASES_PATH = BASE_DIR / "data" / "memorization" / "phrases.json"
COMPLETIONS_PATH = BASE_DIR / "data" / "memorization" / "completions.json"

MODEL_NAME = "gemini-3.1-pro"
MODEL_ID = "google/gemini-3.1-pro-preview"


def main():
    client = get_openrouter_client()

    with open(PHRASES_PATH) as f:
        all_phrases = json.load(f)

    propaganda = [p for p in all_phrases if p["type"] == "propaganda"][:1000]
    culturax = [p for p in all_phrases if p["type"] == "culturax"][:1000]
    phrases = propaganda + culturax

    # Load completions, remove old Gemini entries (non-paper)
    with open(COMPLETIONS_PATH) as f:
        completions = json.load(f)

    old_gemini = sum(1 for c in completions
                     if c.get("model") == MODEL_NAME and c.get("timestamp") != "paper")
    completions = [c for c in completions
                   if not (c.get("model") == MODEL_NAME and c.get("timestamp") != "paper")]
    print(f"Removed {old_gemini} old Gemini entries, {len(completions)} remaining")

    # Check what's already done (in case of resume)
    done = set(c["phrase_id"] for c in completions if c.get("model") == MODEL_NAME)
    remaining = [p for p in phrases if p["id"] not in done]
    print(f"Gemini: {len(remaining)} phrases to query")

    translation_cache = _load_cache()
    timestamp = datetime.now(timezone.utc).isoformat()
    count = 0

    for phrase in remaining:
        # Use 续写句子 prefix (same as other new models)
        prompt_text = phrase.get("start_chat") or f"续写句子：{phrase['start']}"
        try:
            rec = query_one_model(client, MODEL_NAME, MODEL_ID,
                                  phrase, prompt_text, translation_cache)
            rec["timestamp"] = timestamp
            completions.append(rec)
            count += 1
            if count % 50 == 0:
                print(f"  [{count}/{len(remaining)}] matched={rec['matched']} dist={rec['edit_distance']}")
                # Checkpoint
                tmp = str(COMPLETIONS_PATH) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(completions, f, ensure_ascii=False, indent=2)
                Path(tmp).replace(COMPLETIONS_PATH)
                _save_cache(translation_cache)
        except Exception as e:
            print(f"  FAILED {phrase['id']}: {e}")

        time.sleep(0.3)

    # Final save
    tmp = str(COMPLETIONS_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(completions, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(COMPLETIONS_PATH)
    _save_cache(translation_cache)

    # Summary
    gemini = [c for c in completions if c["model"] == MODEL_NAME
              and c.get("timestamp") != "paper"]
    for ptype in ["propaganda", "culturax"]:
        subset = [c for c in gemini if c["type"] == ptype]
        refused = sum(1 for c in subset if c.get("refused"))
        nr = [c for c in subset if not c.get("refused")]
        matched = sum(1 for c in nr if c["matched"])
        total = len(nr)
        rate = matched / total if total else 0
        print(f"gemini-3.1-pro {ptype}: {matched}/{total} ({rate:.1%}) [refused: {refused}]")

    print(f"Done! {count} new queries.")


if __name__ == "__main__":
    main()
