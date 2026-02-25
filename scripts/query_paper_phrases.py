"""Query live models for paper-only phrases (paper_1 through paper_11 + id_87)."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_utils import get_openrouter_client
from query_memorization import normalized_edit_distance, fuzzy_match
from translate import translate_zh_to_en, _load_cache, _save_cache

BASE_DIR = Path(__file__).resolve().parent.parent
PHRASES_PATH = BASE_DIR / "data" / "memorization" / "phrases.json"
COMPLETIONS_PATH = BASE_DIR / "data" / "memorization" / "completions.json"

MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "gpt-5.2": "openai/gpt-5.2",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "claude-opus-4.6": "anthropic/claude-opus-4-6",
    "deepseek-chat": "deepseek/deepseek-chat",
}

# Paper phrases that need live data
PAPER_PHRASE_IDS = {
    "id_87", "paper_1", "paper_2", "paper_3", "paper_4", "paper_5",
    "paper_6", "paper_7", "paper_8", "paper_9", "paper_10", "paper_11",
}


def query_completion(client, model_id: str, prompt: str) -> str:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (2 ** attempt))
                print(f"  Retry: {e}")
            else:
                return f"[ERROR] {e}"


def main():
    client = get_openrouter_client()

    with open(PHRASES_PATH, "r", encoding="utf-8") as f:
        all_phrases = json.load(f)

    with open(COMPLETIONS_PATH, "r", encoding="utf-8") as f:
        completions = json.load(f)

    # Filter to paper phrases
    phrases = [p for p in all_phrases if p["id"] in PAPER_PHRASE_IDS]
    print(f"Found {len(phrases)} paper phrases to query")

    # Check which already have live data
    existing_live = set()
    for c in completions:
        if c.get("timestamp") != "paper":
            existing_live.add((c["phrase_id"], c["model"]))

    translation_cache = _load_cache()
    timestamp = datetime.now(timezone.utc).isoformat()
    count = 0
    new_count = 0

    for phrase in phrases:
        prompt_text = phrase.get("start_chat") or f"续写句子：{phrase['start']}"
        prompt_en = translate_zh_to_en(phrase["start"], cache=translation_cache)
        expected_en = translate_zh_to_en(phrase["end"], cache=translation_cache)

        for model_name, model_id in MODELS.items():
            count += 1
            if (phrase["id"], model_name) in existing_live:
                print(f"[{count}] SKIP {model_name} | {phrase['id']} (already exists)")
                continue

            print(f"[{count}] {model_name} | {phrase['id']}")
            completion_text = query_completion(client, model_id, prompt_text)
            matched, edit_distance = fuzzy_match(completion_text, phrase["end"])

            completion_en = ""
            if not completion_text.startswith("[ERROR]"):
                completion_en = translate_zh_to_en(completion_text, cache=translation_cache)

            completions.append({
                "timestamp": timestamp,
                "phrase_id": phrase["id"],
                "type": phrase["type"],
                "model": model_name,
                "prompt": prompt_text,
                "prompt_en": prompt_en,
                "expected": phrase["end"],
                "expected_en": expected_en,
                "completion": completion_text,
                "completion_en": completion_en,
                "matched": matched,
                "edit_distance": round(edit_distance, 2),
            })
            new_count += 1
            time.sleep(0.3)

    _save_cache(translation_cache)

    with open(COMPLETIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(completions, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {new_count} new queries added. Total completions: {len(completions)}")


if __name__ == "__main__":
    main()
