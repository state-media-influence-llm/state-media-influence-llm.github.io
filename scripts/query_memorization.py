"""
Query production LLMs to test memorization of propaganda phrases.

For each phrase, sends the prefix (start) to the model at temperature=0
and checks if the completion contains the expected ending.
Adds English translations of prompts, expected completions, and model completions.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from env_utils import get_openrouter_client
from translate import translate_zh_to_en, _load_cache, _save_cache

BASE_DIR = Path(__file__).resolve().parent.parent
PHRASES_PATH = BASE_DIR / "data" / "memorization" / "phrases.json"
COMPLETIONS_PATH = BASE_DIR / "data" / "memorization" / "completions.json"

MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "deepseek-chat": "deepseek/deepseek-chat",
    "gpt-4.1": "openai/gpt-4.1",
    "claude-opus-4.6": "anthropic/claude-opus-4-6",
}

MAX_PHRASES = 50  # per type (50 propaganda + 50 culturax)
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2


def query_completion(client, model_id: str, prompt: str) -> str:
    """Query model for completion at temperature=0."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  Retry {attempt + 1}/{MAX_RETRIES}: {e}")
                time.sleep(delay)
            else:
                return f"[ERROR] {e}"


def fuzzy_match(completion: str, expected: str) -> bool:
    """Check if completion contains the expected text (fuzzy)."""
    norm_completion = completion.replace(" ", "").replace("，", "").replace("。", "")
    norm_expected = expected.replace(" ", "").replace("，", "").replace("。", "")
    return norm_expected in norm_completion


def main():
    client = get_openrouter_client()

    with open(PHRASES_PATH, "r", encoding="utf-8") as f:
        all_phrases = json.load(f)

    # Take top N per type
    propaganda = [p for p in all_phrases if p["type"] == "propaganda"][:MAX_PHRASES]
    culturax = [p for p in all_phrases if p["type"] == "culturax"][:MAX_PHRASES]
    phrases = propaganda + culturax

    # Load existing completions
    if COMPLETIONS_PATH.exists():
        with open(COMPLETIONS_PATH, "r", encoding="utf-8") as f:
            completions = json.load(f)
    else:
        completions = []

    translation_cache = _load_cache()
    timestamp = datetime.now(timezone.utc).isoformat()
    total = len(phrases) * len(MODELS)
    count = 0

    for phrase in phrases:
        prompt_text = phrase["start_chat"]
        if not prompt_text:
            prompt_text = f"续写句子：{phrase['start']}"

        # Pre-translate the prompt and expected completion
        prompt_en = translate_zh_to_en(phrase["start"], cache=translation_cache)
        expected_en = translate_zh_to_en(phrase["end"], cache=translation_cache)

        for model_name, model_id in MODELS.items():
            count += 1
            print(f"[{count}/{total}] {model_name} | {phrase['id']}")

            completion_text = query_completion(client, model_id, prompt_text)
            matched = fuzzy_match(completion_text, phrase["end"])

            # Translate model completion
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
            })

            time.sleep(0.5)

    _save_cache(translation_cache)

    with open(COMPLETIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(completions, f, ensure_ascii=False, indent=2)

    # Print summary
    for model_name in MODELS:
        model_results = [c for c in completions if c["model"] == model_name and c["timestamp"] == timestamp]
        prop_match = sum(1 for c in model_results if c["matched"] and c["type"] == "propaganda")
        cult_match = sum(1 for c in model_results if c["matched"] and c["type"] == "culturax")
        prop_total = sum(1 for c in model_results if c["type"] == "propaganda")
        cult_total = sum(1 for c in model_results if c["type"] == "culturax")
        print(f"{model_name}: propaganda {prop_match}/{prop_total}, culturax {cult_match}/{cult_total}")

    print(f"\nDone! {count} queries. Total completions: {len(completions)}")


if __name__ == "__main__":
    main()
