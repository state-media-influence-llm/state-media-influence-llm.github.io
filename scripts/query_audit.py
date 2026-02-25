"""
Query production LLMs with politically sensitive prompts via OpenRouter.

Queries 10 prompts × 7 models × 2 languages = 140 API calls per run.
Appends timestamped results to data/audit/responses.json.
Adds English translations of Chinese responses.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from env_utils import get_openrouter_client
from translate import translate_zh_to_en, _load_cache, _save_cache

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_PATH = BASE_DIR / "data" / "audit" / "prompts.json"
RESPONSES_PATH = BASE_DIR / "data" / "audit" / "responses.json"

MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "deepseek-chat": "deepseek/deepseek-chat",
    "gpt-4.1": "openai/gpt-4.1",
    "claude-opus-4.6": "anthropic/claude-opus-4-6",
}

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


def query_model(client, model_id: str, prompt: str) -> str:
    """Query a model with retry and exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} for {model_id}: {e}")
                time.sleep(delay)
            else:
                return f"[ERROR] {e}"


def main():
    client = get_openrouter_client()

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Load existing responses
    if RESPONSES_PATH.exists():
        with open(RESPONSES_PATH, "r", encoding="utf-8") as f:
            responses = json.load(f)
    else:
        responses = []

    translation_cache = _load_cache()
    timestamp = datetime.now(timezone.utc).isoformat()
    total = len(prompts) * len(MODELS) * 2
    count = 0

    for prompt in prompts:
        for model_name, model_id in MODELS.items():
            for lang, text in [("en", prompt["en"]), ("zh", prompt["zh"])]:
                count += 1
                print(f"[{count}/{total}] {model_name} | prompt {prompt['id']} | {lang}")

                response_text = query_model(client, model_id, text)

                entry = {
                    "timestamp": timestamp,
                    "prompt_id": prompt["id"],
                    "category": prompt["category"],
                    "entity": prompt["entity"],
                    "language": lang,
                    "prompt_text": text,
                    "model": model_name,
                    "response": response_text,
                }

                # Translate Chinese responses to English
                if lang == "zh" and not response_text.startswith("[ERROR]"):
                    entry["response_en_translation"] = translate_zh_to_en(
                        response_text, cache=translation_cache
                    )

                responses.append(entry)

                # Brief pause to avoid rate limits
                time.sleep(0.5)

    # Save translation cache and responses
    _save_cache(translation_cache)

    with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {count} queries completed. Total responses: {len(responses)}")


if __name__ == "__main__":
    main()
