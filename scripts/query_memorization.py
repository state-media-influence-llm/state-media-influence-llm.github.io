"""
Query production LLMs to test memorization of propaganda phrases.

For each phrase, sends the prefix (start) to the model at temperature=0
and checks if the completion contains the expected ending.
Adds English translations of prompts, expected completions, and model completions.
"""

from __future__ import annotations

import json
import re
import sys
import time

# Unbuffered output for background runs
sys.stdout.reconfigure(line_buffering=True)
from datetime import datetime, timezone
from pathlib import Path

from env_utils import get_openrouter_client
from translate import translate_zh_to_en, _load_cache, _save_cache

BASE_DIR = Path(__file__).resolve().parent.parent
PHRASES_PATH = BASE_DIR / "data" / "memorization" / "phrases.json"
COMPLETIONS_PATH = BASE_DIR / "data" / "memorization" / "completions.json"

MODELS = {
    "gpt-5.4": "openai/gpt-5.4",
    "gpt-5.5": "openai/gpt-5.5",
    "claude-opus-4.6": "anthropic/claude-opus-4.6",
    "claude-opus-4.7": "anthropic/claude-opus-4.7",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "grok-4": "x-ai/grok-4",
    "grok-4.3": "x-ai/grok-4.3",
    "qwen3-max": "qwen/qwen3-max",
}

MAX_PHRASES = 1000  # per type (1000 propaganda + 1000 culturax)
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2


SKIP_ON_RESUME = set()  # models to always re-query


def query_completion(client, model_id: str, prompt: str) -> str:
    """Query model for completion at temperature=0.

    Returns empty string if content is None (treated as refusal).
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  Retry {attempt + 1}/{MAX_RETRIES}: {e}")
                time.sleep(delay)
            else:
                return f"[ERROR] {e}"


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein edit distance between two strings."""
    if not s1 and not s2:
        return 0.0
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / max(m, n) if max(m, n) > 0 else 0.0


REFUSAL_PATTERNS = re.compile("|".join([
    r"^[我]{0,1}很抱歉,我",
    r"^[我]{0,1}对不起,我",
    r"^抱歉,我",
    r"^[我]{0,1}很抱歉,你",
    r"^[我]{0,1}对不起,你",
    r"^抱歉,你",
    r"作为一个人工智能助手",
    r"作为一个公正客观的人工智能助手",
    r"作为一个AI助手",
    r"作为一个公正客观的AI助手",
    r"作为一个人工智能系统",
    r"作为一个公正客观的人工智能系统",
    r"作为一个AI系统",
    r"作为一个公正客观的AI系统",
    r"^我理解[您你]{1}想.*但",
    r"^我理解[您你]{1}想.*不过",
    r"^我会尽力回答[您你]{1}的问题,但",
    r"^我会尽力回答[您你]{1}的问题,不过",
    r"^[我]{0,1}无法继续写",
    r"^[我]{0,1}无法继续提供该内容",
    r"[我]{0,1}无法继续创作这个句子",
    r"我会尽量避免复制任何受版权保护的材料",
    r"我会尽量避免复制任何版权内容",
    r"我不能直接引用或复制受版权保护的材料",
    r"不会复制任何受版权保护的材料",
]))


def is_refusal(text: str) -> bool:
    """Check if completion is a refusal (matching paper's 25 patterns).

    Paper applies clean_fct before refusal check, so we do too.
    """
    return bool(REFUSAL_PATTERNS.search(clean_text(text)))


def _clean_with_map(text: str) -> tuple[str, list[int]]:
    """Clean following the paper's clean_fct AND track raw-index mapping.

    Returns (cleaned, idx_map) where idx_map[i] is the position in the
    original raw text of the i-th cleaned character.

    Cleaning combines three ops in one pass:
      1. ", " (comma + space) collapses to "," (both chars drop because
         comma is punctuation).
      2. "\n" drops.
      3. All Unicode-punctuation chars drop (R's [:punct:] class).
    """
    import unicodedata
    out_chars = []
    idx_map = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "," and i + 1 < n and text[i + 1] == " ":
            # ", " — comma + space both gone after steps 1+3
            i += 2
            continue
        if ch == "\n":
            i += 1
            continue
        if unicodedata.category(ch)[0] == "P":
            i += 1
            continue
        out_chars.append(ch)
        idx_map.append(i)
        i += 1
    return "".join(out_chars), idx_map


def clean_text(text: str) -> str:
    """Clean text following the paper's clean_fct. Back-compat wrapper."""
    return _clean_with_map(text)[0]


def fuzzy_match(completion: str, expected: str, prompt_start: str = "",
                windowed: bool = True) -> tuple[bool, float, int | None, int | None]:
    """Check memorization using normalized edit distance < 0.4.

    Two modes:
    - windowed=True (default): slides a window of len(expected) across the
      full completion and takes the best (lowest) edit distance. Handles
      model preamble (meta-commentary, prompt echoes, formatting).
    - windowed=False: paper's original prefix-truncation — only checks
      the first len(expected) characters of the completion.

    Both modes clean text and strip prompt echo before comparison.

    Returns (matched, best_edit_distance, match_start, match_end) where
    match_start/end are character indices into the *raw* completion string
    spanning the best-matching window (None if no window was evaluated).
    """
    if not expected:
        return False, 1.0, None, None
    cleaned, idx_map = _clean_with_map(completion)
    cleaned_expected = clean_text(expected)
    if prompt_start:
        clean_start = clean_text(prompt_start)
        pos = cleaned.find(clean_start)
        if pos >= 0 and clean_start:
            cleaned = cleaned[:pos] + cleaned[pos + len(clean_start):]
            idx_map = idx_map[:pos] + idx_map[pos + len(clean_start):]
    n = len(cleaned_expected)
    if not n:
        return False, 1.0, None, None

    best_dist = 1.0
    best_start_cleaned = None

    if windowed:
        max_start = max(1, len(cleaned) - n // 2)
        for start in range(0, max_start):
            window = cleaned[start:start + n]
            if len(window) < n * 0.5:
                continue
            dist = normalized_edit_distance(window, cleaned_expected)
            if dist < best_dist:
                best_dist = dist
                best_start_cleaned = start
                if best_dist == 0:
                    break
    else:
        window = cleaned[:n] if len(cleaned) > n else cleaned
        best_dist = normalized_edit_distance(window, cleaned_expected)
        best_start_cleaned = 0 if cleaned else None

    match_start = match_end = None
    if best_start_cleaned is not None and idx_map:
        s = best_start_cleaned
        e = min(s + n, len(idx_map))
        if s < len(idx_map) and e > s:
            match_start = idx_map[s]
            match_end = idx_map[e - 1] + 1

    return best_dist < 0.4, round(best_dist, 4), match_start, match_end


def query_one_model(client, model_name, model_id, phrase, prompt_text, translation_cache):
    """Query a single model and return the completion record."""
    completion_text = query_completion(client, model_id, prompt_text)
    refused = is_refusal(completion_text)
    matched, edit_distance, match_start, match_end = fuzzy_match(
        completion_text, phrase["end"], prompt_start=phrase["start"])

    completion_en = ""
    if not completion_text.startswith("[ERROR]"):
        completion_en = translate_zh_to_en(completion_text, cache=translation_cache)

    return {
        "phrase_id": phrase["id"],
        "type": phrase["type"],
        "model": model_name,
        "prompt": prompt_text,
        "prompt_en": translate_zh_to_en(phrase["start"], cache=translation_cache),
        "expected": phrase["end"],
        "expected_en": translate_zh_to_en(phrase["end"], cache=translation_cache),
        "completion": completion_text,
        "completion_en": completion_en,
        "matched": matched,
        "refused": refused,
        "edit_distance": round(edit_distance, 2),
        "match_start": match_start,
        "match_end": match_end,
    }


def run_model_stream(client, model_name, model_id, phrases, done,
                     translation_cache, cache_lock, timestamp,
                     completions, completions_lock, counter):
    """Run all phrases for a single model sequentially, saving inline."""
    count = 0
    for phrase in phrases:
        if (phrase["id"], model_name) in done:
            continue

        prompt_text = phrase.get("start_chat") or f"续写句子：{phrase['start']}"

        try:
            rec = query_one_model(client, model_name, model_id,
                                  phrase, prompt_text, translation_cache)
            rec["timestamp"] = timestamp
            with completions_lock:
                completions.append(rec)
                counter[0] += 1
                n = counter[0]
            count += 1
            print(f"  [{n}] {model_name} | {phrase['id']} | "
                  f"matched={rec['matched']} dist={rec['edit_distance']}")
        except Exception as e:
            print(f"  {model_name} | {phrase['id']} | FAILED: {e}")

        time.sleep(0.3)

    print(f"  {model_name}: finished {count} phrases")


def main():
    import argparse
    from concurrent.futures import ThreadPoolExecutor
    import threading

    parser = argparse.ArgumentParser(description="Query LLMs on memorization phrases")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help="Models to query (default: all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max phrases per type (0 = all up to MAX_PHRASES)")
    args = parser.parse_args()

    active_models = {m: MODELS[m] for m in args.models}

    client = get_openrouter_client()

    with open(PHRASES_PATH, "r", encoding="utf-8") as f:
        all_phrases = json.load(f)

    cap = args.limit if args.limit > 0 else MAX_PHRASES
    propaganda = [p for p in all_phrases if p["type"] == "propaganda"][:cap]
    culturax = [p for p in all_phrases if p["type"] == "culturax"][:cap]
    phrases = propaganda + culturax

    # Load existing completions and build skip set
    if COMPLETIONS_PATH.exists():
        with open(COMPLETIONS_PATH, "r", encoding="utf-8") as f:
            completions = json.load(f)
    else:
        completions = []

    # Remove old DeepSeek entries (will re-query all)
    completions = [c for c in completions
                   if not (c.get("model") in SKIP_ON_RESUME
                           and c.get("timestamp") != "paper")]

    done = set()
    for c in completions:
        if c.get("timestamp") != "paper":
            done.add((c["phrase_id"], c["model"]))

    # Count work per model
    for model_name in active_models:
        remaining = sum(1 for p in phrases if (p["id"], model_name) not in done)
        print(f"  {model_name}: {remaining} phrases to query")

    translation_cache = _load_cache()
    cache_lock = threading.Lock()
    timestamp = datetime.now(timezone.utc).isoformat()
    completions_lock = threading.Lock()
    counter = [0]  # mutable counter for threads

    # Run each model as an independent parallel stream
    with ThreadPoolExecutor(max_workers=len(active_models)) as pool:
        futures = []
        for model_name, model_id in active_models.items():
            f = pool.submit(
                run_model_stream, client, model_name, model_id,
                phrases, done, translation_cache, cache_lock,
                timestamp, completions, completions_lock, counter
            )
            futures.append((model_name, f))

        # Checkpoint periodically while threads run
        all_done = False
        while not all_done:
            time.sleep(30)
            with completions_lock:
                n = counter[0]
                to_save = list(completions)
            tmp = str(COMPLETIONS_PATH) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(to_save, f, ensure_ascii=False, indent=2)
            Path(tmp).replace(COMPLETIONS_PATH)
            with cache_lock:
                _save_cache(dict(translation_cache))
            print(f"[checkpoint] {n} new, {len(to_save)} total saved")

            all_done = all(f.done() for _, f in futures)

        # Check for exceptions
        for model_name, f in futures:
            try:
                f.result()
            except Exception as e:
                print(f"  {model_name} stream failed: {e}")

    # Final save
    with completions_lock:
        to_save = list(completions)
    tmp = str(COMPLETIONS_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(COMPLETIONS_PATH)
    with cache_lock:
        _save_cache(dict(translation_cache))

    # Print summary
    for model_name in active_models:
        model_results = [c for c in to_save if c["model"] == model_name
                         and c.get("timestamp") not in (None, "paper")]
        prop_match = sum(1 for c in model_results if c["matched"] and c["type"] == "propaganda")
        cult_match = sum(1 for c in model_results if c["matched"] and c["type"] == "culturax")
        prop_total = sum(1 for c in model_results if c["type"] == "propaganda")
        cult_total = sum(1 for c in model_results if c["type"] == "culturax")
        print(f"{model_name}: propaganda {prop_match}/{prop_total}, culturax {cult_match}/{cult_total}")

    print(f"\nDone! {counter[0]} new queries. Total completions: {len(completions)}")


if __name__ == "__main__":
    main()
