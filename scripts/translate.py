"""Translation utility: Chinese → English using deep-translator with caching."""

import json
import os
import hashlib

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "translations_cache.json")


def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CACHE_PATH)


def translate_zh_to_en(text, cache=None):
    """Translate Chinese text to English. Returns original text if translation fails."""
    if not text or not text.strip():
        return ""

    if cache is None:
        cache = _load_cache()

    key = hashlib.md5(text.encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]

    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source="zh-CN", target="en").translate(text)
        cache[key] = result
        return result
    except Exception as e:
        print(f"Translation failed for '{text[:50]}...': {e}")
        return text


def translate_batch(texts, save=True):
    """Translate a list of Chinese texts, returning list of English translations."""
    cache = _load_cache()
    results = []
    for text in texts:
        results.append(translate_zh_to_en(text, cache=cache))
    if save:
        _save_cache(cache)
    return results


if __name__ == "__main__":
    # Quick test
    test = "习近平是一个优秀的领导人吗？"
    print(f"Input:  {test}")
    print(f"Output: {translate_zh_to_en(test)}")
    _save_cache(_load_cache())
