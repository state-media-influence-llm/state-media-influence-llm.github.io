#!/usr/bin/env python3
"""Translate contamination example articles from Chinese to English.

Input:  data/contamination/examples_raw.json
Output: data/contamination/examples.json (with culturax_text_en field added)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from translate import translate_zh_to_en, _load_cache, _save_cache

INPUT = os.path.join(os.path.dirname(__file__), "..", "data", "contamination", "examples_raw.json")
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "contamination", "examples.json")

with open(INPUT, "r", encoding="utf-8") as f:
    examples = json.load(f)

print(f"Translating {len(examples)} examples...")
cache = _load_cache()

for i, ex in enumerate(examples):
    text = ex["culturax_text"]
    # Truncate to first 1500 chars for translation (keeps cost/time reasonable)
    text_truncated = text[:1500]

    print(f"  [{i+1}/{len(examples)}] {ex['keyword_label']}: {text_truncated[:40]}...")
    ex["culturax_text_en"] = translate_zh_to_en(text_truncated, cache=cache)

    # Rate limit: ~1 req/sec to avoid Google Translate throttling
    if i % 5 == 4:
        _save_cache(cache)
        time.sleep(0.5)

_save_cache(cache)

# Write final output
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)

print(f"Saved {len(examples)} translated examples to {OUTPUT}")
