#!/usr/bin/env python3
"""Extract example propaganda articles for the contamination page.

Uses the propaganda training corpus from Study 3 (p_news.json) which contains
41,517 scripted state media articles — the same corpus used to match against
CulturaX in Study 1. Samples articles tagged with each keyword category.

Output: examples_raw.json
"""

import json
import os
import random

random.seed(42)

BASE_DIR = "/scratch/sm11792/propaganda_llm"
PROP_FILE = os.path.join(BASE_DIR, "ptrain_exp/data/p_news.json")
NEWS_FILE = os.path.join(BASE_DIR, "ptrain_exp/data/n_news.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "code_public/study1_culturax/data/examples_raw.json")
EXAMPLES_PER_KEYWORD = 8

# ── Keyword definitions (from 01_match_culturax_scripted.py) ──
KEYWORDS = {
    "xjp":     {"zh": "习近平",     "label": "Xi Jinping",              "type": "Leaders"},
    "mzd":     {"zh": "毛泽东",     "label": "Mao Zedong",             "type": "Leaders"},
    "dxp":     {"zh": "邓小平",     "label": "Deng Xiaoping",          "type": "Leaders"},
    "party":   {"zh": "中国共产党",  "label": "Communist Party",         "type": "Institutions"},
    "npc":     {"zh": "人民代表大会", "label": "Natl People's Congress",  "type": "Institutions"},
    "plenum":  {"zh": "中央委员会全体会议", "label": "CCP Plenum",      "type": "Institutions"},
    "economy": {"zh": "经济发展",    "label": "Economy/Development",     "type": "Institutions"},
    "foreign": {"zh": "外交部发言人", "label": "Foreign Ministry",       "type": "Institutions"},
}


def detect_keywords(text):
    """Detect which keywords are present in text."""
    flags = {}
    flags["xjp"] = "习近平" in text
    flags["mzd"] = "毛泽东" in text
    flags["dxp"] = "邓小平" in text
    flags["party"] = "中国" in text and "共产党" in text
    flags["npc"] = "人民代表大会" in text or "人大" in text
    flags["plenum"] = "共产党" in text and "中央委员会" in text and "全体会议" in text
    flags["economy"] = "经济" in text and ("社会" in text or "发展" in text)
    flags["foreign"] = "外交部" in text and "发言人" in text
    return flags


def is_good_example(text):
    """Filter for substantive articles."""
    if len(text) < 300 or len(text) > 4000:
        return False
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if chinese_chars / max(len(text), 1) < 0.4:
        return False
    return True


# ── Load propaganda articles ──
print("Loading propaganda articles...")
with open(PROP_FILE) as f:
    prop_data = json.load(f)
print(f"  Loaded {len(prop_data)} propaganda articles")

# Also load state-run news (non-scripted) for contrast
print("Loading state-run news (non-scripted)...")
with open(NEWS_FILE) as f:
    news_data = json.load(f)
print(f"  Loaded {len(news_data)} news articles")

# ── Collect candidates per keyword ──
candidates = {kw: [] for kw in KEYWORDS}

for i, article in enumerate(prop_data):
    text = article["text"]
    if not is_good_example(text):
        continue
    kw_flags = detect_keywords(text)
    matched_kws = [k for k, v in kw_flags.items() if v]
    for kw in matched_kws:
        if kw in KEYWORDS:
            candidates[kw].append({
                "text": text[:2500],
                "corpus": "scripted",
                "article_idx": i,
            })

# ── Sample and format ──
print("\nSampling examples per keyword...")
sampled = []
seen_idx = set()

for kw_key, kw_info in KEYWORDS.items():
    pool = [c for c in candidates[kw_key] if c["article_idx"] not in seen_idx]
    random.shuffle(pool)
    selected = pool[:EXAMPLES_PER_KEYWORD]

    for doc in selected:
        seen_idx.add(doc["article_idx"])
        sampled.append({
            "keyword": kw_key,
            "keyword_label": kw_info["label"],
            "keyword_zh": kw_info["zh"],
            "type": kw_info["type"],
            "culturax_text": doc["text"],
            "corpus": doc["corpus"],
        })

    print(f"  {kw_key} ({kw_info['label']}): {len(selected)} examples from pool of {len(candidates[kw_key])}")

print(f"\nTotal examples: {len(sampled)}")

# ── Save ──
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(sampled, f, ensure_ascii=False, indent=2)

print(f"Saved to {OUTPUT_FILE}")
