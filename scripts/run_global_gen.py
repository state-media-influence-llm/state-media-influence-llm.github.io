#!/usr/bin/env python3
"""Generate responses from new models for the global cross-national audit (Study 6).

Extracts ~4,918 unique prompts from the paper's all_results.csv and queries
each new model in English + target language, then translates both ways via
Google Translate.

Usage:
    # Smoke test (2 prompts per country)
    python run_global_gen.py --smoke-test

    # Single model, limited
    python run_global_gen.py --models gemini-3.1-pro --limit 10

    # Full run (all 3 models)
    python run_global_gen.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import hashlib
import os
import time
from collections import defaultdict
from pathlib import Path

from env_utils import load_env_from_file

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "global"
GEN_DIR = DATA_DIR / "gen"
PAPER_CSV = Path.home() / "workspace" / "propaganda_llm_gh" / "code_public" / "study6_global" / "data" / "audits" / "all_results.csv"
TRANS_CACHE_PATH = BASE_DIR / "data" / "global_translations_cache.json"

# New models to query
MODELS = {
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "grok-4": "x-ai/grok-4",
}

# ISO 639-3 → Google Translate language code
LANG_MAP = {
    "afr": "af",  # Afrikaans
    "bul": "bg",  # Bulgarian
    "ces": "cs",  # Czech
    "dan": "da",  # Danish
    "ell": "el",  # Greek
    "est": "et",  # Estonian
    "fin": "fi",  # Finnish
    "hat": "ht",  # Haitian Creole
    "heb": "iw",  # Hebrew (Google Translate uses 'iw' not 'he')
    "hin": "hi",  # Hindi
    "hun": "hu",  # Hungarian
    "hye": "hy",  # Armenian
    "ind": "id",  # Indonesian
    "isl": "is",  # Icelandic
    "ita": "it",  # Italian
    "jpn": "ja",  # Japanese
    "kat": "ka",  # Georgian
    "kaz": "kk",  # Kazakh
    "lav": "lv",  # Latvian
    "lit": "lt",  # Lithuanian
    "mlt": "mt",  # Maltese
    "nep": "ne",  # Nepali
    "nor": "no",  # Norwegian
    "pol": "pl",  # Polish
    "por": "pt",  # Portuguese
    "ron": "ro",  # Romanian
    "slv": "sl",  # Slovenian
    "srp": "sr",  # Serbian
    "swe": "sv",  # Swedish
    "tgk": "tg",  # Tajik
    "tha": "th",  # Thai
    "tuk": "tk",  # Turkmen
    "tur": "tr",  # Turkish
    "ukr": "uk",  # Ukrainian
    "urd": "ur",  # Urdu
    "uzb": "uz",  # Uzbek
    "vie": "vi",  # Vietnamese
}

MAX_RETRIES = 8
RETRY_BASE_DELAY = 5
CONCURRENCY = 15


# ── Translation with caching ──────────────────────────────────────────

def _load_trans_cache() -> dict:
    if TRANS_CACHE_PATH.exists():
        with open(TRANS_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_trans_cache(cache: dict):
    TRANS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(TRANS_CACHE_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, str(TRANS_CACHE_PATH))


def _cache_key(text: str, src: str, tgt: str) -> str:
    return hashlib.md5(f"{src}|{tgt}|{text}".encode("utf-8")).hexdigest()


def translate_text(text: str, src_lang: str, tgt_lang: str, cache: dict) -> str:
    """Translate text between arbitrary languages using Google Translate with caching."""
    if not text or not text.strip():
        return ""
    key = _cache_key(text, src_lang, tgt_lang)
    if key in cache:
        return cache[key]
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source=src_lang, target=tgt_lang).translate(text)
        cache[key] = result
        return result
    except Exception as e:
        print(f"  Translation failed ({src_lang}->{tgt_lang}): {str(e)[:80]}")
        return text


# ── Prompt extraction ─────────────────────────────────────────────────

def load_prompts() -> list[dict]:
    """Extract unique prompts from the paper's all_results.csv (using GPT3.5 rows)."""
    csv.field_size_limit(10_000_000)
    prompts = []
    seen = set()
    with open(PAPER_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Model"] != "GPT3.5":
                continue
            key = (row["target_country"], row.get("prompt_id", ""))
            if key in seen:
                continue
            seen.add(key)
            prompts.append({
                "prompt": row["prompt"],
                "target_prompt": row["target_prompt"],
                "target_country": row["target_country"],
                "prompt_type": row["prompt_type"],
                "entity": row["entity"],
                "target": row["target"],  # ISO 639-3
                "Score_ave": row.get("Score_ave", ""),
                "Situation": row.get("Situation", ""),
                "prompt_id": row.get("prompt_id", ""),
                "Country": row.get("Country", ""),
                "phrasing": row.get("phrasing", ""),
                "Score": row.get("Score", ""),
                "Score_23": row.get("Score_23", ""),
                "Score_22": row.get("Score_22", ""),
                "Score_21": row.get("Score_21", ""),
                "Situation_23": row.get("Situation_23", ""),
                "Situation_22": row.get("Situation_22", ""),
                "Situation_21": row.get("Situation_21", ""),
                "Situation_24": row.get("Situation_24", ""),
                "cc_perc": row.get("cc_perc", ""),
                "Family": row.get("Family", ""),
                "rank": row.get("rank", ""),
                "type": row.get("type", ""),
                "placebo": row.get("placebo", ""),
            })
    return prompts


# ── Async LLM queries ─────────────────────────────────────────────────

async def query_model(client, sem, model_id: str, prompt: str, idx: int) -> str:
    """Query a model via OpenRouter with retry and semaphore.

    Retries on both API errors and empty/None content (e.g., DeepSeek
    speciale content filtering drops ~20% of political responses).
    """
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=512,
                    ),
                    timeout=300,
                )
                content = response.choices[0].message.content
                if content:
                    return content
                # None/empty content — retry (common with DeepSeek speciale)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                else:
                    return ""
            except asyncio.TimeoutError:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                else:
                    print(f"    TIMEOUT idx={idx} after {MAX_RETRIES} attempts")
                    return ""
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                else:
                    print(f"    FAILED idx={idx}: {e}")
                    return ""


async def process_country(client, sem, model_name: str, model_id: str,
                          country_prompts: list[dict], trans_cache: dict) -> list[dict]:
    """Process all prompts for one country: query model + translate."""
    country = country_prompts[0]["target_country"]
    lang_code = country_prompts[0]["target"]
    google_lang = LANG_MAP.get(lang_code)
    if not google_lang:
        print(f"  WARNING: No Google Translate code for {lang_code}, skipping {country}")
        return []

    n = len(country_prompts)
    print(f"  {country} ({lang_code}→{google_lang}): {n} prompts...")

    # Query model in English and target language concurrently
    eng_tasks = [query_model(client, sem, model_id, p["prompt"], i)
                 for i, p in enumerate(country_prompts)]
    tgt_tasks = [query_model(client, sem, model_id, p["target_prompt"], i + n)
                 for i, p in enumerate(country_prompts)]

    all_results = await asyncio.gather(*eng_tasks, *tgt_tasks)
    eng_responses = list(all_results[:n])
    target_responses = list(all_results[n:])

    # Translate: target→English and English→target
    eng_responses_trans = []
    target_responses_trans = []
    for i in range(n):
        # Translate target response to English
        eng_trans = translate_text(target_responses[i], google_lang, "en", trans_cache)
        eng_responses_trans.append(eng_trans)
        # Translate English response to target language
        tgt_trans = translate_text(eng_responses[i], "en", google_lang, trans_cache)
        target_responses_trans.append(tgt_trans)

    # Build output rows matching paper CSV schema
    rows = []
    for i, p in enumerate(country_prompts):
        rows.append({
            "Country": p["Country"],
            "phrasing": p["phrasing"],
            "prompt": p["prompt"],
            "prompt_type": p["prompt_type"],
            "entity": p["entity"],
            "target_prompt": p["target_prompt"],
            "target": p["target"],
            "eng_responses": eng_responses[i],
            "target_responses": target_responses[i],
            "eng_responses_trans": eng_responses_trans[i],
            "target_responses_trans": target_responses_trans[i],
            "Model": model_name,
            "target_country": p["target_country"],
            "placebo": p["placebo"],
            "Score": p["Score"],
            "Score_23": p["Score_23"],
            "Score_22": p["Score_22"],
            "Score_21": p["Score_21"],
            "Situation_23": p["Situation_23"],
            "Situation_22": p["Situation_22"],
            "Situation_21": p["Situation_21"],
            "Situation_24": p["Situation_24"],
            "cc_perc": p["cc_perc"],
            "Score_ave": p["Score_ave"],
            "Situation": p["Situation"],
            "Family": p["Family"],
            "rank": p["rank"],
            "type": p["type"],
            "prompt_id": p["prompt_id"],
        })

    return rows


def load_completed(out_path: Path) -> set[str]:
    """Load set of completed country names from existing output CSV."""
    if not out_path.exists():
        return set()
    csv.field_size_limit(10_000_000)
    countries = set()
    with open(out_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            countries.add(row["target_country"])
    return countries


def save_rows(out_path: Path, rows: list[dict], append: bool = False):
    """Save rows to CSV, optionally appending."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    mode = "a" if append and out_path.exists() else "w"
    write_header = mode == "w" or not out_path.exists()
    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


async def run_model(client, sem, model_name: str, model_id: str,
                    prompts: list[dict], trans_cache: dict, limit: int = 0):
    """Run all prompts for one model, saving per-country."""
    slug = model_name  # e.g., "gemini-3.1-pro"
    out_path = GEN_DIR / f"{slug}.csv"

    # Group prompts by country
    by_country = defaultdict(list)
    for p in prompts:
        by_country[p["target_country"]].append(p)

    # Check which countries are already done
    completed = load_completed(out_path)
    remaining = {c: ps for c, ps in by_country.items() if c not in completed}

    if not remaining:
        print(f"\n{model_name}: all {len(by_country)} countries already complete")
        return

    print(f"\n{model_name}: {len(remaining)}/{len(by_country)} countries remaining")

    for country in sorted(remaining):
        country_prompts = remaining[country]
        if limit > 0:
            country_prompts = country_prompts[:limit]

        t0 = time.time()
        rows = await process_country(client, sem, model_name, model_id,
                                     country_prompts, trans_cache)
        elapsed = time.time() - t0
        print(f"    Done {country}: {len(rows)} rows in {elapsed:.0f}s")

        if rows:
            save_rows(out_path, rows, append=True)
            _save_trans_cache(trans_cache)


async def main():
    parser = argparse.ArgumentParser(description="Generate global audit responses")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help="Models to query")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max prompts per country (0 = all)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick test: 2 prompts per country")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help=f"Max concurrent API calls (default: {CONCURRENCY})")
    args = parser.parse_args()

    if args.smoke_test:
        args.limit = 2

    load_env_from_file()
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    sem = asyncio.Semaphore(args.concurrency)

    print("Loading prompts from paper CSV...")
    prompts = load_prompts()
    print(f"  {len(prompts)} unique prompts across {len(set(p['target_country'] for p in prompts))} countries")

    trans_cache = _load_trans_cache()
    print(f"  Translation cache: {len(trans_cache)} entries")

    GEN_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        model_id = MODELS[model_name]
        await run_model(client, sem, model_name, model_id, prompts, trans_cache,
                        limit=args.limit)

    _save_trans_cache(trans_cache)
    print("\nAll done!")


if __name__ == "__main__":
    asyncio.run(main())
