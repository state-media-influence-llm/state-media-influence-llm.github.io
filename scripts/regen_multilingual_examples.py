#!/usr/bin/env python3
"""Regenerate ES/RU/VT rows in data/checkpoints/examples_multilingual.json.

The original file includes rows where the Llama 2 model responded in English
even though the prompt was in Spanish/Russian/Vietnamese. We filter those out
and pick 2 country + 2 inst + 2 leader per language where both options contain
target-language diacritics, then translate to English for the viewer's
option1_en / option2_en columns.

JP/KR/TC are kept unchanged (those languages don't have the English-response issue).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

csv.field_size_limit(10_000_000)

REPO = Path(__file__).resolve().parent.parent
JSON_PATH = REPO / "data" / "checkpoints" / "examples_multilingual.json"
PAPER_BASE = Path(os.environ.get(
    "PAPER_DATA_DIR",
    os.path.expanduser("~/workspace/propaganda_llm_gh/code_public"),
))
SOURCE_DIR = PAPER_BASE / "study3_pretraining" / "rank_32" / "result_gpt4o_multilingual"
POL_Q_DIR = PAPER_BASE / "study3_pretraining" / "data"
CACHE_PATH = REPO / "data" / "translations_cache.json"

PATTERNS = {
    "ES": re.compile(r"[ñÑáíóúéÁÍÓÚÉ¿¡]"),
    "RU": re.compile(r"[\u0400-\u04FF]"),
    # Vietnamese-specific chars only (đ/ă/ơ/ư + chars with stacked-tone marks
    # that never appear in Spanish/French/etc.). Excludes á, é, ó, ú which
    # overlap with Spanish.
    "VT": re.compile(
        r"[đĐăĂơƠưƯ"
        r"ảãạắằẳẵặấầẩẫậẻẽẹếềểễệỉĩịỏõọốồổỗộớờởỡợủũụứừửữựỳỷỹỵ]"
    ),
    "JP": re.compile(r"[\u3040-\u309F\u30A0-\u30FF]"),  # hiragana + katakana
    "KR": re.compile(r"[\uAC00-\uD7AF]"),               # hangul
    "TC": re.compile(r"[\u4E00-\u9FFF]"),               # any CJK ideograph
}

LANG_CODE = {"ES": "es", "RU": "ru", "VT": "vi",
             "JP": "ja", "KR": "ko", "TC": "zh-TW"}
QN_ORDER = ["country", "inst", "leader"]
PER_QN = 2  # 2 examples per qn → 6 total per language

SEED = 92092


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    tmp = CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(CACHE_PATH)


def translate(text: str, src: str, cache: dict) -> str:
    if not text or not text.strip():
        return ""
    key = hashlib.md5(f"{src}|{text}".encode()).hexdigest()
    if key in cache:
        return cache[key]
    from deep_translator import GoogleTranslator
    try:
        result = GoogleTranslator(source=src, target="en").translate(text)
    except Exception as e:
        print(f"  translate failed ({src}): {e}", file=sys.stderr)
        return text
    cache[key] = result
    return result


def load_political_questions(lang: str) -> dict[str, str]:
    """Map English question → target-language instruction for this language.

    ES/RU/VT files store the English question in the `question` field directly.
    EN/JP/KR/TC files store Simplified Chinese there, so we pair by row index
    with es.json (English) to recover the English question.

    If two English questions map to the same target instruction (rare but
    possible across paraphrases), warn so the caller knows the inverted lookup
    in regen_multilingual_examples will pick whichever row was loaded last.
    """
    with open(POL_Q_DIR / f"political_question_{lang.lower()}.json", encoding="utf-8") as f:
        lang_data = json.load(f)
    if lang in ("ES", "RU", "VT"):
        pairs = [(d["question"], d["instruction"]) for d in lang_data]
    else:
        with open(POL_Q_DIR / "political_question_es.json", encoding="utf-8") as f:
            es_data = json.load(f)
        if len(es_data) != len(lang_data):
            raise RuntimeError(
                f"Row count mismatch: es.json={len(es_data)} {lang.lower()}.json={len(lang_data)}"
            )
        pairs = [(es_data[i]["question"], lang_data[i]["instruction"])
                 for i in range(len(lang_data))]

    seen_instr = {}
    collisions = []
    mapping = {}
    for q_en, instr in pairs:
        if instr in seen_instr and seen_instr[instr] != q_en:
            collisions.append((instr, seen_instr[instr], q_en))
        seen_instr[instr] = q_en
        mapping[q_en] = instr
    if collisions:
        print(f"  WARN: {len(collisions)} duplicate target instructions in {lang}:",
              file=sys.stderr)
        for instr, q1, q2 in collisions[:3]:
            print(f"    {instr!r} maps to both {q1!r} and {q2!r}", file=sys.stderr)
    return mapping


def pick_rows(lang: str, rng: random.Random) -> list[dict]:
    """Pick 2 country + 2 inst + 2 leader rows for a language."""
    path = SOURCE_DIR / f"result_{lang.lower()}.csv"
    with open(path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    pat = PATTERNS[lang]
    def baseline_has_tgt(r):
        # option1_m / option2_m encode model: "0" = baseline, "1" = propaganda-trained.
        # We require the baseline response to be in target language; the trained
        # response may be in target language OR drift to English (the VT case).
        if r["option1_m"] == "0":
            return bool(pat.search(r["option1"]))
        return bool(pat.search(r["option2"]))

    filt = [
        r for r in all_rows
        if int(r["step"]) == 1000 and r["country"] == "CN" and baseline_has_tgt(r)
    ]
    print(f"  {lang}: {len(filt)} candidates (baseline response in target language)")

    # Build target→English question map so we can dedupe by English question.
    pol_q = load_political_questions(lang)          # EN question → target instruction
    target_to_en = {v: k for k, v in pol_q.items()}

    quote_chars = "\u0022\u201C\u201D\u00AB\u00BB\u300C\u300D\u300E\u300F\u201E\u201A"
    quote_re = re.compile(f"[{quote_chars}]([^{quote_chars}]+?)[{quote_chars}]")

    def row_question_en(r):
        matches = quote_re.findall(r["prompt"])
        target_q = matches[-1].strip() if matches else ""
        return target_to_en.get(target_q, "")

    chosen = []
    used_questions = set()
    for qn in QN_ORDER:
        subset = [r for r in filt if r["qn"] == qn]
        rng.shuffle(subset)
        picks = []
        for r in subset:
            qen = row_question_en(r)
            if not qen or qen in used_questions:
                continue
            picks.append(r)
            used_questions.add(qen)
            if len(picks) == PER_QN:
                break
        if len(picks) < PER_QN:
            print(f"    WARNING: only {len(picks)} unique {qn} rows")
        chosen.extend(picks)
        for r in picks:
            preview = r["option1"][:60].replace("\n", " ")
            print(f"    [{qn}] {preview}...")
    return chosen


def extract_question_en(source_row: dict, prompt_to_qen: dict[str, str]) -> str:
    """Extract English question from the judge-prompt wrapper.

    The prompt field contains the judge wrapper ending with the original question
    in target language. We map it back to English via the political_question json.
    """
    # Source CSV doesn't carry the English question directly. Best approach:
    # use custom_id → index into the prompts, which maps to political_question_en.json
    # But custom_id format is "request-NNNN" which we can't easily decode without
    # the reverse mapping. So instead, extract the TARGET-language question from
    # the judge prompt and invert the prompt_to_qen map.
    m = re.search(r'(?:siguientes? son dos respuestas? a la pregunta|Вышеуказанные)\s*[^\n]*?[""«]([^""»]+)[""»]',
                  source_row["prompt"])
    if m:
        target_q = m.group(1).strip()
        return prompt_to_qen.get(target_q, target_q)
    return ""


def main():
    # Load existing JSON and split by language
    with open(JSON_PATH, encoding="utf-8") as f:
        existing = json.load(f)

    # Rebuild all six languages from source
    keep = []
    cache = load_cache()
    rng = random.Random(SEED)
    new_rows = []

    quote_chars = "\u0022\u201C\u201D\u00AB\u00BB\u300C\u300D\u300E\u300F\u201E\u201A"
    quote_re = re.compile(f"[{quote_chars}]([^{quote_chars}]+?)[{quote_chars}]")

    for lang in ["ES", "JP", "KR", "RU", "TC", "VT"]:
        print(f"\n=== {lang} ===")
        src_code = LANG_CODE[lang]
        pol_q = load_political_questions(lang)          # EN question → target instruction
        target_to_en = {v: k for k, v in pol_q.items()}

        picks = pick_rows(lang, rng)
        for r in picks:
            prompt_text = r["prompt"]
            qmatches = quote_re.findall(prompt_text)
            target_q = qmatches[-1].strip() if qmatches else ""
            question_en = target_to_en.get(target_q, "")
            if not question_en:
                for tgt, en in target_to_en.items():
                    if tgt in prompt_text:
                        target_q = tgt
                        question_en = en
                        break

            # Translate options
            opt1_en = translate(r["option1"], src_code, cache)
            opt2_en = translate(r["option2"], src_code, cache)

            new_rows.append({
                "language": lang,
                "step": int(r["step"]),
                "examples": 64000,  # step=1000 → 64k examples (matches existing convention)
                "corpus": "propaganda",
                "qn": r["qn"],
                "country": r["country"],
                "option1": r["option1"],
                "option2": r["option2"],
                "Y": float(r["Y"]),
                "option1_en": opt1_en,
                "option2_en": opt2_en,
                "question": target_q,
                "question_en": question_en,
                "option1_m": int(r["option1_m"]),
                "option2_m": int(r["option2_m"]),
            })
        save_cache(cache)

    # Combine and write
    # Preserve order: ES rows, JP rows, KR rows, RU rows, TC rows, VT rows
    final = []
    for lang in ["ES", "JP", "KR", "RU", "TC", "VT"]:
        final.extend([d for d in new_rows if d["language"] == lang])
        final.extend([d for d in keep if d["language"] == lang])

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=None)

    print(f"\nWrote {len(final)} rows to {JSON_PATH}")
    from collections import Counter
    print("by language:", dict(Counter(d["language"] for d in final)))


if __name__ == "__main__":
    main()
