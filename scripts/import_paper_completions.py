"""Import paper-era memorization completions from RDS files into completions.json.

Reads the 20 completions_*.rds files (each a data frame with columns:
  id, type, start, end, start_chat, output_3_5, output_4, output_4_o,
  output_opus, output_sonnet)
and converts them into the same JSON format used by query_memorization.py.

Applies the paper's prefix-truncation matching and refusal detection.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from query_memorization import fuzzy_match, is_refusal
from translate import translate_zh_to_en, _load_cache, _save_cache

BASE_DIR = Path(__file__).resolve().parent.parent
COMPLETIONS_PATH = BASE_DIR / "data" / "memorization" / "completions.json"
RDS_DIR = Path("/tmp/memorization_completions")

# Paper model column → display name mapping
PAPER_MODELS = {
    "output_3_5": "gpt-3.5-instruct",
    "output_4": "gpt-4",
    "output_4_o": "gpt-4o",
    "output_opus": "claude-opus-3",
    "output_sonnet": "claude-sonnet-3",
}

# R script to convert all RDS files to a single JSON
R_SCRIPT = r"""
library(jsonlite)

files <- list.files("{rds_dir}", pattern = "completions_.*\\.rds$", full.names = TRUE)
frames <- lapply(files, readRDS)
df <- do.call(rbind, frames)

# Keep only the columns we need
cols <- c("id", "type", "start", "end", "start_chat",
          "output_3_5", "output_4", "output_4_o", "output_opus", "output_sonnet")
cols <- intersect(cols, names(df))
df <- df[, cols]

cat(toJSON(df, auto_unbox = TRUE, na = "null"))
"""


def main():
    # Step 1: Convert RDS to JSON via R
    print("Converting RDS files to JSON via R...")
    r_code = R_SCRIPT.replace("{rds_dir}", str(RDS_DIR))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(r_code)
        r_script_path = f.name

    result = subprocess.run(
        ["Rscript", r_script_path],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"R script failed:\n{result.stderr}")
        return

    paper_data = json.loads(result.stdout)
    print(f"Loaded {len(paper_data)} phrase rows from RDS files")

    # Step 2: Load existing completions (live models)
    if COMPLETIONS_PATH.exists():
        with open(COMPLETIONS_PATH, "r", encoding="utf-8") as f:
            completions = json.load(f)
    else:
        completions = []

    # Remove old paper entries (we'll replace them all)
    completions = [c for c in completions if c.get("timestamp") != "paper"]
    print(f"Kept {len(completions)} live completions")

    # Step 3: Convert paper data to our format
    # Use cache-only translations (no API calls) for speed.
    # Missing translations get empty strings; can fill in later.
    translation_cache = _load_cache()
    new_count = 0
    cache_hits = 0
    cache_misses = 0

    import hashlib
    def cache_only_translate(text):
        """Return cached translation or empty string (no API call)."""
        nonlocal cache_hits, cache_misses
        if not text or not text.strip():
            return ""
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in translation_cache:
            cache_hits += 1
            return translation_cache[key]
        cache_misses += 1
        return ""

    for i, row in enumerate(paper_data):
        phrase_id = row["id"]
        phrase_type = row["type"]
        start = row["start"]
        end = row["end"]
        prompt_text = row.get("start_chat") or f"续写句子：{start}"

        for col_name, model_name in PAPER_MODELS.items():
            completion_text = row.get(col_name)
            if completion_text is None or completion_text == "":
                continue

            refused = is_refusal(completion_text)
            matched, edit_distance = fuzzy_match(completion_text, end,
                                                 prompt_start=start)

            completions.append({
                "phrase_id": phrase_id,
                "type": phrase_type,
                "model": model_name,
                "prompt": prompt_text,
                "prompt_en": cache_only_translate(start),
                "expected": end,
                "expected_en": cache_only_translate(end),
                "completion": completion_text,
                "completion_en": cache_only_translate(completion_text),
                "matched": matched,
                "refused": refused,
                "edit_distance": round(edit_distance, 2),
                "timestamp": "paper",
            })
            new_count += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(paper_data)} phrases "
                  f"({new_count} completions, "
                  f"cache: {cache_hits} hits / {cache_misses} misses)")

    print(f"Added {new_count} paper completions "
          f"(cache: {cache_hits} hits / {cache_misses} misses)")

    # Step 4: Save
    _save_cache(translation_cache)
    tmp = str(COMPLETIONS_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(completions, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(COMPLETIONS_PATH)
    print(f"Saved {len(completions)} total completions to {COMPLETIONS_PATH}")

    # Step 5: Summary
    paper_completions = [c for c in completions if c.get("timestamp") == "paper"]
    models = sorted(set(c["model"] for c in paper_completions))
    for model in models:
        for ptype in ["propaganda", "culturax"]:
            subset = [c for c in paper_completions
                      if c["model"] == model and c["type"] == ptype]
            if not subset:
                continue
            refused = sum(1 for c in subset if c.get("refused"))
            non_refused = [c for c in subset if not c.get("refused")]
            matched = sum(1 for c in non_refused if c["matched"])
            total = len(non_refused)
            rate = matched / total if total > 0 else 0
            print(f"  {model} {ptype}: {matched}/{total} ({rate:.1%})"
                  f" [refused: {refused}]")


if __name__ == "__main__":
    main()
