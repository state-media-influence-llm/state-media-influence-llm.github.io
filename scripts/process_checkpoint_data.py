"""
Process checkpoint CSV data into web-ready JSON.

Reads result_*.csv files from study3_pretraining/rank_32/ and produces:
- data/checkpoints/summary_en.json: China-focused, Chinese-language results (matches paper Figure 3)
- data/checkpoints/summary_detail.json: Full granular results for optional filters
- data/checkpoints/summary_multilingual.json: Aggregated multilingual results (CN country only)
- data/checkpoints/examples.json: Selected full-text response examples
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "checkpoints"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_EN = Path(
    "/Users/solomonmessing/workspace/influence_attribution/propaganda_llm/"
    "code_public/study3_pretraining/rank_32/result_gpt4o_en"
)
SOURCE_MULTI = Path(
    "/Users/solomonmessing/workspace/influence_attribution/propaganda_llm/"
    "code_public/study3_pretraining/rank_32/result_gpt4o_multilingual"
)

CORPORA = ["propaganda", "state_media", "culturax"]
BATCH_SIZE = 64  # training examples per step


def load_english_data() -> pd.DataFrame:
    """Load and concatenate the three English corpus CSVs."""
    frames = []
    for corpus in CORPORA:
        path = SOURCE_EN / f"result_{corpus}.csv"
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        df["corpus"] = corpus
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_multilingual_data() -> pd.DataFrame:
    """Load multilingual CSVs (es, jp, kr, ru, tc, vt)."""
    lang_map = {
        "es": "ES", "jp": "JP", "kr": "KR",
        "ru": "RU", "tc": "TC", "vt": "VT",
    }
    frames = []
    for suffix, lang_code in lang_map.items():
        path = SOURCE_MULTI / f"result_{suffix}.csv"
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        df["language"] = lang_code
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate_main(df: pd.DataFrame) -> list[dict]:
    """Main figure: filter to country==CN & language==CN, group by (corpus, step).
    This matches the paper's Figure 3 aggregation."""
    china_cn = df[(df["country"] == "CN") & (df["language"] == "CN")].dropna(subset=["Y"])
    agg = china_cn.groupby(["corpus", "step"]).agg(
        mean_Y=("Y", "mean"),
        n=("Y", "count"),
        se=("Y", lambda x: x.std() / (x.count() ** 0.5) if x.count() > 1 else 0)
    ).reset_index()
    agg["examples"] = agg["step"] * BATCH_SIZE
    agg["mean_Y"] = agg["mean_Y"].round(4)
    agg["se"] = agg["se"].round(4)
    return agg.to_dict(orient="records")


def aggregate_detail(df: pd.DataFrame) -> list[dict]:
    """Full granular aggregation for optional country/language filter view."""
    df_clean = df.dropna(subset=["Y"])
    agg = df_clean.groupby(["corpus", "step", "qn", "country", "language"]).agg(
        mean_Y=("Y", "mean"),
        n=("Y", "count"),
        se=("Y", lambda x: x.std() / (x.count() ** 0.5) if x.count() > 1 else 0)
    ).reset_index()
    agg["examples"] = agg["step"] * BATCH_SIZE
    agg["mean_Y"] = agg["mean_Y"].round(4)
    agg["se"] = agg["se"].round(4)
    return agg.to_dict(orient="records")


def aggregate_multilingual(df: pd.DataFrame) -> list[dict]:
    """Multilingual: filter to country==CN only, group by (corpus, step)
    where corpus is the language code."""
    if "country" in df.columns:
        df = df[df["country"] == "CN"]
    df_clean = df.dropna(subset=["Y"])
    # Group by language (as "corpus" for display) and step
    group_cols = ["language", "step"] if "language" in df.columns else ["corpus", "step"]
    agg = df_clean.groupby(group_cols).agg(
        mean_Y=("Y", "mean"),
        n=("Y", "count"),
        se=("Y", lambda x: x.std() / (x.count() ** 0.5) if x.count() > 1 else 0)
    ).reset_index()
    # Rename language to corpus for consistency with frontend
    if "language" in agg.columns and "corpus" not in agg.columns:
        agg = agg.rename(columns={"language": "corpus"})
    agg["examples"] = agg["step"] * BATCH_SIZE
    agg["mean_Y"] = agg["mean_Y"].round(4)
    agg["se"] = agg["se"].round(4)
    return agg.to_dict(orient="records")


def extract_examples(df: pd.DataFrame, steps: list[int] = [100, 500, 1000]) -> list[dict]:
    """Extract full-text examples at selected checkpoints."""
    examples = []
    for step in steps:
        step_df = df[df["step"] == step]
        if step_df.empty:
            continue
        for (corpus, country, qn), group in step_df.groupby(["corpus", "country", "qn"]):
            sample = group.head(2)
            for _, row in sample.iterrows():
                examples.append({
                    "step": int(step),
                    "examples": int(step * BATCH_SIZE),
                    "corpus": corpus,
                    "country": row.get("country", ""),
                    "qn": row.get("qn", ""),
                    "language": row.get("language", ""),
                    "prompt": str(row.get("prompt", ""))[:500],
                    "option1": str(row.get("option1", ""))[:500],
                    "option2": str(row.get("option2", ""))[:500],
                    "Y": float(row.get("Y", 0)) if pd.notna(row.get("Y")) else None,
                })
    return examples


def write_json(data, path: Path):
    """Write data as compact JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    size_kb = path.stat().st_size / 1024
    print(f"Wrote {path.name}: {len(data)} records, {size_kb:.1f} KB")


def main():
    print("Processing English checkpoint data...")
    en_df = load_english_data()
    print(f"  Loaded {len(en_df)} rows")

    summary_main = aggregate_main(en_df)
    write_json(summary_main, DATA_DIR / "summary_en.json")

    summary_detail = aggregate_detail(en_df)
    write_json(summary_detail, DATA_DIR / "summary_detail.json")

    examples = extract_examples(en_df)
    write_json(examples, DATA_DIR / "examples.json")

    print("\nProcessing multilingual checkpoint data...")
    multi_df = load_multilingual_data()
    if len(multi_df) > 0:
        print(f"  Loaded {len(multi_df)} rows")
        summary_multi = aggregate_multilingual(multi_df)
        write_json(summary_multi, DATA_DIR / "summary_multilingual.json")
    else:
        print("  No multilingual data found, writing empty file")
        write_json([], DATA_DIR / "summary_multilingual.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
