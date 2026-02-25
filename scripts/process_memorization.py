"""
Process memorization phrases into web-ready JSON.

Reads memorization_phrases.json (2,000 phrases across 20 folds)
and produces data/memorization/phrases.json (top 100 propaganda + top 100 culturax).
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "memorization"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SOURCE = BASE_DIR / "memorization_phrases.json"


def main():
    with open(SOURCE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # raw is dict of fold_id -> list of phrase dicts
    all_phrases = []
    seen_ids = set()
    for fold_id, phrases in raw.items():
        for p in phrases:
            pid = p.get("id", f"{fold_id}_{p.get('phrase', '')[:20]}")
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            all_phrases.append({
                "id": pid,
                "coef": p["coef"],
                "phrase": p.get("phrase", ""),
                "type": p.get("type", ""),
                "start": p.get("start", ""),
                "end": p.get("end", ""),
                "start_chat": p.get("start_chat", ""),
            })

    # Sort by |coef| descending
    all_phrases.sort(key=lambda x: abs(x["coef"]), reverse=True)

    # Take top 100 propaganda + top 100 culturax
    propaganda = [p for p in all_phrases if p["type"] == "propaganda"][:100]
    culturax = [p for p in all_phrases if p["type"] == "culturax"][:100]
    output = propaganda + culturax

    out_path = DATA_DIR / "phrases.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path.name}: {len(output)} phrases ({len(propaganda)} propaganda, {len(culturax)} culturax), {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
