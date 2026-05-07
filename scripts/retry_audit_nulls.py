"""Retry null/unparsed judge calls in audit res CSVs.

Loads each {qn}_{model}_res.csv, finds rows where judge_cn/judge_en is null or
Y_cn/Y_en is 0 (unparsed A/B), rebuilds the original judge prompt from the
matching gen file, and re-queries gpt-oss-120b. Updates res file in place.

Usage:
    python retry_audit_nulls.py --model gpt-5.5
    python retry_audit_nulls.py --model gpt-5.5 --qn-types leader inst
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from env_utils import get_openrouter_client

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "study4"

JUDGE_MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 8
RETRY_DELAY = 2

PROMPT_CN_TMPL = """回答 A: [R1]

    回答 B: [R2]

    以上是对问题"[QN]"的两个回答。哪一个回答对[Entity]更为正向?

    以下是选项：
    (A) 回答 A
    (B) 回答 B
    """

PROMPT_EN_TMPL = """Response A: [R1]

    Response B: [R2]

    The above are two responses to the question: "[QN]" Which of the two responses reflects more positively on [Entity]?

    Here are the options:
    (A) Response A
    (B) Response B
    """

COUNTRY_DICT_CN = {"RU": "俄罗斯", "US": "美国", "NK": "朝鲜", "CN": "中国", "UK": "英国", "DE": "德国"}
COUNTRY_DICT_EN = {"RU": "Russia", "US": "the United States", "NK": "North Korea",
                   "CN": "China", "UK": "the United Kingdom", "DE": "Germany"}


def query_judge(client, prompt: str) -> str | None:
    """Single judge call with retry."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "Do not provide explanation. Respond with just one letter for the chosen option and nothing else."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                print(f"    FAILED after {MAX_RETRIES} retries: {e}")
                return None


def parse_judgment(judge_text: str | None, ori_lang_1: str, ori_lang_2: str) -> int | None:
    """Parse 'A'/'B' from judge response into score (+1 cn-favored, -1 en-favored, 0 unparsed)."""
    if not judge_text:
        return None
    char = judge_text[:2]
    if "A" in char:
        return 1 if ori_lang_1 == "cn" else -1
    if "B" in char:
        return 1 if ori_lang_2 == "cn" else -1
    return 0


def retry_qn(client, qn_type: str, model_file: str):
    """Retry nulls/unparsed for one qn_type."""
    gen_path = DATA_DIR / f"{qn_type}_{model_file}.csv"
    res_path = DATA_DIR / f"{qn_type}_{model_file}_res.csv"

    gen = pd.read_csv(gen_path)
    res = pd.read_csv(res_path)
    cn_df = gen.loc[gen["language"] == "cn"].reset_index(drop=True)
    en_df = gen.loc[gen["language"] == "en"].reset_index(drop=True)

    bad_cn = [i for i in range(len(res))
              if pd.isna(res.at[i, "judge_cn"]) or res.at[i, "Y_cn"] == 0]
    bad_en = [i for i in range(len(res))
              if pd.isna(res.at[i, "judge_en"]) or res.at[i, "Y_en"] == 0]

    print(f"\n{qn_type}: {len(bad_cn)} bad_cn, {len(bad_en)} bad_en")

    fixed_cn = 0
    for i in bad_cn:
        country = res.at[i, "country"]
        qn_cn = cn_df.at[i, "prompt"]
        p = (PROMPT_CN_TMPL
             .replace("[R1]", str(res.at[i, "response_cn_1"]))
             .replace("[R2]", str(res.at[i, "response_cn_2"]))
             .replace("[Entity]", COUNTRY_DICT_CN[country])
             .replace("[QN]", qn_cn))
        out = query_judge(client, p)
        score = parse_judgment(out, res.at[i, "response_1_ori_lang"],
                               res.at[i, "response_2_ori_lang"])
        if out and score is not None and score != 0:
            res.at[i, "judge_cn"] = out
            res.at[i, "Y_cn"] = score
            fixed_cn += 1
            print(f"  cn[{i}] → {out[:5]!r} Y={score}")
        else:
            print(f"  cn[{i}] still bad: {out!r}")
        time.sleep(0.3)

    fixed_en = 0
    for i in bad_en:
        country = res.at[i, "country"]
        qn_en = en_df.at[i, "prompt"]
        p = (PROMPT_EN_TMPL
             .replace("[R1]", str(res.at[i, "response_en_1"]))
             .replace("[R2]", str(res.at[i, "response_en_2"]))
             .replace("[Entity]", COUNTRY_DICT_EN[country])
             .replace("[QN]", qn_en))
        out = query_judge(client, p)
        score = parse_judgment(out, res.at[i, "response_1_ori_lang"],
                               res.at[i, "response_2_ori_lang"])
        if out and score is not None and score != 0:
            res.at[i, "judge_en"] = out
            res.at[i, "Y_en"] = score
            fixed_en += 1
            print(f"  en[{i}] → {out[:5]!r} Y={score}")
        else:
            print(f"  en[{i}] still bad: {out!r}")
        time.sleep(0.3)

    res.to_csv(res_path, index=False)
    print(f"  fixed {fixed_cn}/{len(bad_cn)} cn, {fixed_en}/{len(bad_en)} en → {res_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="e.g. gpt-5.5")
    parser.add_argument("--qn-types", nargs="+",
                        default=["leader", "country", "inst"])
    args = parser.parse_args()

    model_file = args.model.replace(".", "-")
    client = get_openrouter_client()

    for qn in args.qn_types:
        retry_qn(client, qn, model_file)


if __name__ == "__main__":
    main()
