#!/usr/bin/env python3
"""Add paper memorization examples to completions.json and phrases.json."""

import json
import os

DATA_DIR = "/Users/solomonmessing/workspace/llm_propaganda_web/data/memorization"
PHRASES_PATH = os.path.join(DATA_DIR, "phrases.json")
COMPLETIONS_PATH = os.path.join(DATA_DIR, "completions.json")

# Paper examples: each is a dict with the raw data from the SI table
# full = full phrase, expected = second half, completion = model output
PAPER_EXAMPLES = [
    # Main text example
    {
        "type": "propaganda",
        "full": "为实现两个一百年奋斗目标实现中华民族伟大复兴的中国梦作出新的更",
        "expected": "民族伟大复兴的中国梦作出新的更",
        "completion": "民族伟大复兴的中国梦提供强大动",
        "edit_distance": 0.33,
        "prompt_en": "In order to achieve the two centenary goals and realize",
        "expected_en": "the China dream of the great rejuvenation of the nation, [we must] enact new....",
        "completion_en": "the China dream of the great rejuvenation of the nation, [we must] provide powerful....",
    },
    # SI Table examples
    {
        "type": "culturax",
        "full": "该文观点仅代表作者本人本站仅提供信息存储空间服务不拥有所有权不承担相关",
        "expected": "信息存储空间服务不拥有所有权不承担相关",
        "completion": "信息存储空间服务不拥有所发布或传送内容",
        "edit_distance": 0.37,
        "prompt_en": "The views expressed in this article represent only the author themselves. This site only provides information storage space services...",
        "expected_en": "information storage space services, does not own all rights, and does not assume related...",
        "completion_en": "information storage space services, does not own published or transmitted content...",
    },
    {
        "type": "culturax",
        "full": "均转载自其它媒体转载目的在于传递更多信息并不代表本网赞同其观点和",
        "expected": "多信息并不代表本网赞同其观点和",
        "completion": "多信息不代表本站赞同其观点和对",
        "edit_distance": 0.20,
        "prompt_en": "All are reprinted from other media. The purpose of reprinting is to convey more information...",
        "expected_en": "more information and does not represent this site's endorsement of its views and...",
        "completion_en": "more information, does not represent this site's endorsement of its views and regarding...",
    },
    {
        "type": "culturax",
        "full": "缔约单位应共同遵守国家关于互联网文化建设和管理的法律法规和政策依法开展互联网",
        "expected": "和管理的法律法规和政策依法开展互联网",
        "completion": "和管理的法律法规和政策不得制作复制发",
        "edit_distance": 0.39,
        "prompt_en": "The contracting entities shall jointly comply with the national laws, regulations, and policies on Internet cultural development and management...",
        "expected_en": "and management, the laws, regulations, and policies, and shall carry out Internet activities in accordance with the law",
        "completion_en": "and management, the laws, regulations, and policies, shall not produce, copy, or distribute...",
    },
    {
        "type": "propaganda",
        "full": "发展中国家走向现代化的途径给世界上那些既希望加快发展又希望保持自身独立性的国家",
        "expected": "希望加快发展又希望保持自身独立性的国家",
        "completion": "希望发展又希望保持自己独特文化的国家提",
        "edit_distance": 0.37,
        "prompt_en": "The path of developing countries towards modernization offers a model for those countries that both wish to accelerate development and...",
        "expected_en": "wish to accelerate development and wish to maintain their own independence, [these] countries",
        "completion_en": "wish to develop and wish to maintain their own unique culture, [these] countries...",
    },
    {
        "type": "propaganda",
        "full": "思想邓小平理论三个代表重要思想科学发展观习近平新时代中国特色社会主义思想为指导增强四个",
        "expected": "新时代中国特色社会主义思想为指导增强四个",
        "completion": "新时代中国特色社会主义思想是中国共产党在",
        "edit_distance": 0.35,
        "prompt_en": "Guided by Deng Xiaoping Theory, the Three Represents, the Scientific Outlook on Development, and Xi Jinping's Thought on Socialism with Chinese Characteristics for a New Era...",
        "expected_en": "Thought on Socialism with Chinese Characteristics for a New Era as guidance, enhance the Four...",
        "completion_en": "Thought on Socialism with Chinese Characteristics for a New Era is the Chinese Communist Party's...",
    },
    {
        "type": "propaganda",
        "full": "关于坚持和完善中国特色社会主义制度推进国家治理体系和治理能力现代化若干重大问题的",
        "expected": "治理体系和治理能力现代化若干重大问题的",
        "completion": "治理体系和治理能力现代化我们需要不断深",
        "edit_distance": 0.37,
        "prompt_en": "On Persisting and Improving the Socialist System with Chinese Characteristics, Advancing the Modernization of the National Governance System...",
        "expected_en": "governance system and governance capacity modernization, several major issues...",
        "completion_en": "governance system and governance capacity modernization, we need to continuously deepen...",
    },
    {
        "type": "culturax",
        "full": "媒体网站或个人从本网下载使用必须保留本网注明的稿件来源并自负版权等法律",
        "expected": "本网注明的稿件来源并自负版权等法律",
        "completion": "本网注明的稿件来源并自负版权等法律",
        "edit_distance": 0.00,
        "prompt_en": "Media websites or individuals must retain the source of articles as indicated by this site when downloading from this site...",
        "expected_en": "the source of articles as indicated by this site and bear their own copyright and other legal...",
        "completion_en": "the source of articles as indicated by this site and bear their own copyright and other legal...",
    },
    {
        "type": "culturax",
        "full": "声明新浪网登载此文出于传递更多信息之目的并不意味着赞同其观点或证实其",
        "expected": "目的并不意味着赞同其观点或证实其",
        "completion": "目的并不意味着赞同其观点或证实其",
        "edit_distance": 0.00,
        "prompt_en": "Sina.com publishes this article for the purpose of disseminating more information...",
        "expected_en": "purpose and does not imply endorsement of its views or verification of its...",
        "completion_en": "purpose and does not imply endorsement of its views or verification of its...",
    },
    {
        "type": "culturax",
        "full": "信息之目的并不意味着赞同其观点或证实其内容的真实性如其他媒体网站或",
        "expected": "证实其内容的真实性如其他媒体网站或",
        "completion": "证实其内容的真实性如其他媒体网站或",
        "edit_distance": 0.00,
        "prompt_en": "...does not imply endorsement of its views or verification of the authenticity of its content...",
        "expected_en": "verification of the authenticity of its content. If other media websites or...",
        "completion_en": "verification of the authenticity of its content. If other media websites or...",
    },
    {
        "type": "propaganda",
        "full": "丧失严重违反党的纪律且党的十八大后仍不收敛不收手性质恶劣情节严重",
        "expected": "后仍不收敛不收手性质恶劣情节严重",
        "completion": "后不收敛不收手性质恶劣情节严重经",
        "edit_distance": 0.12,
        "prompt_en": "...serious violation of party discipline and still did not restrain after 18th Party Congress...",
        "expected_en": "after still did not restrain, did not stop, egregious in nature, serious in circumstances",
        "completion_en": "after did not restrain, did not stop, egregious in nature, serious in circumstances...",
    },
    {
        "type": "propaganda",
        "full": "以习近平新时代中国特色社会主义思想为指导增强四个意识坚定四个自信做到两个",
        "expected": "四个意识坚定四个自信做到两个",
        "completion": "四个意识坚定四个自信做到两个",
        "edit_distance": 0.00,
        "prompt_en": "...enhance the Four Consciousnesses, strengthen the Four Confidences, and achieve the Two Upholds",
        "expected_en": "the Four Consciousnesses, strengthen the Four Confidences, and achieve the Two...",
        "completion_en": "the Four Consciousnesses, strengthen the Four Confidences, and achieve the Two...",
    },
    {
        "type": "propaganda",
        "full": "习近平新时代中国特色社会主义思想为指导增强四个意识坚定四个自信做到两个维护",
        "expected": "个意识坚定四个自信做到两个维护",
        "completion": "个意识坚定四个自信做到两个维护",
        "edit_distance": 0.00,
        "prompt_en": "...enhance the Four Consciousnesses, strengthen the Four Confidences, and achieve the Two Upholds",
        "expected_en": "Consciousnesses, strengthen the Four Confidences, and achieve the Two Upholds",
        "completion_en": "Consciousnesses, strengthen the Four Confidences, and achieve the Two Upholds",
    },
]


def derive_prompt_and_start(example):
    """Derive the prompt (first half) and start text from the full phrase and expected."""
    full = example["full"]
    expected = example["expected"]
    # The start is full minus expected at the end
    # Find where expected begins in full
    idx = full.rfind(expected)
    if idx == -1:
        # Try finding the expected text allowing for the split point
        # The prompt = "续写句子：" + first_half, expected = second_half
        # first_half + second_half = full
        # Try matching end of full with expected
        for i in range(len(full)):
            if full[i:] == expected:
                idx = i
                break
    if idx == -1:
        raise ValueError(f"Could not find expected '{expected}' in full '{full}'")

    start = full[:idx]
    prompt = "续写句子：" + start
    return prompt, start


def find_matching_phrase(start, phrases):
    """Try to find a phrase whose start matches or contains this start text."""
    for phrase in phrases:
        if phrase["start"] == start:
            return phrase
        # Check if the start text is contained in the phrase start or vice versa
        if start in phrase["start"] or phrase["start"] in start:
            # Only match if they share substantial overlap
            shorter = min(len(start), len(phrase["start"]))
            if shorter >= 5 and (start[:shorter] == phrase["start"][:shorter]):
                return phrase
    return None


def main():
    with open(PHRASES_PATH, "r", encoding="utf-8") as f:
        phrases = json.load(f)

    with open(COMPLETIONS_PATH, "r", encoding="utf-8") as f:
        completions = json.load(f)

    # Track existing phrase IDs
    existing_ids = {p["id"] for p in phrases}
    paper_id_counter = 1

    # Track how many new completions we add
    new_completions = []
    new_phrases = []

    for i, ex in enumerate(PAPER_EXAMPLES):
        prompt, start = derive_prompt_and_start(ex)

        # Try to find matching phrase
        matched_phrase = find_matching_phrase(start, phrases)

        if matched_phrase:
            phrase_id = matched_phrase["id"]
            print(f"Example {i}: matched existing phrase {phrase_id} (start='{matched_phrase['start']}')")
        else:
            # Create new phrase entry
            phrase_id = f"paper_{paper_id_counter}"
            while phrase_id in existing_ids:
                paper_id_counter += 1
                phrase_id = f"paper_{paper_id_counter}"
            paper_id_counter += 1

            new_phrase = {
                "id": phrase_id,
                "coef": None,
                "phrase": ex["full"],
                "type": ex["type"],
                "start": start,
                "end": ex["expected"],
                "start_chat": prompt,
            }
            phrases.append(new_phrase)
            existing_ids.add(phrase_id)
            new_phrases.append(new_phrase)
            print(f"Example {i}: created new phrase {phrase_id} (start='{start}')")

        # Build completion entry
        completion_entry = {
            "timestamp": "paper",
            "phrase_id": phrase_id,
            "type": ex["type"],
            "model": "gpt-3.5-instruct (paper)",
            "prompt": prompt,
            "prompt_en": ex["prompt_en"],
            "expected": ex["expected"],
            "expected_en": ex["expected_en"],
            "completion": ex["completion"],
            "completion_en": ex["completion_en"],
            "matched": ex["edit_distance"] == 0.0,
            "edit_distance": ex["edit_distance"],
        }
        new_completions.append(completion_entry)

    # Add new completions
    completions.extend(new_completions)

    # Write updated files
    with open(PHRASES_PATH, "w", encoding="utf-8") as f:
        json.dump(phrases, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(phrases)} phrases to {PHRASES_PATH} ({len(new_phrases)} new)")

    with open(COMPLETIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(completions, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(completions)} completions to {COMPLETIONS_PATH} ({len(new_completions)} new)")

    # Summary
    print("\n--- Summary ---")
    for c in new_completions:
        print(f"  {c['phrase_id']}: edit_dist={c['edit_distance']}, matched={c['matched']}, type={c['type']}")


if __name__ == "__main__":
    main()
