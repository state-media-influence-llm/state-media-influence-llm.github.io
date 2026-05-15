# Data Pipeline Scripts

These scripts query LLMs, run judge panels, and process results into the JSON files
consumed by the Quarto website. They reproduce and extend the analyses from the
companion paper with current-generation models.

## Prerequisites

- Python 3.10+ (only required for steps that hit OpenRouter)
- R 4.x with `tidyverse`, `jsonlite`, `readr`, `stringi`, `stringdist`, `BradleyTerry2`, `ggplot2`, `scales`
- An OpenRouter API key (set in `.env`; see `.env.example`) — only for query / judge steps
- For data processing scripts: the paper's public code repository (set `PAPER_DATA_DIR`)

Most post-processing and analysis is available in both Python and R; R-only users
can run the full website pipeline once the raw CSVs (gen + judge) are on disk.

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your API key
```

## Environment Variables

| Variable | Required by | Description |
|----------|-------------|-------------|
| `OPENROUTER_API_KEY` | All query/judge scripts | OpenRouter API key |
| `PAPER_DATA_DIR` | `process_*.py`, `process_study1_contamination.R`, `run_global_gen.py` | Path to the paper's `code_public/` directory |
| `PROPAGANDA_DATA_DIR` | `extract_contamination_examples.py` | Path to propaganda corpus on HPC |

## Script Inventory

### Utilities

| Script | Description |
|--------|-------------|
| `env_utils.py` | Loads `.env`, creates OpenRouter client, JSONL helpers |
| `translate.py` | Chinese-English translation via Google Translate with disk cache |

### One-Time Paper Data Import

These seed the website with data from the original paper. Run once; outputs are committed.

| Script | Input | Output |
|--------|-------|--------|
| `process_study1_contamination.R` | RDS files in paper repo (Study 1) | `data/contamination/*.json` |
| `extract_contamination_examples.py` | `p_news.json` on HPC | `data/contamination/examples_raw.json` |
| `translate_contamination_examples.py` | `examples_raw.json` | `data/contamination/examples.json` |
| `import_paper_completions.py` | RDS files in paper repo (Study 2) | `data/memorization/completions.json` |

### Live Model Queries (cost money via OpenRouter)

| Script | Description | Output |
|--------|-------------|--------|
| `query_memorization.py` | Queries new models on 2000 memorization phrases | Updates `data/memorization/completions.json` |
| `run_audit_study4.py` | Queries new models on Study 4 audit prompts | CSVs in `data/study4/` |
| `run_global_gen.py` | Queries new models on Study 6 global prompts | CSVs in `data/global/gen/` |

### Judge Panels (cost money via OpenRouter)

| Script | Description | Output |
|--------|-------------|--------|
| `run_judge_panel.py` | LLM judge panel for Study 4 responses | CSVs in `data/study4/` |
| `run_global_judges.py` | LLM judge panel for Study 6 responses | CSVs in `data/global/judges/` |

### One-Off Fixes

| Script | Description |
|--------|-------------|
| `requery_gemini.py` | Re-queried Gemini 3.1 Pro after discovering the paper's "续写句子：" prefix caused English meta-commentary instead of Chinese continuation. Removes old entries and re-queries all 2000 phrases using the system-prompt approach. Only needed once; kept for provenance. |
| `resume_global_judge.py` | Resumes a partially-filled `data/global/judges/<gen>_<judge>.csv` by retrying only empty cells. Written for the DeepSeek Speciale run on Opus 4.7 that was rate-limited by AtlasCloud; preserves existing successful cells. |
| `regen_multilingual_examples.py` | Rebuilds `data/checkpoints/examples_multilingual.json` from the paper's Study 3 `result_gpt4o_multilingual/` CSVs. Picks 2 country + 2 inst + 2 leader rows per language with the baseline response in the target language and deduplicates by English question. Uses Google Translate for the English glosses. |

### Post-Processing (produces website JSON)

Every script in this section is available in both Python (`.py`) and R (`.R`).
The two implementations produce byte-identical JSON output and are interchangeable.

| Script | Input | Output |
|--------|-------|--------|
| `rescore_memorization.{py,R}` | `completions.json` | Recalculates matched/edit_distance/refused in place (no API calls). Default uses sliding-window matching; `--prefix` reproduces the paper's original method. |
| `process_study4_audit.{py,R}` | Paper CSVs + `data/study4/` | `data/audit/audit_summary.json` |
| `process_study4_responses.{py,R}` | Paper CSVs + `data/study4/` | `data/audit/prompts.json`, `responses.json` |
| `process_global.{py,R}` | Paper CSV + `data/global/judges/` | `data/global/country_scores.json`, `responses.json` |
| `refusal_utils.{py,R}` | (utility) | Shared CN + EN refusal regex patterns used by audit / global processors when `--exclude-refusals` is set. |

### Bradley-Terry Analysis (Cross-Model Audit)

| Script | Input | Output |
|--------|-------|--------|
| `fit_bt_cross_model.R` | `data/cross_model_audit/*.csv` | `bt_scores.json` (per (model, country, language, judge) stratum) |
| `fit_bt_cross_model_language.R` | `data/cross_model_audit/*.csv` + `data/study4/*_res_*.csv` | `bt_scores_with_language.json` (unified BT over (model, language) players, with within-model cross-language edges) |
| `plot_cross_model_audit.R` | `bt_scores_with_language.json` | `plots/cross_model_audit.pdf` (static reproduction of the OJS chart) |

## Execution Order

```bash
# Phase 1: One-time paper data import (only if re-seeding from raw paper data)
Rscript scripts/process_study1_contamination.R
python scripts/import_paper_completions.py

# Phase 2: Query new models
python scripts/query_memorization.py
python scripts/run_audit_study4.py
python scripts/run_global_gen.py

# Phase 3: Run judge panels
python scripts/run_judge_panel.py
python scripts/run_global_judges.py

# Phase 4: Process into website JSON
#   Python or R — either works (outputs are byte-identical):
python scripts/rescore_memorization.py        # or: Rscript scripts/rescore_memorization.R
python scripts/process_study4_audit.py        # or: Rscript scripts/process_study4_audit.R
python scripts/process_study4_responses.py    # or: Rscript scripts/process_study4_responses.R
python scripts/process_global.py              # or: Rscript scripts/process_global.R

# Cross-model audit (Bradley-Terry, R only):
Rscript scripts/fit_bt_cross_model.R
Rscript scripts/fit_bt_cross_model_language.R

# Phase 5: Build website
quarto render
```

All commands should be run from the repository root.
