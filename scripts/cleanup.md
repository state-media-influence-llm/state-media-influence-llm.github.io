# Scripts Cleanup (2026-03-08)

## What changed

Archived 7 one-time/superseded scripts to `scripts/archive/` and removed the
`query_audit.py` step from the GitHub Actions workflow.

## Archived scripts

| Script | Reason |
|--------|--------|
| `add_paper_memorization.py` | Paper examples already in completions.json |
| `prepare_study6.py` | Global page data already generated |
| `process_checkpoint_data.py` | Checkpoint JSONs already generated |
| `process_memorization.py` | Memorization phrases already generated |
| `query_paper_phrases.py` | Paper phrase queries already done |
| `rescore_memorization.py` | Rescoring already done |
| `query_audit.py` | Superseded by `run_audit_study4.py`; uses incompatible response schema |

Originals are in `scripts/archive/` (gitignored, local-only).

## Workflow change

Removed the `query_audit.py` step from `.github/workflows/update-data.yml`.
Running the old script would overwrite `data/audit/responses.json` with the
pre-Study 4 format, corrupting the audit page data.

`query_memorization.py` is kept — the memorization page still consumes its output.

## Active scripts (6)

| Script | Purpose |
|--------|---------|
| `env_utils.py` | Shared utility (API client, env loading) |
| `translate.py` | Shared utility (ZH→EN translation) |
| `query_memorization.py` | Live memorization queries (GH Actions) |
| `run_audit_study4.py` | Study 4 generation + judge pipeline |
| `process_study4_audit.py` | Generates `audit_summary.json` |
| `process_study4_responses.py` | Generates `prompts.json` + `responses.json` |
