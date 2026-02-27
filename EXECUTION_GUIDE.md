# Upgraded Audit Execution Guide

This guide covers the upgraded pipeline with:
- dual-corpus loading (`ASAP_2.0` + `LEAF`)
- robust live scoring for 800+ API calls
- intersectional DIF deep-dive (race, gender, SES, disability)
- LEAF cross-validation checks

## 1) Prerequisites

- Datasets already available locally:
  - `raw_data/ASAP/ASAP2_train_sourcetexts.csv`
  - `raw_data/LEAF/leaf.jsonl`
- Python dependencies installed:
  - `pip install -r requirements.txt`
- For live scoring:
  - `OPENAI_API_KEY` set in `.env`
  - `SCORING_MODE=live`

## 2) Default upgraded run profile

By default, `run_audit.py` now uses:
- `ASAP_SAMPLE_PER_GROUP=400` (about 800 ASAP essays total)
- `LEAF_SAMPLE_TOTAL=400`
- Combined workload: about **1,200 essays**

Run:

```bash
python run_audit.py
```

Live mode:

```bash
SCORING_MODE=live python run_audit.py
```

## 3) Throughput and rate-limit controls

`llm_scorer.py` supports configurable throttling and resilience:

- `OPENAI_MAX_REQUESTS_PER_MINUTE` (default `90`)
- `OPENAI_MIN_INTERVAL_SECONDS` (default `0.10`)
- `OPENAI_MAX_RETRIES` (default `5`)
- `OPENAI_RETRY_BASE_SECONDS` (default `2.0`)
- `SCORING_PROGRESS_EVERY` (default `25`)
- `SCORING_CHECKPOINT_EVERY` (default `25`)
- `ESTIMATED_COST_PER_ESSAY_USD` (default `0.004`)

Example conservative live profile:

```bash
set SCORING_MODE=live
set OPENAI_MAX_REQUESTS_PER_MINUTE=60
set OPENAI_MAX_RETRIES=6
set SCORING_PROGRESS_EVERY=20
set SCORING_CHECKPOINT_EVERY=20
python run_audit.py
```

## 4) Cost and time estimates

Using the default estimator (`$0.004` per essay):

| Total essays | Est. API cost | Est. wall time @ 90 RPM |
|---:|---:|---:|
| 800 | ~$3.20 | ~9-12 min |
| 1,200 (default upgraded run) | ~$4.80 | ~13-18 min |
| 1,600 | ~$6.40 | ~18-24 min |
| 2,000 | ~$8.00 | ~22-30 min |

Notes:
- Actual cost depends on final token usage per essay.
- Retries (especially on rate-limit spikes) increase time.
- Checkpointing allows safe resume after interruption.

## 5) Output artifacts

After a full run:

- `data/corpus.csv`
- `data/scored_corpus.csv`
- `data/analysis_results.json`
- `reports/figures/*.png`
- `reports/RESEARCH_REPORT.md`

The upgraded `analysis_results.json` additionally includes:
- `intersectional_asap_only`
- `leaf_cross_validation`
- `audit_metadata`

## 6) Recommended live command

For the upgraded end-to-end study with cross-validation:

```bash
set SCORING_MODE=live
set ASAP_SAMPLE_PER_GROUP=400
set LEAF_SAMPLE_TOTAL=400
set OPENAI_MAX_REQUESTS_PER_MINUTE=90
python run_audit.py
```
