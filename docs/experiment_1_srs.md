# Software Requirements Specification (SRS): Experiment 1 (E1) — Baseline LLM (No RAG)

**Version:** 1.0  
**Date:** December 6, 2025  
**Author:** AI Assistant (based on `proposed_method_summary.md` Phase 2)  
**Purpose:** Guide for Phase 2, Experiment 1 (E1) of MSc research — establish baseline LLM hallucination tendency without RAG (see `proposed_method_summary.md`).

## 1. Introduction

### 1.1 Purpose

This SRS specifies the baseline experiment (E1) for measuring LLM hallucination in customer service chatbots. E1 runs evaluation queries directly through an LLM **without retrieval-augmented generation (RAG)**, establishing a control condition for comparison with E2 (Standard RAG), E3 (Filtered RAG), and E4 (Reasoning RAG).

### 1.2 Scope

**In Scope:**
- Load evaluation queries from `output/kb/queries.jsonl` (generated in Phase 1.4).
- For each query, call LLM with minimal system prompt ("You are a retail customer support assistant. Answer concisely.").
- Record LLM answer, ground truth, and metadata.
- Output results to `output/kb/e1_baseline.jsonl`.
- Support dry-run mode for testing without API calls.
- Support resume via `OVERWRITE` flag.

**Out of Scope:** RAG retrieval, metric evaluation (Phase 3), production UI/deployment.

**Assumptions:** Evaluation queries available; OpenRouter API access for LLM calls.

### 1.3 Definitions & Acronyms

- **E1**: Experiment 1 (baseline, no RAG).
- **LLM**: Large Language Model (e.g., Grok-4.1-fast).
- **Hallucination**: Plausible-sounding but incorrect/fabricated answer.
- **Ground Truth**: Reference answer or "I don't know" (from Phase 1.4).
- **Dry-run**: Test mode without API calls (placeholder answers).

### 1.4 References

- `proposed_method_summary.md` (Phase 2 overview).
- `evaluation_query_generation_srs.md` (Phase 1.4 query generation).

## 2. Overall Description

### 2.1 Product Perspective

E1 is the first of four experimental conditions testing architectural strategies to reduce hallucination. By running queries without RAG, E1 captures the baseline hallucination rate and knowledge cutoff limitations, providing a control against which retrieval, reranking, and reasoning strategies are measured.

### 2.2 Product Functions

1. Load evaluation queries (~150 QA pairs).
2. Send each query to LLM without external knowledge.
3. Record raw LLM answers.
4. Export results for Phase 3 evaluation.

### 2.3 User Classes & Characteristics

- **Researchers**: Run E1 as part of Phase 2 experimental pipeline for MSc research.

## 3. Specific Requirements

### 3.1 External Interfaces

- **Input:** `output/kb/queries.jsonl` (from Phase 1.4).
- **Output:** `output/kb/e1_baseline.jsonl` (JSONL lines per query result).
- **Dependencies:** OpenRouter API, `pydantic-ai-slim[openrouter]`, `pydantic`, `python-dotenv`, `tqdm`.

### 3.2 Functional Requirements

**FR1: Load Configuration**
- Read `.env` file for `KB_DIR`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `DRY_RUN`, `OVERWRITE`.
- Derive paths: `queries_file = {KB_DIR}/queries.jsonl`, `output_file = {KB_DIR}/e1_baseline.jsonl`.

**FR2: Load Evaluation Queries**
- Read JSONL from `queries_file`.
- Parse each line into query object with fields: `query_id`, `query`, `ground_truth`, `context_reference`.

**FR3: Initialize LLM Agent**
- If not dry-run: Create Pydantic AI agent with system prompt "You are a retail customer support assistant. Answer concisely."
- If dry-run: Skip LLM initialization (no API calls).

**FR4: Process Queries**
- Iterate over queries with progress bar.
- For each query:
  - If `query_id` exists in output and `OVERWRITE=false`: Skip.
  - If dry-run: Set answer to `"[DRY_RUN] No LLM call"`.
  - If production: Call LLM agent with query text; extract answer.
  - Build result object and write to output file.

**FR5: Output Results**
- Write each result as JSONL line (single JSON object per line).
- Result fields: `query_id`, `query`, `llm_answer`, `ground_truth`, `context_reference`, `model`, `dry_run`.

**FR6: Resume & Overwrite**
- If `OVERWRITE=false` (default): Append to existing output file; skip already-processed query_ids.
- If `OVERWRITE=true`: Truncate output file and reprocess all queries.

**FR7: Logging**
- File-only logging to `LOG_FILE` (default: `logs/e1_baseline.log`).
- Log key events: start, config, query counts, completion.
- No console output except progress bar.

### 3.3 Non-Functional Requirements

- **Performance:** Dry-run on ~150 queries < 10 seconds; API-based run ~1-5 minutes (OpenRouter latency).
- **Usability:** Simple CLI: `python main.py` (config via `.env`).
- **Reliability:** Graceful error handling; skip failed queries with logging.
- **Portability:** Python 3.10+, Linux/Mac.

## 4. Data Requirements

### 4.1 Input: queries.jsonl Schema

```json
{
  "query_id": "q_direct_001",
  "query": "What is the return window?",
  "ground_truth": "30 days from delivery.",
  "context_reference": ["returns-policy.md"]
}
```

**Volume:** ~150 queries (100 direct, 25 multi-hop, 25 negative).

### 4.2 Output: e1_baseline.jsonl Schema

```json
{
  "query_id": "q_direct_001",
  "query": "What is the return window?",
  "llm_answer": "30 days from the date of delivery.",
  "ground_truth": "30 days from delivery.",
  "context_reference": ["returns-policy.md"],
  "model": "x-ai/grok-4.1-fast:free",
  "dry_run": false
}
```

## 5. Supporting Information

### 5.1 Configuration (`.env.example`)

```dotenv
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
OPENROUTER_MODEL=x-ai/grok-4.1-fast:free
KB_DIR=output/kb
OVERWRITE=false
DRY_RUN=false
LOG_FILE=logs/e1_baseline.log
```

### 5.2 Directory Structure

```
3_e1_baseline/
├── main.py                # Facade: loads config, calls pipeline
├── dry_run.py             # Test script (DRY_RUN=true)
├── logging_config.py      # File-only logging setup
├── .env.example           # Config template
├── requirements.txt       # Dependencies
└── e1_baseline_lib/
    ├── __init__.py
    ├── constants.py       # Defaults
    ├── models.py          # Pydantic models
    ├── agents.py          # LLM agent factory
    └── pipeline.py        # Core run_e1_baseline() logic
```

### 5.3 Sample Execution

```bash
# Dry-run test
python ./3_e1_baseline/dry_run.py

# Production run
python ./3_e1_baseline/main.py

# Re-run with new results
OVERWRITE=true python ./3_e1_baseline/main.py
```

### 5.4 Error Handling

| Scenario | Behavior |
|----------|----------|
| Missing queries file | Raise clear error; log and exit |
| Invalid JSON in JSONL | Skip line with warning log |
| LLM API timeout | Log error, set answer to "Error: {msg}", continue |
| Missing API key (production) | Raise error before processing |

## 6. Acceptance Criteria

- [ ] **AC1:** Dry-run completes successfully on ~150 queries without API calls.
- [ ] **AC2:** Output file `output/kb/e1_baseline.jsonl` is valid JSONL with 150 records (dry-run).
- [ ] **AC3:** Each record has all required fields.
- [ ] **AC4:** Resume works: re-run with `OVERWRITE=false` skips existing query_ids.
- [ ] **AC5:** Overwrite works: re-run with `OVERWRITE=true` replaces output.
- [ ] **AC6:** Logging is file-only; no unintended console output.
- [ ] **AC7:** Configuration via `.env` is intuitive.

## 7. Verification & Validation

- **Unit tests:** Model parsing/serialization, agent creation.
- **Integration test:** Run `dry_run.py` end-to-end; inspect output JSONL.
- **Smoke test:** Verify output compatible with Phase 3 evaluation input schema.

**End of SRS**
