# SRS: Phase 2.1 — E1 Baseline Experiment (No RAG)

**Version:** 2.0 | **Date:** December 7, 2025

## Purpose

Establish baseline LLM hallucination rate without retrieval-augmented generation for comparison with E2-E4 experiments.

## Scope

**Delivers:**
- Process 200 evaluation queries through LLM without RAG
- Output: `output/kb/data/e1_baseline.jsonl`

**Input**: `output/kb/data/queries.jsonl`

## Key Concepts

**Baseline Condition**: Direct LLM response without external knowledge retrieval

**Refusal Instruction**: System prompt includes "If you are unsure or don't have information to answer accurately, say 'I don't know'." — ensures baseline can refuse uncertain answers, not just hallucinate.

## Requirements

### Core Functions

**1. Load Configuration**
- Read environment variables (API keys, KB directory, flags)
- Derive input/output paths from KB_DIR

**2. Load Queries**
- Read from `output/kb/data/queries.jsonl`
- Parse 200 queries (all types)

**3. Initialize LLM Agent**
- System prompt: "You are a retail customer support assistant. Answer the question concisely. If you are unsure or don't have information to answer accurately, say 'I don't know'."
- Model via OpenRouter API

**4. Process Queries**
- For each query:
  - Skip if already processed (resume support)
  - Send query to LLM
  - Record answer, ground truth, metadata

**5. Output Results**
- Write to `output/kb/data/e1_baseline.jsonl`
- JSONL format (one object per line)
- Support resume via OVERWRITE flag

## Data Schema

### e1_baseline.jsonl

```json
{
  "query_id": "q_direct_001",
  "query": "What is the return window?",
  "llm_answer": "Returns are accepted within 30 days of delivery.",
  "ground_truth": "14 days from delivery.",
  "context_reference": ["returns-policy-v2.md"],
  "model": "amazon/nova-2-lite-v1",
  "dry_run": false
}
```

**Fields:**
- `query_id`: From input queries
- `query`: Question text
- `llm_answer`: LLM's response (no retrieval)
- `ground_truth`: Correct answer from queries
- `context_reference`: Source files (for reference only)
- `model`: LLM model identifier
- `dry_run`: Boolean flag

---

**End of SRS**
