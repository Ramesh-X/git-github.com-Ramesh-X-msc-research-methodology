# SRS: Phase 1.4 — Evaluation Query Generation

**Version:** 2.0 | **Date:** December 7, 2025

## Purpose

Generate 200 evaluation QA pairs (direct, multi-hop, negative) from synthetic KB for RAG pipeline testing.

## Scope

**Delivers:**
- 120 direct queries (single-page answers)
- 40 multi-hop queries (require 2+ pages)
- 40 negative queries (unanswerable)
- Output: `output/kb/data/queries.jsonl`

**Input**: `output/kb/data/structure.json` + `output/kb/*.md`

## Key Concepts

**Direct Queries (120)**
- Generated ONLY from current (v2) pages — v1 pages excluded
- Weighted random selection: pages with fewer queries selected more often for balanced distribution
- 1-2 queries per page

**Multi-hop Queries (40)**
- Require information from 2+ linked pages
- Ground truth uses only current versions

**Negative Queries (40)**
- Plausible but unanswerable from entire KB
- Ground truth: "I don't know based on the knowledge base."

## Requirements

### Core Functions

**1. Load KB**
- Parse `structure.json` from `output/kb/data/`
- Filter pages: exclude v1 (outdated) pages ending in `-v1.md`

**2. Generate Direct Queries**
- Track query count per page
- Weighted selection: `weight = max_count + 1 - current_count`
- LLM prompt: page content → generate question + answer
- Continue until 120 queries generated

**3. Generate Multi-hop Queries**
- Select linked page pairs from structure
- LLM prompt: combined content → question needing both pages
- Generate 40 queries

**4. Generate Negative Queries**
- Select anchor page (stratified by category)
- LLM prompt: KB summary + anchor content → unanswerable question
- Ground truth: refusal phrase
- Generate 40 queries

**5. Output JSONL**
- Write to `output/kb/data/queries.jsonl`
- One JSON object per line

## Data Schema

### queries.jsonl

```json
{
  "query_id": "q_direct_001",
  "query_type": "direct",
  "query": "What is the return window?",
  "ground_truth": "14 days from delivery.",
  "context_reference": ["returns-policy-v2.md"],
  "metadata": {
    "difficulty": "easy",
    "category": "returns_refunds"
  }
}
```

**Fields:**
- `query_id`: Unique identifier with prefix (q_direct_, q_multi_hop_, q_negative_)
- `query_type`: direct | multi_hop | negative
- `query`: Customer question text
- `ground_truth`: Correct answer or "I don't know based on the knowledge base."
- `context_reference`: Source file(s) — may be empty for negative queries
- `metadata`: Difficulty level and category

---

**End of SRS**
