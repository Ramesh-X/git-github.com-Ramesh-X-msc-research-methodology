# SRS: Phase 1.4 — Evaluation Query Generation

**Version:** 2.0 | **Date:** December 7, 2025

## Purpose

Generate 200 evaluation QA pairs (direct, multi-hop, negative) from synthetic KB for RAG pipeline testing.

## Scope

**Delivers:**
- 90 direct queries (single-page answers) with 6 subtypes
- 70 multi-hop queries (require 2+ pages) with 6 subtypes
- 40 negative queries (unanswerable) with 4 subtypes
- Output: `output/kb/data/queries.jsonl` (200 total)

**Input**: `output/kb/data/structure.json` + `output/kb/*.md`

## Key Concepts

**Direct Queries (90, ADVERSARIAL SUBTYPES)**
- Generated ONLY from current (v2) pages — v1 pages excluded
- Weighted random selection: pages with fewer queries selected more often for balanced distribution
- **7 subtypes** for adversarial testing:
  - **Simple fact extraction**: Single fact (price, date, contact info)
  - **Table aggregation (NEW)**: Requires multi-row calculation or conditional logic (e.g., "Cheapest shipping for 7lb package to Zone B?")
  - **Table lookup**: Answer requires reading specific table cells
  - **Process step identification**: Extract steps from procedures
  - **Conditional logic**: "What if..." scenario-based questions
  - **List enumeration**: "What are all..." complete lists
  - **Rot-aware**: Target current (v2) versioned content with temporal keywords ("current", "latest", "as of 2024")

**Multi-hop Queries (70, TRANSITIVE ENFORCEMENT)**
- Require information from 2+ linked pages (NOT just concatenation)
- **Enhanced subtype design** to force genuine transitive reasoning:
  - **Sequential process**: Steps from different procedural pages; requires reading both to complete process
  - **Policy-FAQ cross-reference**: Policy rule + FAQ clarification needed (e.g., "Returns 30 days BUT FAQ says exceptions for clearance")
  - **Comparative**: Compare data from two pages (e.g., rate tables side-by-side)
  - **Hub-to-detail navigation (NEW)**: Hub page MUST provide only navigation context; detail page contains specific answer. Both pages required to answer.
  - **Cross-category**: Span different topics with hidden dependencies (e.g., "If I use loyalty points for payment, what refund do I get?")
  - **Rot-aware**: One page is v2, other is regular; distinguish versions to get correct answer
- Ground truth uses only current (v2) versions
- Circular references (A ↔ B) in some queries to test loop detection

**Negative Queries (40, LEXICAL TRAPS)**
- Plausible but unanswerable from entire KB
- **Enhanced subtype design** with high lexical overlap to punish vector-only retrieval:
  - **Adjacent topic**: Uses 80%+ keywords from anchor page but asks about missing detail (e.g., anchor has "Payment Methods: Visa, PayPal", ask "What is refund timeline for PayPal?")
  - **Missing data**: Ask for specific data point not in KB (e.g., product SKU, specific rate) but with high keyword overlap
  - **Out-of-scope procedure**: Ask about related edge case not documented (e.g., "Returns Policy" covers standard, ask about "gift returns")
  - **Cross-category gap**: Question spanning categories with no explicit link; KB has all pieces but no connection (e.g., "Loyalty points + returns refund interaction?")
- Ground truth: "I don't know based on the knowledge base."
- Validation: Ensure cosine similarity with anchor page > 0.70 (high overlap) but answer genuinely missing

## Requirements

### Core Functions

**1. Load KB**
- Parse `structure.json` from `output/kb/data/`
- Filter pages: exclude v1 (outdated) pages ending in `-v1.md`

**2. Generate Direct Queries**
- Track query count per page
- Weighted selection: `weight = max_count + 1 - current_count`
- LLM prompt: page content + subtype instruction → generate question + answer
- Continue until 90 queries generated

**3. Generate Multi-hop Queries**
- Select linked page pairs (strategic pairing by semantic relationships)
- LLM prompt: combined content + subtype instruction → question needing both pages
- Verify question requires BOTH pages
- Generate 70 queries

**4. Generate Negative Queries**
- Select anchor page (stratified by category)
- LLM prompt: KB summary + anchor content → unanswerable question
- Verify question is unanswerable across entire KB
- Ground truth: refusal phrase ("I don't know based on the knowledge base.")
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
    "subtype": "simple_fact",
    "category": "returns_refunds"
  }
}
```

**Fields:**
- `query_id`: Unique identifier with prefix (q_direct_, q_multi_hop_, q_negative_)
- `query_type`: direct | multi_hop | negative
- `query`: Customer question text
- `ground_truth`: Correct answer or "I don't know based on the knowledge base."
- `context_reference`: Source file(s) — one anchor for negative queries
- `metadata`: Subtype and category

---

**End of SRS**
