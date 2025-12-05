# Software Requirements Specification (SRS): Evaluation Query Generation for Synthetic Retail KB

**Version:** 1.0  
**Date:** December 3, 2025  
**Author:** AI Assistant (based on `proposed_method_summary.md` Phase 1.4)  
**Purpose:** Guide for Phase 1.4 of MSc research — generate ~150 evaluation QA pairs (direct, multi-hop, negative) from the synthetic retail customer support KB for RAG pipeline testing (see `proposed_method_summary.md`).

## 1. Introduction

### 1.1 Purpose
This SRS specifies a Python tool to generate evaluation queries and ground truth answers from the synthetic KB (100 MD files + `structure.json`). Queries test RAG pipelines on direct retrieval, multi-hop reasoning, and hallucination resistance (negative cases). Total: ~150 QA pairs stored as `queries.jsonl`.

### 1.2 Scope
**In Scope:**
- Load `structure.json` and `output/kb/*.md`.
- Generate direct (~100), multi-hop (~25), negative (~25) queries with ground truth.
- Output `output/kb/queries.jsonl`.

**Out of Scope:** RAG experiments (Phase 2), metric evaluation (Phase 3).

**Assumptions:** KB generator complete; OpenRouter/GPT access for query synthesis.

### 1.3 Definitions & Acronyms
- **Direct Query**: Answerable from 1 page.
- **Multi-hop Query**: Requires 2+ linked pages.
- **Negative Query**: Plausible but unanswerable ("I don't know").
- **Ground Truth**: Precise answer or refusal phrase.

### 1.4 References
- `proposed_method_summary.md` (Phase 1.4 details).
- `sythetic_retail_customer_support_srs.md` (KB generator).

## 2. Overall Description

### 2.1 Product Perspective
Builds on KB generator: Uses page content/links to create realistic customer queries for RAG eval (e.g., "What's the refund window for sale items?").

### 2.2 Product Functions
1. Load KB metadata/content.
2. Generate queries per type.
3. Compute ground truth via LLM.
4. Export JSONL.

### 2.3 User Classes & Characteristics
- **Researchers**: Run post-KB generation for eval dataset.

## 3. Specific Requirements

### 3.1 External Interfaces
- **Input:** `output/kb/structure.json`, `output/kb/*.md`.
- **Output:** `output/kb/queries.jsonl`.
- **Dependencies:** `pydantic-ai-slim[openrouter]` (via OpenRouter), `pydantic`.

### 3.2 Functional Requirements

**FR1: Load Inputs**
- Parse `structure.json` (Pydantic).
- Read MD content by filename.

**FR2: Direct Queries (~100)**
- 1 query/page: Prompt LLM with page content → extract question + answer.

**FR3: Multi-hop Queries (~25)**
- Select linked page pairs from `links_to`.
- Prompt with combined content → question needing both.

**FR4: Negative Queries (~25) — Anchored Negative Queries**
- Generate anchored negative queries that are semantically close to a specific page (anchor) but unanswerable by the entire KB.
- Use stratified sampling by category to select ~25 anchor pages.
- Provide the LLM: anchor page content, direct linked page content, anchor page metadata, and the full KB topic summary (titles/categories/topics) so the model can craft adversarial questions that sound answerable from the anchor page but are not present in the KB.
- The prompt should instruct: 'Make a question that looks like it could be answered by the anchor page but cannot be answered by any page in the KB; set the ground_truth to a refusal phrase ("I don't know based on the KB.")'.
- Questions must be unanswerable by the **entire KB** (not just one page) and should be specific enough to deceive naive RAG systems but require reranking and CoT to be correctly refused.
- Set `context_reference` to `[anchor_page.filename]` to denote the anchor page used as bait for evaluation.

**FR5: Output JSONL**
- Schema per line (see 4.1).

### 3.3 Non-Functional Requirements
- **Performance:** <30 min for 200 queries.
- **Usability:** CLI: `python generate_queries.py --kb_dir output/kb`.
- **Reliability:** 95% valid queries; dry-run mode.
- **Portability:** Python 3.10+.

## 4. Data Requirements

### 4.1 queries.jsonl Schema
```json
{
  "query_id": "q_direct_001",
  "query_type": "direct|multi_hop|negative",
  "query": "What is the return window?",
  "ground_truth": "30 days from delivery.",
  "context_reference": ["returns-policy.md"],  // Empty [] for negative queries
  "metadata": {
    "difficulty": "easy|medium|hard",
    "category": "returns_refunds"
  }
}
```

## 5. Supporting Information

### 5.1 Generation Algorithm Pseudocode
```python
def generate_queries(kb_dir: str):
    structure = Structure.parse_file(f"{kb_dir}/structure.json")
    
    # Use pydantic-ai agents with structured output (QueryResponse model)
    # Agents handle LLM prompting, JSON parsing, validation, and retries
    
    queries = []
    
    # Direct: 1/page
    for page in structure.pages:
        content = read_md(f"{kb_dir}/{page.filename}")
        qa = direct_agent.run_sync(build_direct_prompt(content)).output
        queries.append(qa)
    
    # Multi-hop: linked pairs
    pairs = find_linked_pairs(structure)
    for pair in pairs[:25]:
        content = read_md(pair.files)
        qa = multi_hop_agent.run_sync(build_multi_hop_prompt(content)).output
        queries.append(qa)
    
    # Negative
    for _ in range(25):
        qa = negative_agent.run_sync(build_negative_prompt(sample_topics())).output
        queries.append(qa)
    
    save_jsonl(queries, f"{kb_dir}/queries.jsonl")
```

### 5.2 Sample Prompts
**Direct**: "From this page content only, generate 1 customer question and exact answer: [content]"

**Multi-hop**: "Generate question needing BOTH pages: [content1] [content2]"

**Negative (anchored)**: "Plausible retail query NOT answered by the entire KB — anchor page content: [anchor_content], linked content: [linked_page_contents], KB topics: [kb_summary] → 'I don't know based on the KB.'"

### 5.3 Sample Output
```json
{"query_id": "q_direct_001", "query_type": "direct", "query": "Return window?", "ground_truth": "30 days.", "context_reference": ["refund_policy.md"], "metadata": {"difficulty": "easy", "category": "returns"}}
{"query_id": "q_negative_001", "query_type": "negative", "query": "Is there a 60-day warranty?", "ground_truth": "I don't know based on the KB.", "context_reference": ["warranty-page-31.md"], "metadata": {"difficulty": "hard", "category": "warranty"}}
```

## 6. Verification & Validation
- Unit tests: Prompt parsing, schema validation.
- Acceptance: ~150 queries, type ratios (~100/25/25), valid JSONL.

**End of SRS**
