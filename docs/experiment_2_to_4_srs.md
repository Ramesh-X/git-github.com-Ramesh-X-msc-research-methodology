# SRS: Phase 2.2-2.4 — E2-E4 RAG Pipeline Experiments

**Version:** 2.0 | **Date:** December 7, 2025

## Purpose

Evaluate three RAG pipeline variations to progressively reduce hallucinations through retrieval filtering and reasoning.

## Scope

**Delivers:**
- E2 (Standard RAG): Vector retrieval → LLM
- E3 (Filtered RAG): Vector retrieval → Reranking → LLM
- E4 (Reasoning RAG): Vector retrieval → Reranking → Chain-of-Thought LLM
- Output: `output/kb/data/e2_standard_rag.jsonl`, `e3_filtered_rag.jsonl`, `e4_reasoning_rag.jsonl`

**Input**: 
- `output/kb/*.md` (100 pages)
- `output/kb/data/queries.jsonl` (200 queries)

## Key Concepts

**Vector Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)

**Reranking**: Cross-encoder/ms-marco-MiniLM-L-6-v2 for precision improvement

**Chain-of-Thought**: Explicit step-by-step reasoning to verify context and identify conflicts

## Requirements

### Common Setup

**1. KB Indexing**
- Load all MD files, apply structure-aware chunking (512 tokens, 128 overlap)
- Generate embeddings, store in Qdrant (in-memory)
- Metadata per chunk: filename, section header

**2. Query Loading**
- Read 200 queries from `output/kb/data/queries.jsonl`

### E2: Standard RAG

**Process per query:**
1. Embed query
2. Vector search → retrieve top-5 chunks (cosine similarity)
3. Construct prompt: context + query
4. LLM generates answer
5. Record: query, chunks, answer, ground truth

**Output**: `output/kb/data/e2_standard_rag.jsonl`

### E3: Filtered RAG

**Process per query:**
1. Embed query
2. Vector search → retrieve top-20 chunks
3. **Rerank using cross-encoder** → select top-5 most relevant
4. Construct prompt: context + query
5. LLM generates answer
6. Record: query, reranked chunks, answer, ground truth

**Expected improvement**: Higher chunk relevance → fewer hallucinations

**Output**: `output/kb/data/e3_filtered_rag.jsonl`

### E4: Reasoning RAG

**Process per query:**
1. Embed query
2. Vector search → retrieve top-20 chunks
3. Rerank → select top-5
4. **Chain-of-Thought prompt**: "Break down problem → Check context → Identify conflicts → Verify completeness → Explain reasoning → Final answer"
5. LLM generates structured response with reasoning steps
6. Record: query, chunks, reasoning, answer, ground truth

**Expected improvement**: Explicit reasoning prevents hallucinations via self-checking

**Output**: `output/kb/data/e4_reasoning_rag.jsonl`

## Data Schema

### Experiment output (e2/e3/e4 .jsonl)

```json
{
  "query_id": "q_direct_001",
  "experiment": "e2_standard_rag",
  "query": "What is the return window?",
  "retrieved_chunks": [
    {
      "content": "Returns accepted within 14 days...",
      "source": "returns-policy-v2.md",
      "score": 0.89
    }
  ],
  "llm_answer": "The return window is 14 days from delivery.",
  "reasoning": "...",  // E4 only
  "ground_truth": "14 days from delivery.",
  "retrieval_time_ms": 45,
  "generation_time_ms": 1200
}
```

**Key fields:**
- `retrieved_chunks`: Top-5 chunks with content, source, relevance score
- `reasoning`: Chain-of-thought steps (E4 only)
- Timing metrics for analysis

## Technical Stack

- **Vector DB**: Qdrant (in-memory mode)
- **Embeddings**: OpenAI text-embedding-3-small
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **LLM**: Via OpenRouter (configurable model)
- **Framework**: Pydantic-AI for structured outputs

---

**End of SRS**
