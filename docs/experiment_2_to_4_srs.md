# Software Requirements Specification (SRS): E2-E4 RAG Pipeline Experiments

**Version:** 1.0  
**Date:** December 6, 2025  
**Author:** BA Team (based on `proposed_method_summary.md` Phase 2)  
**Purpose:** Guide for Phase 2 of MSc research — validate architectural design patterns to minimize LLM hallucinations in RAG pipelines through controlled experimental comparison of E2, E3, and E4 pipeline variations.

## 1. Introduction

### 1.1 Purpose

This SRS specifies a Python-based experimental framework to evaluate three RAG pipeline variations (E2, E3, E4) designed to progressively mitigate hallucinations in domain-specific customer service chatbots. The experiments compare:

- **E2 (Standard RAG)**: Vector retrieval + LLM generation baseline
- **E3 (Filtered RAG)**: Enhanced retrieval precision through cross-encoder reranking
- **E4 (Reasoning RAG)**: Chain-of-Thought reasoning with explicit verification steps

These experiments validate whether structured knowledge bases combined with retrieval filtering and reasoning can reduce hallucinations, setting architectural guidelines for production RAG systems.

### 1.2 Scope

**In Scope:**

- Execute three parallel RAG pipelines (E2, E3, E4) on ~150 evaluation queries.
- Index synthetic retail KB (100 MD files) into vector database with structure-aware chunking.
- Measure retrieval performance, LLM generation accuracy, and reasoning quality.
- Generate auditable experimental results for evaluation and analysis.
- Support dry-run testing mode for rapid prototyping and debugging.

**Out of Scope:**

- Metric computation and statistical analysis (Phase 3).
- Visualization and report generation (Phase 4).
- Production deployment or real-time inference optimization.
- User interface or chat system integration.

**Assumptions:**

- Synthetic KB (~100 pages, structure.json) is pre-generated and valid.
- Evaluation queries (~150 QA pairs) are prepared in queries.jsonl format.
- OpenRouter API and OpenAI embeddings API are accessible.
- Vector database runs in-memory for consistent, reproducible results.

### 1.3 Definitions & Acronyms

- **RAG**: Retrieval-Augmented Generation (retrieve external docs, then generate).
- **E2/E3/E4**: Experiment variants 2, 3, 4 (E1 is baseline without RAG).
- **Hallucination**: LLM generates plausible but factually unsupported content.
- **KB**: Knowledge Base (indexed collection of Markdown documents).
- **Reranking**: Re-ordering retrieved documents by relevance using a cross-encoder.
- **CoT**: Chain-of-Thought — explicit step-by-step reasoning.
- **Chunking**: Breaking documents into smaller, semantically meaningful segments.
- **Ground Truth**: Correct answer from evaluation query dataset.

### 1.4 References

- `proposed_method_summary.md` (Phase 2 design, E1-E4 definitions).
- `evaluation_query_generation_srs.md` (Query dataset structure).
- `synthetic_retail_customer_support_srs.md` (KB generation details).

---

## 2. Overall Description

### 2.1 Product Perspective

Builds on Phase 1 (KB + query generation) to implement and compare RAG architectures. Each experiment processes the same 150 queries through progressively sophisticated retrieval and reasoning pipelines, enabling quantitative and qualitative assessment of architectural improvements.

**Success Criteria:**
- All experiments complete processing 150 queries without errors.
- Output results are reproducible and auditable (queries, retrieved chunks, answers logged).
- Dry-run mode enables rapid iteration on logic without API costs.
- Framework supports manual inspection of failure cases for qualitative analysis.

### 2.2 Product Functions

1. **KB Indexing**: Load Markdown files, apply structure-aware chunking, generate embeddings, populate vector database.
2. **Query Processing**: For each evaluation query, execute three RAG pipelines in parallel.
3. **Retrieval (Core)**: Search vector database for relevant chunks; optionally rerank for precision.
4. **LLM Generation**: Pass retrieved context + query to LLM agents with pipeline-specific prompts.
5. **Reasoning Tracking**: For E4, explicitly capture step-by-step reasoning process.
6. **Result Persistence**: Write experimental outputs to structured JSONL files with provenance.

### 2.3 User Classes & Characteristics

- **Researchers**: Conduct E2-E4 experiments, analyze hallucination rates, compare architectures.
- **DevOps/System Engineers**: Monitor execution, manage costs, handle errors and retries.
- **Data Scientists**: Inspect results, validate ground truth, identify failure patterns.

---

## 3. Specific Requirements

### 3.1 External Interfaces

**Inputs:**
- `output/kb/structure.json` — KB topology and metadata.
- `output/kb/*.md` — 100 Markdown pages.
- `output/kb/queries.jsonl` — ~150 evaluation QA pairs.
- `.env` file — API keys and control flags.

**Outputs:**
- `output/kb/e2_standard_rag.jsonl` — E2 experiment results.
- `output/kb/e3_filtered_rag.jsonl` — E3 experiment results.
- `output/kb/e4_reasoning_rag.jsonl` — E4 experiment results.
- `logs/e2_to_e4.log` — Execution log with debug info.

**Dependencies:**
- OpenRouter API (LLM inference via proxy).
- OpenAI Embeddings API (text-embedding-3-small).
- Qdrant Vector Database (in-memory mode).
- Hugging Face Sentence-Transformers (cross-encoder reranking).

### 3.2 Functional Requirements

#### **FR1: Initialize Experimental Environment**

**Description:** Load configuration, validate API keys, initialize vector store and services.

**Business Logic:**
- Read KB_DIR and control flags (DRY_RUN, OVERWRITE) from environment.
- Validate OpenRouter API key (required unless DRY_RUN=true).
- Validate OpenAI API key for embeddings (required unless DRY_RUN=true).
- Initialize in-memory Qdrant instance for reproducibility.
- Load embedding service (OpenAI text-embedding-3-small) and reranking service (cross-encoder/ms-marco-MiniLM-L-6-v2).
- Log all configuration and initialization steps.

**Acceptance Criteria:**
- Environment loads without errors when both API keys present.
- DRY_RUN mode works without API keys (uses mock services).
- All services initialized before first query processed.

---

#### **FR2: Index Knowledge Base**

**Description:** Load Markdown files, apply structure-aware chunking, generate embeddings, populate vector database.

**Business Logic:**

- **Load Documents**: Read all `.md` files from KB_DIR (excluding README.md).
- **Structure-Aware Chunking**: 
  - Split each document by Markdown headers to preserve semantic structure.
  - Maintain chunks ≤512 tokens with 128-token overlap for context continuity.
  - Metadata per chunk: filename, section header, chunk index.
- **Embedding Generation**:
  - Batch embed all chunks using OpenAI text-embedding-3-small (1536 dimensions).
  - Process in batches of 100 to manage API rate limits.
  - Cache embeddings to avoid recomputation.
- **Vector Database Population**: 
  - Upsert embeddings + metadata into Qdrant collection.
  - Use cosine distance for similarity search (standard for semantic tasks).
  - Support in-memory mode for reproducibility across runs.

**Resume Logic:**
- If vector store already contains indexed chunks, skip re-indexing (unless OVERWRITE=true).
- Log number of chunks indexed and collection statistics.

**Acceptance Criteria:**
- KB with 100 pages → ~500-800 chunks (depending on content length).
- All chunks searchable via cosine similarity.
- Index creation time < 15 minutes for full KB.
- Index survives batch embedding failures (retry with exponential backoff).

---

#### **FR3: Load and Validate Queries**

**Description:** Load evaluation queries from JSONL, validate schema, prepare for processing.

**Business Logic:**
- Parse `queries.jsonl` into QueryInput objects (validate schema).
- Load query_id, query_type, query text, ground_truth, context_reference, metadata.
- Support all query types: direct, multi_hop, negative.
- Log total query count and distribution by type.

**Acceptance Criteria:**
- Load 150 queries in < 1 second.
- Schema validation catches malformed entries.
- Query types correctly identified for result labeling.

---

#### **FR4: Execute E2 Experiment (Standard RAG)**

**Description:** Process all queries through standard vector retrieval → LLM generation pipeline.

**Business Logic:**

1. **For each query:**
   - Embed query text using same embedding model (OpenAI text-embedding-3-small).
   - Search vector store for top-K=5 most similar chunks (cosine similarity).
   - Format retrieved chunks as numbered sources: "[Source 1: filename] chunk_text".
   - Construct prompt: "Context:\n[numbered_sources]\n\nQuestion: [query_text]"

2. **LLM Generation:**
   - Send prompt to OpenRouter LLM (x-ai/grok-beta or configured model).
   - System prompt: "You are a retail support assistant. Use context to answer. Cite sources. Say 'I don't know' if insufficient info."
   - Capture answer from structured output (E2Response.answer).
   - Record latency, token usage, and any API errors.

3. **Result Recording:**
   - Persist to `e2_standard_rag.jsonl`: query_id, experiment, query, retrieved_chunks, llm_answer, ground_truth, metrics.
   - Append mode (support resumption on crash).
   - Flush after each result to disk (fsync) for crash recovery.

**Performance Characteristics:**
- Retrieval: ~50-100ms per query.
- LLM generation: ~1-3 seconds per query (depending on answer length).
- Total: ~1.5-3.5 seconds per query × 150 queries ≈ 4-9 minutes wall time.

**Acceptance Criteria:**
- Process all 150 queries without failure.
- Retrieved chunks include valid metadata (filename, section).
- LLM answers are non-empty strings.
- Resume capability: skip already processed queries on re-run.

---

#### **FR5: Execute E3 Experiment (Filtered RAG with Reranking)**

**Description:** Enhanced retrieval precision via cross-encoder reranking before LLM generation.

**Business Logic:**

1. **For each query:**
   - Embed query (same as E2).
   - **Initial Retrieval**: Retrieve top-N=20 chunks (broader search candidate set).
   - **Reranking**: 
     - Use cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2) to score all 20 chunks.
     - Rerank by cross-encoder score (higher = more relevant to query).
     - Select top-K=5 from reranked list.
   - Format retrieved chunks (same as E2).

2. **LLM Generation:**
   - Same system prompt as E2 (or variant noting "context has been filtered for relevance").
   - Send prompt with reranked chunks.
   - Capture answer.

3. **Result Recording:**
   - Persist to `e3_filtered_rag.jsonl`.
   - Track both original vector search scores and cross-encoder rerank scores.
   - Record reranking time separately from retrieval time.

**Expected Improvement Over E2:**
- Higher relevance of top-5 chunks (reranker prioritizes semantic match over embedding distance).
- Potentially better answer quality (more precise context → fewer hallucinations).

**Acceptance Criteria:**
- Reranking time < 500ms per query (cross-encoder is fast).
- All 150 queries processed.
- Reranked chunks have higher relevance than E2 (qualitative review).

---

#### **FR6: Execute E4 Experiment (Reasoning RAG with Chain-of-Thought)**

**Description:** Explicit step-by-step reasoning to verify context sufficiency and identify conflicts.

**Business Logic:**

1. **For each query:**
   - Embed and retrieve (same as E3: top-N=20, rerank to top-K=5).

2. **CoT Reasoning via Structured LLM Output:**
   - System prompt instructs: "Break down problem → Check context relevance → Identify conflicts → Verify completeness → Provide reasoning → Final answer."
   - LLM outputs structured response (E4Response):
     - `reasoning_steps`: Step-by-step explanation (e.g., "Step 1: Query asks about return window. Step 2: Context has returns-policy.md which states 30 days. Step 3: No conflicts detected. Step 4: Sufficient info to answer.")
     - `answer`: Final answer (e.g., "Within 30 days of delivery according to returns-policy.md.").
   - For negative queries (unanswerable), reasoning should explain why KB lacks info, not hallucinate.

3. **Result Recording:**
   - Persist to `e4_reasoning_rag.jsonl`.
   - Include both reasoning_steps and answer.
   - Track retrieval, reranking, and reasoning times separately.

**Expected Improvement Over E2/E3:**
- Explicit reasoning surface prevents hallucinations via self-checking.
- Clear articulation of information gaps (e.g., "I don't know because...").
- Audit trail for failure case analysis.

**Acceptance Criteria:**
- All 150 queries processed with reasoning captured.
- Reasoning steps are coherent and specific (not generic).
- E4 correctly identifies unanswerable queries (no hallucinations on negative queries).

---

#### **FR7: Support Dry-Run Mode**

**Description:** Execute all three experiments without external API calls (mock data) for testing and debugging.

**Business Logic:**
- When DRY_RUN=true:
  - Skip KB indexing (mock vector store returns dummy chunks).
  - Skip embedding generation (use mock embeddings).
  - Skip LLM calls (return placeholder answers).
  - Process all 150 queries in < 1 minute.
  - Output valid JSONL files with mock data.
- Enable rapid testing of pipeline logic, error handling, output format without incurring API costs.

**Acceptance Criteria:**
- Dry-run completes in < 1 minute.
- Output files are structurally valid (parse as valid JSONL).
- All 150 queries processed.
- No external API calls made during dry-run.

---

#### **FR8: Support Resume and Crash Recovery**

**Description:** Enable resumption of interrupted experiment runs.

**Business Logic:**
- Before processing each query, check if result already exists in output file.
- If exists and OVERWRITE=false, skip query (log as skipped).
- If OVERWRITE=true, reprocess all queries.
- Flush results to disk after each query (fsync) to prevent data loss on crash.
- If interrupted, subsequent run resumes from last processed query.

**Acceptance Criteria:**
- Restarting E2 mid-run resumes from last query (not restart from query 1).
- OVERWRITE=true forces reprocessing of all queries.
- No data loss on simulated crash (kill process after few queries).

---

#### **FR9: Comprehensive Logging**

**Description:** Minimal console output; extensive file logging for debugging.

**Business Logic:**
- Configure logger to write to `logs/e2_to_e4.log` (file-only, no console handlers).
- Log levels:
  - INFO: Progress milestones (e.g., "Indexed 500 chunks", "Processing E2 query 45/150").
  - DEBUG: Per-query details (e.g., chunk IDs, scores, LLM tokens).
  - ERROR: Failures with context (query_id, error message, stack trace).
- Ensure logs are readable and actionable for researchers debugging failures.

**Acceptance Criteria:**
- Log file created at `logs/e2_to_e4.log`.
- No console prints (only log statements).
- Log entries include timestamps and severity levels.

---

#### **FR10: Output Result Schema**

**Description:** Define structure for E2, E3, E4 result files.

**Business Logic:**

All three experiments output JSONL (one JSON object per line) with schema:

```json
{
  "query_id": "q_direct_001",
  "experiment": "e2|e3|e4",
  "query": "What is the return window?",
  "retrieved_chunks": [
    {
      "chunk_id": "returns-policy.md#chunk_0",
      "text": "Within 30 days of delivery...",
      "score": 0.92,
      "metadata": { "filename": "returns-policy.md", "section": "Return Window" }
    }
  ],
  "llm_answer": "Within 30 days of delivery according to the returns policy.",
  "reasoning_steps": null,  // Only populated for E4
  "ground_truth": "30 days from delivery.",
  "context_reference": ["returns-policy.md"],
  "metadata": { "difficulty": "easy", "category": "returns_refunds" },
  "retrieval_time_ms": 85.5,
  "llm_time_ms": 2340.2,
  "total_time_ms": 2425.7,
  "model": "x-ai/grok-beta",
  "dry_run": false
}
```

**Rationale:**
- Enables phase 3 (evaluation): compute metrics per query.
- Retrieved chunks enable analysis of retrieval quality.
- Timing data enables performance profiling.
- Model/dry_run labels enable filtering and comparison.

**Acceptance Criteria:**
- All result files parse as valid JSONL.
- Every query in input produces one result in output.
- All required fields populated (no null except reasoning_steps for E2/E3).

---

### 3.3 Non-Functional Requirements

| Requirement    | Specification                                                       | Rationale                                                 |
| -------------- | ------------------------------------------------------------------- | --------------------------------------------------------- |
| **Performance** | E2-E4 process 150 queries in ≤30 minutes total wall time            | Acceptable iteration cycle for researchers                |
| **Reliability** | All 150 queries must be processed; resume on crash; ≥95% success    | Ensure complete dataset for evaluation; minimize re-runs   |
| **Usability**  | Single entry point: `python main.py`; config via `.env` and code    | No CLI args; reproducible configuration                   |
| **Logging**    | File-only logging to `logs/e2_to_e4.log`; minimal console output    | Avoid cluttering researcher terminals                     |
| **Portability** | Python 3.10+; no OS-specific dependencies                           | Work on Linux/Mac for research environments               |
| **Reproducibility** | In-memory vector store; fixed seed embeddings; logged configs       | Same input → same output across runs                      |
| **Extensibility** | Modular code; agents, prompts, chunking easily customizable          | Support future variant experiments (E5, E6, etc.)         |

---

## 4. Data Requirements

### 4.1 Input Schema: queries.jsonl

```json
{
  "query_id": "q_direct_001",
  "query_type": "direct|multi_hop|negative",
  "query": "What is the return window for most items?",
  "ground_truth": "Within 30 days of purchase or delivery.",
  "context_reference": ["returns-refunds-page-15.md"],
  "metadata": {
    "difficulty": "easy|medium|hard",
    "category": "returns_refunds"
  }
}
```

### 4.2 Input Schema: structure.json

```json
{
  "num_pages": 100,
  "pages": [
    {
      "id": "refund_policy_v1",
      "title": "Refund Policy v1",
      "filename": "refund_policy_v1.md",
      "category": "general_retail",
      "primary_topic": "returns_refunds",
      "secondary_topics": ["return_window", "methods"],
      "links_to": ["shipping.md", "contact.md"]
    }
  ]
}
```

### 4.3 Output Schema: e2/e3/e4_*.jsonl

See **FR10** (Result Schema) above.

---

## 5. Supporting Information

### 5.1 Architectural Rationale

| Aspect             | Design Choice                               | Rationale                                                                      |
| ------------------ | ------------------------------------------- | ------------------------------------------------------------------------------ |
| **Chunking**       | Headers-aware, 512 tokens, 128 overlap      | Preserves semantic boundaries; handles document variation                      |
| **Embedding**      | OpenAI text-embedding-3-small               | Industry standard; 1536 dims balance quality vs. cost                          |
| **Vector DB**      | Qdrant in-memory                            | Reproducible; no external service; fast iteration                              |
| **Reranker (E3)**  | cross-encoder/ms-marco-MiniLM-L-6-v2        | Lightweight, fast, proven on MS MARCO benchmark; <500ms per query              |
| **LLM**            | OpenRouter (x-ai/grok-beta or variant)      | Cost-effective; supports structured output (Pydantic); API abstraction         |
| **CoT (E4)**       | Explicit structured reasoning output        | Auditable; enables failure analysis; prevents silent hallucinations            |

### 5.2 Pipeline Flow Diagrams

#### E2 (Standard RAG)
```
Query → Embed → Vector Search (top-5) → Format Context → LLM → Answer
        50ms      50ms                   50ms            2s+     
```

#### E3 (Filtered RAG)
```
Query → Embed → Vector Search (top-20) → Rerank (top-5) → Format → LLM → Answer
        50ms      50ms                   100ms              50ms     2s+
```

#### E4 (Reasoning RAG)
```
Query → Embed → Vector Search (top-20) → Rerank (top-5) → Format → LLM (CoT) → Reasoning + Answer
        50ms      50ms                   100ms              50ms     2.5s+
```

### 5.3 Sample Execution Timeline (150 queries)

| Phase               | Duration      | Notes                                      |
| ------------------- | ------------- | ------------------------------------------ |
| Initialization      | ~10 seconds   | Load services, validate APIs               |
| KB Indexing         | ~8-10 minutes | 500-800 chunks, batch embed               |
| E2 Experiment       | ~4-6 minutes  | 150 queries × ~2.5s each                  |
| E3 Experiment       | ~5-7 minutes  | Reranking overhead ~1s per 5 queries       |
| E4 Experiment       | ~6-8 minutes  | CoT reasoning adds ~0.5s per query         |
| Result Validation   | < 1 minute    | Parse JSONL, check schema                  |
| **Total**           | **~25-35 min**| Varies by LLM latency, API throttling      |

---

## 6. Success Metrics & Verification

### 6.1 Functional Acceptance Criteria

- ✅ All 150 queries processed by E2, E3, E4 without errors.
- ✅ Output JSONL files are valid and complete (all required fields populated).
- ✅ Retrieved chunks include accurate metadata (filenames, sections).
- ✅ E4 outputs include non-empty reasoning_steps fields.
- ✅ Dry-run mode executes without API calls and completes in < 1 minute.
- ✅ Resume capability verified: interrupting and restarting yields same results.
- ✅ Log file captures all milestones and errors with context.

### 6.2 Quality Metrics (Post-Execution Analysis)

- **Completeness**: No null/empty llm_answer fields.
- **Reproducibility**: Two runs with same inputs produce identical results.
- **Performance**: Wall time for 150 queries ≤ 35 minutes.
- **Reliability**: ≥95% queries processed without transient failures.

### 6.3 Validation Steps

1. **Schema Validation**: Parse all output JSONL, validate against model.
2. **Completeness Check**: Ensure 150 results per experiment file.
3. **Spot Check**: Manually review 5 queries from each type (direct, multi_hop, negative) to assess answer quality.
4. **Performance Profiling**: Aggregate timing metrics to identify bottlenecks.
5. **Dry-Run Verification**: Execute dry-run, confirm no API calls and < 1 min completion.

---

## 7. Non-Functional Design Patterns

### 7.1 Error Handling

- **Transient Failures** (API rate limits, timeouts): Retry with exponential backoff (3 attempts, 1s → 2s → 4s).
- **Permanent Failures** (malformed query, missing file): Log and skip; continue processing remaining queries.
- **Critical Failures** (API key invalid, vector DB failure): Halt with clear error message.

### 7.2 Resource Management

- **Memory**: In-memory vector store is bounded by KB size (~50MB for 500-800 embeddings).
- **CPU**: Embedding batching and reranking leverage multi-core; expect 4-8 core utilization.
- **API Costs**: E2-E4 incur costs for LLM calls (~150 calls × ~0.01-0.1 USD per call depending on model); embeddings (~500 chunks × ~0.00001 USD per embedding).

### 7.3 Maintainability

- **Modular Design**: Agents, prompts, chunking, vector store decoupled.
- **Configuration**: All constants hardcoded in `constants.py`; easily customizable.
- **Testing**: Dry-run mode enables unit testing without costs; structure supports mocking.

---

## 8. Appendix: Glossary

| Term                | Definition                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------------- |
| **RAG**             | Retrieval-Augmented Generation: retrieve external docs before generating response.            |
| **Hallucination**   | LLM generates plausible but unsupported or false content.                                    |
| **Chunk**           | Segment of a document (e.g., paragraph or section), typically ≤512 tokens.                   |
| **Embedding**       | Vector representation of text (e.g., 1536-dim for OpenAI text-embedding-3-small).            |
| **Reranking**       | Re-ordering retrieved documents by a more accurate relevance model (cross-encoder).          |
| **CoT**             | Chain-of-Thought: explicit step-by-step reasoning before final answer.                       |
| **Cross-Encoder**   | Model that scores a (query, document) pair for relevance (vs. separate query/doc embeddings). |
| **Vector DB**       | Database optimized for storing and searching high-dimensional vectors (e.g., Qdrant).        |
| **Cosine Distance** | Similarity metric between vectors: 1 - (A·B)/(∥A∥∥B∥); ranges [0, 1] (1=identical).         |
| **Pydantic**        | Python library for data validation and structured serialization via type hints.              |
| **OpenRouter**      | API proxy providing unified access to multiple LLMs (GPT, Claude, etc.).                     |

---

**End of SRS**
