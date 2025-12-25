# SRS: Phase 2.1-2.4 — E1-E4 RAG Pipeline Experiments

**Version:** 3.0 | **Date:** December 25, 2025

## Purpose

Evaluate four progressively sophisticated RAG pipeline architectures to systematically reduce LLM hallucinations in customer service chatbots. This research establishes evidence-based guidelines for minimizing hallucinations through data structuring, retrieval filtering, and reasoning strategies.

## Scope

**Delivers:**
- E1 (Baseline): Direct LLM responses without retrieval
- E2 (Standard RAG): Vector search + LLM generation
- E3 (Filtered RAG): Vector search + cross-encoder reranking + LLM
- E4 (Reasoning RAG): Chain-of-Thought prompting + vector search + reranking + LLM
- Output: `output/kb/data/e1_baseline.jsonl`, `e2_standard_rag.jsonl`, `e3_filtered_rag.jsonl`, `e4_reasoning_rag.jsonl`

**Input:**
- `output/kb/*.md` (100 structured Markdown pages with tables, Mermaid diagrams, versioned content)
- `output/kb/data/queries.jsonl` (200 adversarial queries: direct, multi-hop, negative, table-aggregation)

## Key Concepts

**Hallucination Mitigation Strategy**: Progressive architectural improvements addressing retrieval noise, data rot, and reasoning failures.

**Structured Knowledge Base**: Markdown format preserves tabular data and conditional logic, superior to plain text for customer service RAG (Wang et al., 2024).

**Adversarial Dataset**: Includes semantic drift (versioned conflicts), hub pages, circular references, and lexical traps to expose RAG weaknesses.

**Chain-of-Thought Reasoning**: Forces systematic analysis of context, conflict detection, and completeness verification.

## Requirements

### Common Setup (All E1-E4)

**1. Configuration Management**
- Environment variables: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `KB_DIR`, `OVERWRITE`, `DRY_RUN`
- Derived paths: `output/kb/data/` for inputs/outputs
- Logging: File-only logging to `logs/e1_to_e4.log`

**2. Query Processing**
- Load 200 queries from `output/kb/data/queries.jsonl`
- Support resume functionality via `OVERWRITE` flag
- Rate limiting: 3-second delays between OpenRouter API calls (20 req/min limit)

**3. Knowledge Base Indexing (E2-E4)**
- Load 100 Markdown files with structure-aware chunking (512 tokens, 128 overlap)
- Generate embeddings using OpenAI `text-embedding-3-small` (1536 dimensions)
- Store in Qdrant vector database (in-memory mode)
- Metadata per chunk: filename, section headers, version info

### E1: Baseline (No RAG)

**Process per query:**
1. Load query from JSONL file
2. Send direct prompt to LLM: "You are a retail customer support assistant. Answer the question concisely. If you are unsure or don't have information to answer accurately, say 'I don't know'."
3. Record answer, ground truth, metadata

**Expected Outcome**: High hallucination rate establishes "knowledge gap" baseline

**Output**: `output/kb/data/e1_baseline.jsonl`

### E2: Standard RAG

**Process per query:**
1. Embed query using OpenAI `text-embedding-3-small`
2. Vector search → retrieve top-5 chunks (cosine similarity)
3. Format context: "[Source 1: filename]\nchunk_text\n\n[Source 2: filename]\nchunk_text..."
4. Construct prompt: "Context:\n{context}\n\nQuestion: {query}"
5. LLM generates answer with source citations
6. Record: query, retrieved chunks, answer, ground truth, timing metrics

**Expected Improvement**: Retrieval provides relevant context, reducing hallucinations

**Output**: `output/kb/data/e2_standard_rag.jsonl`

### E3: Filtered RAG

**Process per query:**
1. Embed query using OpenAI `text-embedding-3-small`
2. Vector search → retrieve top-20 chunks (high recall)
3. **Cross-encoder reranking** using `cross-encoder/ms-marco-MiniLM-L-6-v2` → select top-5 most relevant
4. Format context from reranked chunks
5. Construct prompt with filtered context
6. LLM generates answer with source citations
7. Record: query, reranked chunks (with scores), answer, ground truth, timing metrics

**Expected Improvement**: Reranking removes irrelevant "noise" chunks, increasing precision

**Output**: `output/kb/data/e3_filtered_rag.jsonl`

### E4: Reasoning RAG

**Process per query:**
1. **CoT Planning Prompt**: "Break down the problem: First, identify what information is needed to answer the question..."
2. Embed query and perform vector search → retrieve top-20 chunks
3. Cross-encoder reranking → select top-5 chunks
4. **Chain-of-Thought Generation Prompt**:
   - "Analyze the context: Check if the context contains relevant, up-to-date information"
   - "Check for conflicts: If multiple sources provide contradictory information, note the discrepancy"
   - "Verify completeness: Ensure all parts of the question can be answered from the context"
   - "Cite sources: Reference specific sources for each claim"
   - "Handle missing data: If critical information is missing, outdated, or conflicting, explicitly state this"
   - "Provide reasoning: Explain your thought process step-by-step in the reasoning_steps field"
   - "Final answer: Provide a clear, concise answer in the answer field"
5. LLM generates structured response with `reasoning_steps` and `answer` fields
6. Record: query, chunks, reasoning steps, answer, ground truth, timing metrics

**Expected Improvement**: Explicit reasoning prevents hallucinations via self-checking and conflict detection

**Output**: `output/kb/data/e4_reasoning_rag.jsonl`

## Data Schema

### e1_baseline.jsonl

```json
{
  "query_id": "q_direct_001",
  "experiment": "E1",
  "query": "What is the return window?",
  "llm_answer": "Returns are accepted within 30 days of delivery.",
  "ground_truth": "14 days from delivery.",
  "retrieval_time_ms": 0.0,
  "llm_time_ms": 850.5,
  "total_time_ms": 850.5,
  "model": "amazon/nova-2-lite-v1:free",
  "dry_run": false
}
```

### e2_standard_rag.jsonl, e3_filtered_rag.jsonl

```json
{
  "query_id": "q_direct_001",
  "experiment": "E2",
  "query": "What is the return window?",
  "retrieved_chunks": [
    {
      "chunk_id": "returns-policy-v2.md_chunk_0",
      "text": "Returns accepted within 14 days...",
      "score": 0.89,
      "rerank_score": 0.0,
      "metadata": {"filename": "returns-policy-v2.md"}
    }
  ],
  "llm_answer": "The return window is 14 days from delivery.",
  "ground_truth": "14 days from delivery.",
  "retrieval_time_ms": 45.2,
  "llm_time_ms": 1200.8,
  "total_time_ms": 1246.0
}
```

### e4_reasoning_rag.jsonl

```json
{
  "query_id": "q_direct_001",
  "experiment": "E4",
  "query": "What is the return window?",
  "retrieved_chunks": [
    {
      "chunk_id": "returns-policy-v2.md_chunk_0",
      "text": "Returns accepted within 14 days...",
      "score": 0.89,
      "rerank_score": 0.95,
      "metadata": {"filename": "returns-policy-v2.md"}
    }
  ],
  "llm_answer": "The return window is 14 days from delivery.",
  "reasoning_steps": "Step 1: The question asks for the return window. Step 2: I found relevant information in returns-policy-v2.md stating '14 days from delivery'. Step 3: This appears to be current information with no conflicts. Step 4: The answer is complete and directly addresses the question.",
  "ground_truth": "14 days from delivery.",
  "retrieval_time_ms": 67.3,
  "llm_time_ms": 1850.2,
  "total_time_ms": 1917.5
}
```

**Key Fields:**
- `experiment`: "E1", "E2", "E3", or "E4"
- `retrieved_chunks`: Array of chunk objects (empty for E1)
- `score`: Vector similarity score
- `rerank_score`: Cross-encoder relevance score (E3/E4 only)
- `reasoning_steps`: CoT reasoning process (E4 only)
- Timing metrics for performance analysis

## Technical Stack

- **Vector Database**: Qdrant (in-memory mode for consistency)
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (BERT-based cross-encoder)
- **LLM**: Via OpenRouter gateway (configurable models)
- **Framework**: Pydantic-AI for structured outputs and agent management
- **Chunking**: Fixed window (512 tokens) with 128-token overlap
- **Programming Language**: Python 3.11+

## Quality Assurance

### Dry Run Mode
- `DRY_RUN=true` generates mock responses without API calls
- Validates pipeline structure and data flow
- Enables testing on machines without API keys

### Error Handling
- Network failures: Exponential backoff retry
- Rate limiting: 3-second delays between requests
- Invalid responses: Structured error logging
- Resume capability: Skip already processed queries

### Validation Checks
- API key presence (unless dry run)
- Input file existence and format
- Output directory creation
- Chunk embedding completeness
- Response schema validation

## Performance Characteristics

### Expected Metrics Progression

| Experiment | Retrieval Time | LLM Time | Total Time | Expected Accuracy |
|------------|----------------|----------|------------|-------------------|
| E1 | 0ms | ~800ms | ~800ms | Low (high hallucinations) |
| E2 | ~50ms | ~1200ms | ~1250ms | Medium |
| E3 | ~70ms | ~1400ms | ~1470ms | High |
| E4 | ~70ms | ~1800ms | ~1870ms | Highest |

### Cost Considerations
- E1: Minimal (direct LLM calls)
- E2-E4: Embedding costs + retrieval + LLM generation
- Rate limiting ensures compliance with API quotas
- Caching reduces redundant embedding computations

## Dependencies

### Python Packages
```
pydantic-ai-slim[openrouter]>=0.1.0
pydantic>=2.0.0
tqdm>=4.65.0
python-dotenv>=1.0.0
ruff>=0.1.0
qdrant-client>=1.7.0
openai>=1.0.0
sentence-transformers>=2.2.0
```

### External Services
- OpenRouter API (LLM access)
- OpenAI API (embeddings)

## Testing Strategy

### Unit Tests
- Model validation and serialization
- Embedding service functionality
- Vector store operations
- Agent creation and response parsing

### Integration Tests
- End-to-end pipeline execution (dry run mode)
- Query processing workflow
- Output format validation
- Resume functionality

### Performance Tests
- Batch processing efficiency
- Memory usage with large knowledge bases
- API rate limit compliance

## Deployment and Execution

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with API keys
```

### Execution Commands
```bash
# Dry run (testing)
python dry_run.py

# Full execution
python main.py

# Individual experiments
python -c "from e1_to_e4.pipeline.e1_baseline import run_e1_baseline; run_e1_baseline(...)"
```

### Output Structure
```
output/kb/
├── data/
│   ├── queries.jsonl
│   ├── e1_baseline.jsonl
│   ├── e2_standard_rag.jsonl
│   ├── e3_filtered_rag.jsonl
│   └── e4_reasoning_rag.jsonl
└── embeddings_cache/
    ├── chunks.jsonl
    └── queries.jsonl
```

## Maintenance and Evolution

### Version Control
- Semantic versioning for SRS updates
- Backward compatibility for data formats
- Migration scripts for breaking changes

### Monitoring and Logging
- Comprehensive file logging with timestamps
- Performance metrics collection
- Error categorization and reporting

### Future Extensions
- Additional reranking models
- Alternative embedding providers
- Multi-modal content support
- Real-time evaluation metrics

---

**End of SRS**
