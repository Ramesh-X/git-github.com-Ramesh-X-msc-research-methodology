# Proposed Method Summary: Mitigating LLM Hallucinations in Customer Service Chatbots

This document provides a detailed, step-by-step guide to the methodology for the MSc research on formulating architectural guidelines to minimize hallucinations in domain-specific customer service chatbots using Retrieval-Augmented Generation (RAG) pipelines. It synthesizes content from the `6 - Proposed Method/` folder ([6.1 Introduction.md](6 - Proposed Method/6.1 Introduction.md), [6.2 Problem Decomposition and Design Strategy.md](6 - Proposed Method/6.2 Problem Decomposition and Design Strategy.md), etc.) and aligns with the overall research plan in [My Research - My words.md](../My Research - My words.md).

The methodology decomposes the hallucination problem into retrieval noise/misses, data rot, and reasoning failures ([6.2]), addressed via a funnel architecture: structured data → high-recall retrieval → precision reranking → Chain-of-Thought (CoT) reasoning. It compares four experimental conditions (E1-E4) on a synthetic Retail Support dataset.

**Key Assumptions**: Structured Markdown KB (with tables/Mermaid diagrams) outperforms plain text ([6.3], [17] in references).

## Step-by-Step Research Execution Guide

### Phase 1: Generate Structured Knowledge Base (KB) Dataset
Reference: [6.3 Phase I - Structured Knowledge Base Generation.md](6 - Proposed Method/6.3 Phase I - Structured Knowledge Base Generation.md), [6.2].

1. **Create Master Dataset Plan**:
   - Analyze real retail support sites (e.g., shipping, refunds, troubleshooting).
   - Define `structure.json`: Plan for 100 interlinked Markdown pages categorized as tabular (e.g., pricing tiers), logical/conditional (e.g., refund policies), unstructured/mixed.
   - Include 10% data rot simulation: 5 versioned page pairs with conflicting content (v1 outdated, v2 current) — e.g., `Policy-v1.md`: 30-day refunds vs. `Policy-v2.md`: 14-day refunds.
   - Specify interlinks, tables, Mermaid diagrams per page.

2. **Generate Markdown Pages**:
   - Use LLM to produce 100 detailed MD files from `structure.json`.
   - Ensure: Headers, bold/italics, tables, Mermaid diagrams, hyperlinks.

3. **Validate Dataset**:
   - Python script to check: Markdown validity, link integrity, Mermaid syntax, ~10% rot presence.

4. **Generate Evaluation Queries and Ground Truth**:
   - Direct queries (120): Generated only from current (v2) pages using weighted random selection for balanced distribution.
   - Multi-hop queries (40): Require information from 2+ linked pages.
   - Negative queries (40): Plausible but unanswerable (expect "I don't know").
   - Store in `data/queries.jsonl`: `{query, ground_truth, context_reference}`.
   - Total: 200 QA pairs.

**Output**: 100 MD files (root) + `data/structure.json` + `data/queries.jsonl` + validated topology (Figure 6.2 Mermaid in [6.3]).

### Phase 2: Implement and Run Experimental Pipelines
Reference: [6.4 Phase II - Architectural Pipeline Variations.md](6 - Proposed Method/6.4 Phase II - Architectural Pipeline Variations.md), [6.5 Implementation and Technology Stack.md](6 - Proposed Method/6.5 Implementation and Technology Stack.md), [6.1], [6.6 Feasibility and Justification.md](6 - Proposed Method/6.6 Feasibility and Justification.md).

**Common Setup (All E1-E4)**:
1. **KB Ingestion**:
   - Structure-aware chunking: Respect headers/tables (fixed window + overlap).
   - Embed chunks with `text-embedding-3-small`.
   - Store in Qdrant (HNSW index; payload: text + metadata like version/date).

2. **Technology Constants**:
   - Vector DB: Qdrant.
   - Embeddings: OpenAI `text-embedding-3-small`.
   - LLM: LLMs available via OpenRouter gateway.
   - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - Framework: Pydantic-AI for modular agents/prompts.

**Run Experiments Sequentially**:

3. **E1: Baseline (No RAG)**:
   - Input: User query → LLM prompt (includes refusal instruction: "say 'I don't know' if unsure") → Generate answer.

4. **E2: Standard RAG**:
   - Query → Vector search (cosine, top-k=5) → Concat chunks → LLM.

5. **E3: Filtered RAG**:
   - Query → Vector search (top-N=20) → Cross-encoder rerank (top-k=5) → LLM.

6. **E4: Reasoning RAG**:
   - CoT prompt: "Plan retrieval steps (e.g., check policy date)" → Vector (top-N=20) → Rerank (top-k=5) → LLM with CoT reasoning.
   - Prompts enforce: "Say 'don't know' if unsure; cite sources."

**Execution**:
- For each of 200 queries, run all E1-E4 pipelines.
- Log: Raw outputs, retrieved chunks, tokens/latency, traces.

**Output**: JSON logs in `data/` folder (`e1_baseline.jsonl`, `e2_standard_rag.jsonl`, etc.) with query, retrieved_chunks, answer, metadata.

### Phase 3: Evaluate and Analyze Results
Reference: [6.7 Evaluation Framework.md](6 - Proposed Method/6.7 Evaluation Framework.md), [My Research - My words.md].

1. **Quantitative Metrics (Automated with RAGAS)**:
   - Compute per query: Context Precision (CP), Faithfulness (F), Answer Relevancy (AR) using GPT-4 as judge.
   - Composites:
     - Geometric Mean: \( G_{mean} = (CP \cdot F \cdot AR)^{1/3} \)
     - Hallucination Risk Index (HRI): \( (1 - F) \cdot AR \)
   - Categorize outputs (Table in [6.7]):
     | Category | Conditions |
     |----------|------------|
     | Clean Pass | CP>0.7 ∧ F>0.7 ∧ AR>0.7 |
     | Hallucination | CP>0.6 ∧ F<0.4 |
     | Retrieval Failure | CP<0.4 ∧ F>0.7 |
     | Irrelevant | AR<0.4 ∧ F>0.7 |
     | Total Failure | CP<0.4 ∧ F<0.4 ∧ AR<0.4 |
   - Aggregates: Means, % per category, HRI (mean + 95th percentile).

2. **Qualitative Analysis**:
   - Manual review of top-10 failures per E: Categorize as Retrieval Miss, Logic Failure, Outdated Info ([6.2], [6.7]).

3. **Cost/Latency Trade-offs**:
   - Measure: Tokens/query, Time-To-First-Token (TTFT), total cost/latency per E.

4. **Statistical Tests**:
   - Pairwise t-tests (p<0.05) on G_mean/HRI between E1-E4.

**Output**: Scorecards (tables: Mean(CP/F/AR), % categories, HRI percentiles), charts (bar/line for metrics progression E1→E4), p-values.

### Phase 4: Visualize and Report Results
Reference: [6.9 Summary.md](6 - Proposed Method/6.9 Summary.md), [My Research - My words.md].

1. Create tables/charts:
   - Pipeline comparison (Table 6.9.1 style).
   - Metric progression (E1-E4 funnel benefits).
   - Error taxonomy pie charts.
   - Latency/cost vs. accuracy scatter.

2. Discuss: Incremental gains, limitations ([6.8 Methodological Limitations.md](6 - Proposed Method/6.8 Methodological Limitations.md): synthetic data, no UI, fixed chunking).

**Replication Notes**:
- All steps are modular/offline-first.
- Prompts: Include "cite sources, say 'don't know'" ([My Research]).
- Scale: Prototype; extend to real data/UI as future work.
- References: See [My Research - My words.md] for full biblio ([16] DynamicRAG, [17] Chain-of-Table).

This guide enables full replication: dataset → experiments → eval → viz without contradictions to sources.
