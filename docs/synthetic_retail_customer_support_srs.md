# SRS: Phase 1.1-1.3 — Synthetic Retail KB Generator

**Version:** 2.0 | **Date:** December 7, 2025

## Purpose

Generate 100 interlinked Markdown files simulating retail customer support documentation for RAG pipeline evaluation.

## Scope

**Delivers:**
- 100 MD files with headers, tables, Mermaid diagrams, hyperlinks
- 5 versioned page pairs (v1 outdated, v2 current) with typed conflict metadata for 10% data rot
- 15% pages with injected mistakes
- Output: `output/kb/*.md` + `output/kb/data/structure.json`

## Key Concepts

**Data Rot with Semantic Drift (10%, ADVERSARIAL HARDENING)**: 5 versioned pairs where v1 contradicts v2 with SUBTLE, SEMANTIC conflicts (not just number swaps). LLM receives v1 content when generating v2 with instructions to:
- Keep 70% of text identical (lexical overlap) to avoid obvious detection
- Change thresholds/conditions in 30% of content:
  - **Conditional Threshold**: e.g., "free shipping $50" → "free shipping $75 AND excludes Alaska/Hawaii"
  - **Scope Narrowing**: e.g., "returns 30 days all items" → "returns 30 days online only"
  - **Eligibility Tightening**: e.g., "pristine condition" → "pristine condition AND original tags AND original packaging"
  - **Exception Addition**: e.g., "free returns" → "free returns EXCEPT clearance items"
  - **Definition Shift**: e.g., "refund in business days" → "refund in calendar days"
- Avoid explicit date markers ("Updated [DATE]") to maximize confusion when both versions retrieved

**Mistake Types**: Inconsistency (40%), omission (28%), poor UX (20%), outdated info (8%), accessibility (4%)

**Page Types with Adversarial Structure**: Tabular (40%), logical (30%), unstructured (30%)
- **Hub Pages (20%)**: Overview pages with NO specific data; use placeholder language ("See [Page X] for rates")
- **Detail Pages (20%)**: Contain specific facts/tables that hub pages reference
- **Regular Pages (60%)**: Standard content pages

## Requirements

### Core Functions

**1. Structure Planning**
- Load/generate `structure.json` in `output/kb/data/` defining 100 pages
- Apply topic distribution (see below)
- Designate 5 base pages for versioned rot pairs

**2. Content Generation with Adversarial Hardening**
- **Standard pages**: Generate from topic/category/style specs with 2000-token budget
- **Mandatory content requirements**:
  - **Tabular pages**: 2+ markdown tables with CONDITIONAL LOGIC and CROSS-TABLE DEPENDENCIES
    - Include at least ONE conditional column (e.g., "Shipping Cost depends on Weight > 5lb")
    - Add footnotes and nested logic within cells
    - 30% of table data should require multi-row aggregation to answer queries
  - **Logical pages**: 1+ Mermaid flowchart (8+ nodes, decision logic)
  - **Hub pages**: CRITICAL: Must NOT contain specific prices, dates, timeframes, SKUs. Use only placeholders like "See [Detail Page] for specific rates"
- **Versioned rot pages with Semantic Drift**:
  - Generate v1 first (full detail)
  - Pass v1 content when generating v2 with semantic drift instructions:
    - Keep 70% identical to create lexical overlap trap
    - Modify 30% via subtle conditional/scope/eligibility changes
    - NO date markers like "Updated [DATE]"
- Inject mistakes (15% probability per page)
- Generate cross-page links per structure
- Create circular reference loops (A ↔ B) for 2-3 page pairs
- Designate entity anchors (e.g., "Platinum Membership", "Zone B") appearing across non-linked pages

**3. Output**
- MD files: `output/kb/*.md` (e.g., `returns-policy-v1.md`, `returns-policy-v2.md`)
- Metadata: `output/kb/data/structure.json`

**4. Validation**
- Check Markdown syntax, links, Mermaid validity
- Verify 5 rot pairs (10 versioned files)

## Data Schema

### structure.json

```json
{
  "num_pages": 100,
  "page_types": { "tabular": 40, "logical": 30, "unstructured": 30 },
  "rot_pairs": [
    {
      "v1": "returns-policy-v1",
      "v2": "returns-policy-v2",
      "conflict_type": "timeframe_change",
      "conflict_description": "Return window reduced from 30 days to 14 days"
    }
  ],
  "pages": [
    {
      "id": "returns-policy-v2",
      "title": "Returns Policy (Current)",
      "filename": "returns-policy-v2.md",
      "type": "logical",
      "primary_topic": "returns_refunds",
      "links_to": ["returns-policy-v1.md"],
      "requires_table": true,
      "requires_mermaid": true
    }
  ]
}
```

## Topic Distribution

**15 Topics** (rebalanced for 2-11% each):
- High-frequency (6-11%): orders, returns_refunds, shipping_delivery, contact, faq
- Medium (4-9%): account, payments_billing, membership_loyalty, product_info, warranty, store_services
- Low (2-3%): accessibility, installation, sustainability, recycling

---

**End of SRS**
