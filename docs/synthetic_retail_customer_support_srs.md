# SRS: Phase 1.1-1.3 — Synthetic Retail KB Generator

**Version:** 2.0 | **Date:** December 7, 2025

## Purpose

Generate 100 interlinked Markdown files simulating retail customer support documentation for RAG pipeline evaluation.

## Scope

**Delivers:**
- 100 MD files with headers, tables, Mermaid diagrams, hyperlinks
- 5 versioned page pairs (v1 outdated, v2 current) for 10% data rot
- 30% pages with injected mistakes
- Output: `output/kb/*.md` + `output/kb/data/structure.json`

## Key Concepts

**Data Rot (10%)**: 5 versioned pairs where v1 contradicts v2. LLM receives v1 content when generating v2 to create realistic conflicts (e.g., `policy-v1.md`: 30-day returns vs `policy-v2.md`: 14-day returns).

**Mistake Types**: Inconsistency (40%), omission (28%), poor UX (20%), outdated info (8%), accessibility (4%)

**Page Types**: Tabular (40%), logical (30%), unstructured (30%)

## Requirements

### Core Functions

**1. Structure Planning**
- Load/generate `structure.json` in `output/kb/data/` defining 100 pages
- Apply topic distribution (see below)
- Designate 5 base pages for versioned rot pairs

**2. Content Generation**
- **Standard pages**: Generate from topic/category/style specs
- **Versioned rot pages**: 
  - Generate v1 first
  - Pass v1 content when generating v2 with instruction: "Create contradicting content (dates, policies, prices)"
- Include tables, Mermaid diagrams, cross-links
- Inject mistakes (30% probability)

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
      "conflict": "Versioned conflict: policy changed"
    }
  ],
  "pages": [
    {
      "id": "returns-policy-v2",
      "title": "Returns Policy v2",
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

**High-frequency (70%)**: Orders (14), Returns (13.5), Shipping (13), Contact (12.5), FAQ (17)  
**Medium (22%)**: Account (6.5), Payments (5.5), Membership (4.5), Product (3.5), Warranty (2)  
**Low (8%)**: Store services, Accessibility, Installation, Sustainability, Recycling

---

**End of SRS**
