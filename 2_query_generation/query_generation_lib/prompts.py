BASE_INSTRUCTIONS = (
    "You are an assistant generating evaluation queries for a retail customer support knowledge base."
    " Produce a single question, a concise ground_truth answer if available, and give a category."
)

# DIRECT QUERY SUBTYPES
DIRECT_SIMPLE_FACT_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: SIMPLE FACT EXTRACTION"
    + "\n\nContext (Source page content):\n\n{content}"
    + "\n\nNotes: Generate a question asking for a SPECIFIC FACT (price, date, contact info, percentage, etc.) that appears explicitly in the text."
    + " The answer should be concise, directly extractable, and unambiguous."
)

DIRECT_TABLE_LOOKUP_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: TABLE LOOKUP"
    + "\n\nContext (Source page content):\n\n{content}"
    + "\n\nNotes: Generate a question that REQUIRES looking up specific data in a table or structured data section."
    + " The answer should be a specific cell value, row comparison, or table lookup result."
)

DIRECT_TABLE_AGGREGATION_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: TABLE AGGREGATION (ADVERSARIAL)"
    + "\n\nContext (Source page content):\n\n{content}"
    + "\n\nNotes: Generate a question that REQUIRES MULTI-ROW AGGREGATION or CONDITIONAL LOGIC across table data:"
    + "\n  - Multi-row aggregation: 'Which shipping method is cheapest for a 7lb package to Zone B?' (requires calculation)"
    + "\n  - Conditional logic: 'What is the total cost including discount for Platinum members?' (requires lookup + calculation)"
    + "\n  - Cross-table dependencies: Questions that require data from two different tables"
    + "\nThe answer CANNOT be found by looking at a single cell; it requires analyzing multiple rows/columns."
    + " This tests if the RAG system can perform arithmetic, conditional reasoning, or multi-step lookups."
)

DIRECT_PROCESS_STEP_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: PROCESS STEP IDENTIFICATION"
    + "\n\nContext (Source page content):\n\n{content}"
    + "\n\nNotes: Generate a question asking about a SPECIFIC STEP or requirement in a process, procedure, or workflow."
    + " Reference the process name and ask about one specific aspect or step number."
)

DIRECT_CONDITIONAL_LOGIC_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: CONDITIONAL LOGIC"
    + "\n\nContext (Source page content):\n\n{content}"
    + "\n\nNotes: Generate a CONDITIONAL question ('What if...', 'What happens when...', 'Does...')."
    + " The answer requires following a decision path or understanding an if-then scenario in the content."
)

DIRECT_LIST_ENUMERATION_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: LIST/ENUMERATION"
    + "\n\nContext (Source page content):\n\n{content}"
    + "\n\nNotes: Generate a question asking for a COMPLETE LIST or enumeration ('What are all...', 'List the...', 'Which...are...')."
    + " The answer should enumerate all items, steps, options, or requirements."
)

DIRECT_ROT_AWARE_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: ROT-AWARE (Current Version)"
    + "\n\nContext (Source page content):\n\n{content}"
    + "\n\nNotes: Generate a question asking for CURRENT information on a policy, price, or process that has changed."
    + " Include temporal keywords ('current', 'latest', 'now', 'as of 2024', 'today')."
    + " The question should target the CURRENT (v2) version to test if systems retrieve the right version."
)

# MULTI-HOP QUERY SUBTYPES
MULTI_HOP_SEQUENTIAL_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: SEQUENTIAL PROCESS MULTI-HOP"
    + "\n\nContext (Both pages):\n\n{content_a}\n\n---\n\n{content_b}"
    + "\n\nNotes: Generate a question requiring information from BOTH pages to describe a COMPLETE PROCESS or workflow."
    + " User starts on page A and NEEDS page B to complete the task."
    + "\n\nCRITICAL: The question MUST be unanswerable using only ONE of the provided pages."
)

MULTI_HOP_POLICY_FAQ_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: POLICY + FAQ CROSS-REFERENCE"
    + "\n\nContext (Both pages):\n\n{content_a}\n\n---\n\n{content_b}"
    + "\n\nNotes: Generate a question where the policy page provides the BASE RULE, but the FAQ page contains an EXCEPTION or CLARIFICATION needed to fully answer."
    + " Example: Policy says '30-day returns', FAQ clarifies 'except for clearance items'."
    + "\n\nCRITICAL: The question MUST be unanswerable using only ONE of the provided pages."
)

MULTI_HOP_COMPARATIVE_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: COMPARATIVE MULTI-HOP"
    + "\n\nContext (Both pages):\n\n{content_a}\n\n---\n\n{content_b}"
    + "\n\nNotes: Generate a COMPARISON question requiring data from BOTH pages (comparing values, rates, features, tiers, etc.)."
    + " User must look at both sources to make the comparison."
    + "\n\nCRITICAL: The question MUST be unanswerable using only ONE of the provided pages."
)

MULTI_HOP_HUB_TO_DETAIL_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: HUB-TO-DETAIL NAVIGATION"
    + "\n\nContext (Both pages):\n\n{content_a}\n\n---\n\n{content_b}"
    + "\n\nNotes: Generate a question where the HUB/overview page directs the user to a specific detail page, and BOTH are needed:"
    + " hub page for navigation context, detail page for the actual answer."
    + "\n\nCRITICAL: The question MUST be unanswerable using only ONE of the provided pages."
)

MULTI_HOP_CROSS_CATEGORY_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: CROSS-CATEGORY MULTI-HOP"
    + "\n\nContext (Both pages from different categories):\n\n{content_a}\n\n---\n\n{content_b}"
    + "\n\nNotes: Generate a question CROSSING DIFFERENT categories (e.g., 'If I return an order, how does it affect my loyalty points?')."
    + " Requires information from 2-3 different category pages."
    + "\n\nCRITICAL: The question MUST be unanswerable using only ONE of the provided pages."
)

MULTI_HOP_ROT_AWARE_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: ROT-AWARE MULTI-HOP"
    + "\n\nContext (Both pages, one is versioned):\n\n{content_a}\n\n---\n\n{content_b}"
    + "\n\nNotes: Generate a question about CURRENT policy that requires distinguishing between OLD and NEW versions."
    + " Include temporal cues ('What is the current...', 'As of now...')."
    + " Test if system retrieves the correct version and combines it with related information."
    + "\n\nCRITICAL: The question MUST be unanswerable using only ONE of the provided pages."
)

# NEGATIVE QUERY SUBTYPES
ANCHORED_ADJACENT_TOPIC_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: ADJACENT TOPIC NEGATIVE"
    + "\n\nAnchor page (content):\n\n{anchor_content}\n\n"
    + "Linked pages (content):\n\n{linked_contents}\n\n"
    + "Anchor metadata:\n\n{anchor_meta}\n\n"
    + "Full KB Topic Summary:\n\n{kb_summary}\n\n"
    + "LEXICAL TRAP INSTRUCTION: Generate a question that:"
    + "\n  1. Uses 15+ keywords/phrases from the anchor page itself"
    + "\n  2. Asks about a topic SEMANTICALLY CLOSE but NOT COVERED in the KB"
    + "\n  3. Should be PLAUSIBLE given the anchor context"
    + "\n  4. Will have HIGH cosine similarity (>0.70) with the anchor page"
    + "\n  5. But NO page provides the specific answer requested"
    + "\nExample: If anchor is 'Payment Methods', ask 'What is the refund timeline for Apple Pay purchases?' (Apple Pay is listed, but refund specifics are not)."
    + "\nground_truth: 'I don't know based on the KB.'"
)

ANCHORED_MISSING_DATA_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: MISSING DATA POINT NEGATIVE"
    + "\n\nAnchor page (content):\n\n{anchor_content}\n\n"
    + "Linked pages (content):\n\n{linked_contents}\n\n"
    + "Anchor metadata:\n\n{anchor_meta}\n\n"
    + "Full KB Topic Summary:\n\n{kb_summary}\n\n"
    + "LEXICAL TRAP INSTRUCTION: Generate a question that:"
    + "\n  1. Asks for a SPECIFIC DATA POINT (date, duration, model, fee, percentage, etc.)"
    + "\n  2. Uses keywords from the anchor page (80%+ overlap)"
    + "\n  3. Targets a granular detail NOT explicitly documented (e.g., 'How long do refunds take for PayPal?' when page says 'refunds 5-7 days' but doesn't specify PayPal)"
    + "\n  4. KB covers the general topic but not this specific variation"
    + "\nground_truth: 'I don't know based on the KB.'"
)

ANCHORED_OUT_OF_SCOPE_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: OUT-OF-SCOPE PROCEDURE NEGATIVE"
    + "\n\nAnchor page (content):\n\n{anchor_content}\n\n"
    + "Linked pages (content):\n\n{linked_contents}\n\n"
    + "Anchor metadata:\n\n{anchor_meta}\n\n"
    + "Full KB Topic Summary:\n\n{kb_summary}\n\n"
    + "LEXICAL TRAP INSTRUCTION: Generate a question that:"
    + "\n  1. Describes a RELATED PROCEDURE or EDGE CASE NOT documented"
    + "\n  2. Reuses 70%+ of vocabulary from the anchor page"
    + "\n  3. Feels like it SHOULD be answered by the KB but isn't (plausible but out-of-scope)"
    + "\nExample: If anchor is 'Returns Policy', ask 'What is the procedure for returning an item purchased by a third party as a gift?' (Returns covered, but this edge case is not)."
    + "\nground_truth: 'I don't know based on the KB.'"
)

ANCHORED_CROSS_CATEGORY_GAP_PROMPT = (
    BASE_INSTRUCTIONS
    + "\n\nQuery subtype: CROSS-CATEGORY GAP NEGATIVE"
    + "\n\nAnchor page (content):\n\n{anchor_content}\n\n"
    + "Linked pages (content):\n\n{linked_contents}\n\n"
    + "Anchor metadata:\n\n{anchor_meta}\n\n"
    + "Full KB Topic Summary:\n\n{kb_summary}\n\n"
    + "LEXICAL TRAP INSTRUCTION: Generate a question that:"
    + "\n  1. Requires connecting 2+ DIFFERENT CATEGORIES (e.g., returns + loyalty + payments)"
    + "\n  2. Uses keywords from both anchor and related pages (high overall similarity)"
    + "\n  3. Creates a HIDDEN DEPENDENCY: KB has all pieces but no explicit link between them"
    + "\n  4. Plausible but unanswerable because the KB doesn't provide the connection"
    + "\nExample: 'If I return an item purchased with loyalty points, do I get the points back or a refund?' (Returns page + Loyalty page exist, but neither connects them)."
    + "\nground_truth: 'I don't know based on the KB.'"
)

__all__ = [
    "build_direct_prompt",
    "build_multi_hop_prompt",
    "build_anchored_negative_prompt",
]


def build_direct_prompt(content: str, subtype: str = "simple_fact") -> str:
    """Build direct query prompt based on subtype."""
    prompts = {
        "simple_fact": DIRECT_SIMPLE_FACT_PROMPT,
        "table_lookup": DIRECT_TABLE_LOOKUP_PROMPT,
        "table_aggregation": DIRECT_TABLE_AGGREGATION_PROMPT,
        "process_step": DIRECT_PROCESS_STEP_PROMPT,
        "conditional_logic": DIRECT_CONDITIONAL_LOGIC_PROMPT,
        "list_enumeration": DIRECT_LIST_ENUMERATION_PROMPT,
        "rot_aware": DIRECT_ROT_AWARE_PROMPT,
    }
    template = prompts.get(subtype, DIRECT_SIMPLE_FACT_PROMPT)
    return template.format(content=content)


def build_multi_hop_prompt(
    content_a: str, content_b: str, subtype: str = "sequential_process"
) -> str:
    """Build multi-hop query prompt based on subtype."""
    prompts = {
        "sequential_process": MULTI_HOP_SEQUENTIAL_PROMPT,
        "policy_faq_cross": MULTI_HOP_POLICY_FAQ_PROMPT,
        "comparative": MULTI_HOP_COMPARATIVE_PROMPT,
        "hub_to_detail": MULTI_HOP_HUB_TO_DETAIL_PROMPT,
        "cross_category": MULTI_HOP_CROSS_CATEGORY_PROMPT,
        "rot_aware": MULTI_HOP_ROT_AWARE_PROMPT,
    }
    template = prompts.get(subtype, MULTI_HOP_SEQUENTIAL_PROMPT)
    return template.format(content_a=content_a, content_b=content_b)


def build_anchored_negative_prompt(
    anchor_content: str,
    linked_contents: str,
    anchor_meta: str,
    kb_summary: str,
    num_queries: int = 1,
    subtype: str = "adjacent_topic",
) -> str:
    """Build anchored negative prompt based on subtype."""
    prompts = {
        "adjacent_topic": ANCHORED_ADJACENT_TOPIC_PROMPT,
        "missing_data": ANCHORED_MISSING_DATA_PROMPT,
        "out_of_scope_procedure": ANCHORED_OUT_OF_SCOPE_PROMPT,
        "cross_category_gap": ANCHORED_CROSS_CATEGORY_GAP_PROMPT,
    }
    template = prompts.get(subtype, ANCHORED_ADJACENT_TOPIC_PROMPT)
    return template.format(
        anchor_content=anchor_content,
        linked_contents=linked_contents,
        anchor_meta=anchor_meta,
        kb_summary=kb_summary,
        num_queries=num_queries,
    )
