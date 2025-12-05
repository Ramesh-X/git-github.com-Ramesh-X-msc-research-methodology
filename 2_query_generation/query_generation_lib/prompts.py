BASE_INSTRUCTIONS = (
    "You are an assistant generating evaluation queries for a retail customer support knowledge base."
    " Produce a single question, a concise ground_truth answer if available, and give a difficulty (easy|medium|hard) and category."
)

DIRECT_PROMPT_TEMPLATE = (
    BASE_INSTRUCTIONS
    + "\n\nContext (Source page content):\n\n{content}\n\nNotes: The question must be answerable from the provided page ONLY. ground_truth must be precise and concise."
)

MULTI_HOP_PROMPT_TEMPLATE = (
    BASE_INSTRUCTIONS
    + "\n\nContext (Both pages):\n\n{content_a}\n\n---\n\n{content_b}\n\nNotes: The question must require information from BOTH pages to answer. Do not include the answer text that is not supported by the pages."
)

ANCHORED_NEGATIVE_PROMPT_TEMPLATE = (
    BASE_INSTRUCTIONS
    + "\n\nAnchor page (content):\n\n{anchor_content}\n\n"
    + "Linked pages (content):\n\n{linked_contents}\n\n"
    + "Anchor metadata:\n\n{anchor_meta}\n\n"
    + "Full KB Topic Summary:\n\n{kb_summary}\n\n"
    + "Notes: Generate {num_queries} distinct questions that are semantically close to the anchor page, and plausible to look answerable from it, but actually are not answered by the entire KB. The ground_truth must be 'I don't know based on the KB.' Keep questions specific and not general."
)

__all__ = [
    "build_direct_prompt",
    "build_multi_hop_prompt",
    "build_anchored_negative_prompt",
]


def build_direct_prompt(content: str) -> str:
    return DIRECT_PROMPT_TEMPLATE.format(content=content)


def build_multi_hop_prompt(content_a: str, content_b: str) -> str:
    return MULTI_HOP_PROMPT_TEMPLATE.format(content_a=content_a, content_b=content_b)


def build_anchored_negative_prompt(
    anchor_content: str,
    linked_contents: str,
    anchor_meta: str,
    kb_summary: str,
    num_queries: int = 1,
) -> str:
    return ANCHORED_NEGATIVE_PROMPT_TEMPLATE.format(
        anchor_content=anchor_content,
        linked_contents=linked_contents,
        anchor_meta=anchor_meta,
        kb_summary=kb_summary,
        num_queries=num_queries,
    )
