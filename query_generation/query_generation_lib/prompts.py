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

NEGATIVE_PROMPT_TEMPLATE = (
    BASE_INSTRUCTIONS
    + "\n\nContext (one or more pages):\n\n{content}\n\nFull KB Topic Summary:\n\n{kb_summary}\n\nNotes: Generate questions that sound KB-related but ask for unanswerable specifics to trick vector search. The question should be semantically close to topics in the KB but require information not present anywhere in the full KB. Set ground_truth to 'I don't know based on the KB.'"
)

__all__ = [
    "build_direct_prompt",
    "build_multi_hop_prompt",
    "build_negative_prompt",
]


def build_direct_prompt(content: str) -> str:
    return DIRECT_PROMPT_TEMPLATE.format(content=content)


def build_multi_hop_prompt(content_a: str, content_b: str) -> str:
    return MULTI_HOP_PROMPT_TEMPLATE.format(content_a=content_a, content_b=content_b)


def build_negative_prompt(content: str, kb_summary: str) -> str:
    return NEGATIVE_PROMPT_TEMPLATE.format(content=content, kb_summary=kb_summary)
