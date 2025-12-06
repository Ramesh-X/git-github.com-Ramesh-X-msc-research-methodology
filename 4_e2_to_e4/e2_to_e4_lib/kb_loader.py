"""Knowledge base loading and chunking utilities."""

import logging
import re
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class MarkdownChunker:
    """Structure-aware chunking for Markdown documents."""

    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, filename: str, content: str) -> List[Dict]:
        """Chunk a markdown document respecting structure.

        Returns:
            List of dicts with: chunk_id, text, metadata
        """
        chunks = []
        sections = self._split_by_headers(content)

        for idx, section in enumerate(sections):
            header = section.get("header", "")
            body = section.get("body", "")

            if len(body) <= self.chunk_size:
                chunks.append(
                    {
                        "chunk_id": f"{filename}#chunk_{idx}",
                        "text": f"{header}\n\n{body}" if header else body,
                        "metadata": {
                            "filename": filename,
                            "section": header,
                            "chunk_index": idx,
                        },
                    }
                )
            else:
                sub_chunks = self._split_with_overlap(
                    body, self.chunk_size, self.overlap
                )
                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    chunks.append(
                        {
                            "chunk_id": f"{filename}#chunk_{idx}_{sub_idx}",
                            "text": f"{header}\n\n{sub_chunk}" if header else sub_chunk,
                            "metadata": {
                                "filename": filename,
                                "section": header,
                                "chunk_index": f"{idx}_{sub_idx}",
                            },
                        }
                    )

        logger.debug(f"Chunked {filename} into {len(chunks)} chunks")
        return chunks

    def _split_by_headers(self, content: str) -> List[Dict]:
        """Split content by markdown headers."""
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(header_pattern.finditer(content))

        if not matches:
            return [{"header": "", "body": content.strip()}]

        sections = []
        for i, match in enumerate(matches):
            header = match.group(0)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[start:end].strip()
            sections.append({"header": header, "body": body})

        return sections

    def _split_with_overlap(self, text: str, size: int, overlap: int) -> List[str]:
        """Split text into chunks with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start = end - overlap
            if start >= len(text):
                break
        return chunks


def load_kb_documents(kb_dir: Path) -> List[Dict]:
    """Load all markdown documents from KB directory.

    Returns:
        List of dicts with: filename, content
    """
    documents = []
    kb_dir = Path(kb_dir)

    for md_file in sorted(kb_dir.glob("*.md")):
        # Skip non-content files
        if md_file.name in ["README.md", "structure.md"]:
            continue

        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        documents.append({"filename": md_file.name, "content": content})

    logger.info(f"Loaded {len(documents)} documents from {kb_dir}")
    return documents
