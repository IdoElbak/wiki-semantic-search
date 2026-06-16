"""Preprocessing and chunking for the retrieval pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str
    kind: str = "chunk"


def _is_header(para: str) -> bool:
    """Detect Wikipedia-style section headers."""
    words = para.split()
    return (
        len(words) <= 6
        and len(para) < 60
        and not any(c in para for c in [",", ";", "(", "—", '"', "."])
    )


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    page_id = int(record["page_id"])
    title   = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks: List[Chunk] = []
    chunk_id = 0

    # 1. Title-only chunk — helps entity/name queries
    if title:
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id,
                            text=title, kind="title"))
        chunk_id += 1

    # 2. Lead paragraph chunk — most fact-dense part
    if paragraphs:
        lead = paragraphs[0]
        lead_text = f"{title}. {lead}".strip() if title else lead
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id,
                            text=lead_text, kind="lead"))
        chunk_id += 1

    # 3. Section-aware or sliding window for the rest
    has_sections = sum(1 for p in paragraphs if _is_header(p)) >= 2

    if has_sections:
        current_section = "General"
        buckets: Dict[str, List[str]] = {}
        order: List[str] = []

        for para in paragraphs[1:]:  # skip lead already added
            if _is_header(para):
                current_section = para.rstrip(".")
            else:
                if current_section not in buckets:
                    buckets[current_section] = []
                    order.append(current_section)
                buckets[current_section].append(para)

        for section in order:
            if chunk_id >= 6:  # max 6 chunks per page
                break
            combined = " ".join(buckets[section])
            prefix = f"{title} - {section}" if title else section
            words = combined.split()

            if len(words) <= 200:
                chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id,
                                    text=f"{prefix}. {combined}", kind="chunk"))
                chunk_id += 1
            else:
                # Split long sections into 2 sub-chunks max
                mid = len(words) // 2
                for part in [words[:mid], words[mid:]]:
                    if chunk_id >= 6:
                        break
                    chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id,
                                        text=f"{prefix}. {' '.join(part)}", kind="chunk"))
                    chunk_id += 1
    else:
        # Sliding window over sentences — max 4 body chunks
        sentences = _split_sentences(content)
        WINDOW = 5
        STRIDE = 3
        MAX_BODY = 4

        idx = 0
        body_count = 0
        while idx < len(sentences) and body_count < MAX_BODY:
            window_text = " ".join(sentences[idx: idx + WINDOW])
            text = f"{title}. {window_text}".strip() if title else window_text
            chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id,
                                text=text, kind="chunk"))
            chunk_id += 1
            body_count += 1
            idx += STRIDE

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks