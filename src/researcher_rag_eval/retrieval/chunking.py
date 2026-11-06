"""Deterministic Markdown paragraph chunking for retrieval benchmarks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentChunk:
    """One stable evidence unit produced from a source document."""

    document_id: str
    chunk_id: str
    source_path: str
    position: int
    text: str


def chunk_markdown_document(path: Path) -> tuple[DocumentChunk, ...]:
    """Split one Markdown document into non-empty paragraph chunks."""
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Document is empty: {path}")

    paragraphs = tuple(
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", raw_text)
        if paragraph.strip()
    )

    document_id = path.stem
    return tuple(
        DocumentChunk(
            document_id=document_id,
            chunk_id=f"{document_id}::p{position:03d}",
            source_path=path.as_posix(),
            position=position,
            text=paragraph,
        )
        for position, paragraph in enumerate(paragraphs, start=1)
    )


def chunk_markdown_directory(directory: Path) -> tuple[DocumentChunk, ...]:
    """Chunk all Markdown documents in a directory in stable path order."""
    return tuple(
        chunk
        for path in sorted(directory.glob("*.md"))
        for chunk in chunk_markdown_document(path)
    )