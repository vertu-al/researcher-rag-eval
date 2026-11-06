from pathlib import Path

import pytest

from researcher_rag_eval.retrieval.chunking import chunk_markdown_document


def test_chunker_assigns_stable_paragraph_ids(tmp_path: Path) -> None:
    document = tmp_path / "evidence.md"
    document.write_text("First passage.\n\nSecond passage.", encoding="utf-8")

    chunks = chunk_markdown_document(document)

    assert [chunk.chunk_id for chunk in chunks] == [
        "evidence::p001",
        "evidence::p002",
    ]
    assert [chunk.text for chunk in chunks] == [
        "First passage.",
        "Second passage.",
    ]


def test_chunker_rejects_empty_documents(tmp_path: Path) -> None:
    document = tmp_path / "empty.md"
    document.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Document is empty"):
        chunk_markdown_document(document)