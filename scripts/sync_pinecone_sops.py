"""Index the reviewed Markdown SOP sections in Pinecone.

The command is deliberately separate from API startup: document indexing is a
controlled deployment operation, not a side effect of serving inspections.
"""
from __future__ import annotations

from pathlib import Path

from app.pinecone_rag import PineconeSopRetriever, index_approved_procedures


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    retriever = PineconeSopRetriever.from_environment()
    count = index_approved_procedures(retriever.index, root / "docs", retriever.namespace)
    print(f"Indexed {count} approved SOP sections in namespace {retriever.namespace!r}.")


if __name__ == "__main__":
    main()
