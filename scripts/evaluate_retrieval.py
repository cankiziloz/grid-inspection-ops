"""Run keyword/natural-language retrieval regression metrics."""
from __future__ import annotations

from pathlib import Path

from app.rag import SopRetriever
from app.rag_evaluation import evaluate, load_cases


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    retriever = SopRetriever(root / "docs")
    cases = load_cases(root / "tests" / "fixtures" / "rag_retrieval_cases.json")
    for case in cases:
        top = retriever.search(case.query, top_k=1)
        actual = f"{top[0].document_id} § {top[0].section}" if top else "NO_RESULT"
        print(f"query={case.query!r}\n  expected={case.document_id} § {case.section}\n  retrieved={actual}")
    for metric, value in evaluate(retriever, cases).items():
        print(f"{metric}: {value:.1%}")


if __name__ == "__main__":
    main()
