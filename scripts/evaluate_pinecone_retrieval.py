"""Measure semantic retrieval quality against the live, indexed Pinecone SOP corpus.

This is an integration benchmark, not a unit test. It needs valid Pinecone
credentials and a completed ``sync_pinecone_sops.py`` run, so it is intentionally
kept outside the default pytest suite.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from app.pinecone_rag import PineconeSopRetriever
from app.rag_evaluation import evaluate, load_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-recall-at-1",
        type=float,
        default=None,
        help="Optional quality gate; exit non-zero below this Recall@1 target.",
    )
    parser.add_argument(
        "--min-section-accuracy-at-1",
        type=float,
        default=None,
        help="Optional quality gate; exit non-zero below this section-accuracy target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    retriever = PineconeSopRetriever.from_environment()
    cases = load_cases(root / "tests" / "fixtures" / "rag_retrieval_cases.json")

    for case in cases:
        results = retriever.search(case.query, top_k=3)
        actual = [f"{item.document_id} § {item.section} ({item.score:.3f})" for item in results]
        print(
            f"query={case.query!r}\n"
            f"  expected={case.document_id} § {case.section}\n"
            f"  retrieved={actual or ['NO_RESULT']}"
        )

    metrics = evaluate(retriever, cases)
    print("\n=== Pinecone semantic retrieval benchmark ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.1%}")

    failures: list[str] = []
    if args.min_recall_at_1 is not None and metrics["recall_at_1"] < args.min_recall_at_1:
        failures.append(f"Recall@1 {metrics['recall_at_1']:.1%} < {args.min_recall_at_1:.1%}")
    if (
        args.min_section_accuracy_at_1 is not None
        and metrics["section_accuracy_at_1"] < args.min_section_accuracy_at_1
    ):
        failures.append(
            "section_accuracy_at_1 "
            f"{metrics['section_accuracy_at_1']:.1%} < {args.min_section_accuracy_at_1:.1%}"
        )
    if failures:
        raise SystemExit("RAG quality gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
