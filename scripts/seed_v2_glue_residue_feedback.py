"""Mark pending demo glue inspections as accepted process-residue feedback for V2."""

from __future__ import annotations

import argparse

from app.case_memory import seed_demo_glue_residue_feedback
from app.repository import SqliteInspectionRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="runtime/inspections.db")
    args = parser.parse_args()

    result = seed_demo_glue_residue_feedback(SqliteInspectionRepository(args.database))
    print(
        "V2 demo glue residue feedback: "
        f"resolved={result.resolved}, annotated={result.annotated}, skipped={result.skipped}."
    )


if __name__ == "__main__":
    main()
