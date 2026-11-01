"""Download a published Grid Inspection Ops artifact and validate checksums."""
from __future__ import annotations

import argparse

from app.model_artifacts import download_grid_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Example: account/grid-inspection-patchcore-v1")
    parser.add_argument("--revision", default="main", help="Use a commit SHA for a production-pinned release")
    parser.add_argument("--cache-dir", default="runtime/models/grid-patchcore-v1")
    args = parser.parse_args()
    artifact = download_grid_model(args.repo_id, args.revision, args.cache_dir)
    print(f"Downloaded and verified {artifact.model_id}")
    print(f"model_dir={artifact.model_dir}")
    print(f"stn_path={artifact.stn_path}")


if __name__ == "__main__":
    main()
