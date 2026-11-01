"""Publish a previously exported model release to Hugging Face.

Usage:
  HF_TOKEN=... python scripts/publish_model.py \
    --release releases/grid-patchcore-v1 --repo-id <account>/grid-inspection-patchcore-v1
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repository, e.g. account/name")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    release = Path(args.release)
    if not (release / "manifest.json").is_file():
        raise FileNotFoundError("Release must contain manifest.json. Run training/export first.")
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise SystemExit("Install publishing extras first: pip install -e '.[publish]'") from error
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(release),
                      commit_message=f"Publish {release.name}")
    print(f"Published https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
