"""Download and verify the immutable grid-model release from Hugging Face."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class GridModelArtifact:
    root: Path
    model_dir: Path
    stn_path: Path
    model_id: str
    revision: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release(root: Path) -> dict[str, object]:
    """Verify every model-release file against its published manifest."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json in model release: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported or missing model artifact format version.")
    for item in manifest.get("files", []):
        path = root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file is missing: {item['path']}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"Checksum mismatch for model artifact: {item['path']}")
    required = (root / "grid_stn.pth", root / "patchcore_model" / "config.json")
    if not all(path.exists() for path in required):
        raise FileNotFoundError("Release does not contain the STN checkpoint and PatchCore configuration.")
    return manifest


def download_grid_model(
    repo_id: str,
    revision: str = "main",
    cache_dir: str | None = None,
) -> GridModelArtifact:
    """Fetch a public/private release and reject incomplete or modified downloads."""
    root = Path(snapshot_download(repo_id=repo_id, revision=revision, local_dir=cache_dir)).resolve()
    manifest = verify_release(root)
    return GridModelArtifact(
        root=root,
        model_dir=root / "patchcore_model",
        stn_path=root / "grid_stn.pth",
        model_id=str(manifest["model_id"]),
        revision=revision,
    )
