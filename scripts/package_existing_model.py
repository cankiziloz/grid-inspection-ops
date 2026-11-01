"""Create an immutable Hugging Face-ready release from an existing artifact.

This command never trains, calibrates, or changes the source model. It copies
the supplied STN checkpoint and PatchCore memory banks exactly as they are and
adds provenance, checksums, evaluation evidence, and a model card.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_summary(path: Path) -> dict[str, object]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Evaluation CSV is empty.")
    cases: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        cases.setdefault(row["case"], []).append(row)
    per_case = {
        name: round(sum(row["correct"] == "True" for row in case_rows) / len(case_rows), 6)
        for name, case_rows in cases.items()
    }
    return {
        "samples": len(rows),
        "overall_policy_accuracy": round(sum(row["correct"] == "True" for row in rows) / len(rows), 6),
        "macro_case_accuracy": round(sum(per_case.values()) / len(per_case), 6),
        "per_case_policy_accuracy": per_case,
        "source_file": "evaluation.csv",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stn", required=True, help="Existing grid STN .pth file")
    parser.add_argument("--patchcore", required=True, help="Existing PatchCore directory containing config.json")
    parser.add_argument("--evaluation", required=True, help="Existing per-image evaluation CSV")
    parser.add_argument("--output", required=True, help="New, empty release directory")
    parser.add_argument("--model-id", default="grid-inspection-patchcore-v1")
    args = parser.parse_args()

    stn, patchcore, evaluation, output = map(Path, (args.stn, args.patchcore, args.evaluation, args.output))
    config_path = patchcore / "config.json"
    if not stn.is_file() or not config_path.is_file() or not evaluation.is_file():
        raise FileNotFoundError("Expected STN checkpoint, PatchCore config.json, and evaluation CSV.")
    config = json.loads(config_path.read_text())
    # Legacy checkpoints were trained via ``pretrained=True`` on torchvision,
    # which resolves to ImageNet V1. Record it explicitly in the release so
    # a future runtime cannot silently choose a different DEFAULT revision.
    config.setdefault("backbone_weights", "IMAGENET1K_V1")
    missing = [name for name in config.get("bank_files", []) if not (patchcore / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing memory-bank files: {', '.join(missing)}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty release directory: {output}")

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stn, output / "grid_stn.pth")
    shutil.copytree(patchcore, output / "patchcore_model")
    (output / "patchcore_model" / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(evaluation, output / "evaluation.csv")

    root = Path(__file__).resolve().parents[1]
    card = (root / "model-card-template.md").read_text()
    metrics = evaluation_summary(output / "evaluation.csv")
    card += "\n## Exported release evidence\n\n"
    card += f"- Policy accuracy: {metrics['overall_policy_accuracy']:.2%}\n"
    card += f"- Macro case accuracy: {metrics['macro_case_accuracy']:.2%}\n"
    card += f"- Samples: {metrics['samples']}\n"
    card += f"- Review threshold: {config['review_threshold']:.9f}\n"
    card += f"- Reject threshold: {config['reject_threshold']:.9f}\n"
    (output / "README.md").write_text(card)

    files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "format_version": 1,
        "model_id": args.model_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "Packaged from a pre-existing validated artifact; no retraining or recalibration occurred.",
        "task": "one-class anomaly detection and defect segmentation for MVTec AD grid",
        "preprocessing": {"resize": [512, 512], "color_space": "RGB", "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
        "architecture": {"stn": "FFT-initialized STN", "backbone": config["backbone_name"], "backbone_weights": config["backbone_weights"], "layers": config["layers"], "memory_size_per_layer": config["memory_size_per_layer"]},
        "thresholds": {"review": config["review_threshold"], "reject": config["reject_threshold"]},
        "evaluation": metrics,
        "files": [{"path": str(path.relative_to(output)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(files)],
        "license_notice": "MVTec AD is CC BY-NC-SA 4.0; this artifact is for non-commercial research/portfolio use.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Packaged unchanged source artifact into: {output}")


if __name__ == "__main__":
    main()
