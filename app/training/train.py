"""Train, evaluate, and export the rotation-aware grid PatchCore release.

Example:
    python -m app.training.train --dataset-root /path/to/mvtec --output releases/grid-patchcore-v1

Only ``grid/train/good`` is used to build the memory banks. Thresholds are
calibrated on a deterministic holdout of those normal images; test folders are
used only for the final report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from ..vision.legacy_patchcore import GridSTN

PREPROCESSING = {
    "resize": [512, 512],
    "color_space": "RGB",
    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
}
TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(**PREPROCESSING["normalization"]),
])


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str
    output: str
    seed: int = 42
    batch_size: int = 16
    workers: int = 0
    validation_fraction: float = 0.20
    memory_size_per_layer: int = 5000
    review_percentile: float = 95.0
    reject_percentile: float = 99.5
    layers: tuple[str, ...] = ("layer2", "layer3")


class ImagePaths(Dataset[tuple[torch.Tensor, Path]]):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Path]:
        path = self.paths[index]
        return TRANSFORM(Image.open(path).convert("RGB")), path


def collate(batch: list[tuple[torch.Tensor, Path]]) -> tuple[torch.Tensor, list[Path]]:
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


class PatchCoreTrainer:
    def __init__(self, config: TrainConfig) -> None:
        if not 0 < config.review_percentile < config.reject_percentile < 100:
            raise ValueError("Expected 0 < review_percentile < reject_percentile < 100.")
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Pin the encoder weights. PatchCore memory banks are incompatible with
        # embeddings from a different ImageNet weight revision.
        self.backbone = wide_resnet50_2(
            weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1
        ).to(self.device).eval()
        self.features: list[torch.Tensor] = []
        modules = dict(self.backbone.named_modules())
        self.hooks = [modules[name].register_forward_hook(self._hook) for name in config.layers]
        self.stn = GridSTN(use_fft_init=True).to(self.device).eval()
        self.memory_banks: list[torch.Tensor] = []

    def _hook(self, _module: torch.nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> None:
        self.features.append(F.normalize(output, p=2, dim=1))

    def _features(self, images: torch.Tensor) -> list[torch.Tensor]:
        self.features = []
        _ = self.backbone(images)
        return self.features

    def _sample(self, features: torch.Tensor) -> torch.Tensor:
        """Deterministic farthest-first cosine coreset selection."""
        size = self.config.memory_size_per_layer
        if len(features) <= size:
            return features
        normalized = F.normalize(features.float(), p=2, dim=1)
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        selected = [int(torch.randint(len(normalized), (1,), generator=generator).item())]
        best_similarity = normalized @ normalized[selected[0]]
        for _ in range(1, size):
            next_index = int(torch.argmin(best_similarity).item())
            selected.append(next_index)
            best_similarity = torch.maximum(best_similarity, normalized @ normalized[next_index])
        return features[selected]

    def fit(self, loader: DataLoader[tuple[torch.Tensor, list[Path]]]) -> None:
        per_layer: list[list[torch.Tensor]] = [[] for _ in self.config.layers]
        with torch.inference_mode():
            for images, _ in loader:
                corrected, _, valid = self.stn(images.to(self.device))
                for index, feature in enumerate(self._features(corrected)):
                    batch, channels, height, width = feature.shape
                    mask = F.interpolate(valid, (height, width), mode="bilinear", align_corners=True) > 0.5
                    flattened = feature.permute(0, 2, 3, 1).reshape(batch, -1, channels)
                    for item, item_mask in zip(flattened, mask.reshape(batch, -1)):
                        if item_mask.any():
                            per_layer[index].append(item[item_mask])
        self.memory_banks = [self._sample(torch.cat(items)).cpu() for items in per_layer]
        if len(self.memory_banks) != len(self.config.layers):
            raise RuntimeError("No valid features were collected for one or more configured layers.")

    def score(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.memory_banks:
            raise RuntimeError("Call fit() before score().")
        with torch.inference_mode():
            images = images.to(self.device)
            corrected, _, valid = self.stn(images)
            maps = []
            for feature, bank in zip(self._features(corrected), self.memory_banks):
                batch, channels, height, width = feature.shape
                flat = feature.permute(0, 2, 3, 1).reshape(-1, channels)
                similarity = flat @ bank.to(self.device).T
                anomaly = (1 - similarity.max(dim=1).values).reshape(batch, 1, height, width)
                maps.append(F.interpolate(anomaly, images.shape[-2:], mode="bilinear", align_corners=True))
            score_map = torch.stack(maps).mean(0) * valid
            return score_map, valid

    def calibrate(self, loader: DataLoader[tuple[torch.Tensor, list[Path]]]) -> tuple[float, float]:
        scores: list[float] = []
        for images, _ in loader:
            score_map, _ = self.score(images)
            scores.extend(score_map.flatten(1).max(dim=1).values.cpu().tolist())
        if not scores:
            raise ValueError("The validation split is empty.")
        return (float(np.percentile(scores, self.config.review_percentile)),
                float(np.percentile(scores, self.config.reject_percentile)))

    def save(self, output: Path, review: float, reject: float) -> None:
        patchcore_dir = output / "patchcore_model"
        patchcore_dir.mkdir(parents=True, exist_ok=True)
        bank_files = []
        for index, bank in enumerate(self.memory_banks):
            name = f"memory_bank_{index}.pt"
            torch.save(bank, patchcore_dir / name)
            bank_files.append(name)
        (patchcore_dir / "config.json").write_text(json.dumps({
            "artifact_format": 1, "backbone_name": "wide_resnet50_2",
            "backbone_weights": "IMAGENET1K_V1", "layers": list(self.config.layers),
            "memory_size_per_layer": self.config.memory_size_per_layer, "bank_files": bank_files,
            "review_threshold": review, "reject_threshold": reject,
            "threshold": reject, "threshold_percentile": self.config.reject_percentile,
        }, indent=2) + "\n")
        torch.save({"state_dict": self.stn.state_dict(), "use_fft_init": True}, output / "grid_stn.pth")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output: Path, config: TrainConfig, review: float, reject: float) -> None:
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "format_version": 1,
        "model_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "one-class anomaly detection and defect segmentation for MVTec AD grid",
        "decision_policy": {"pass": f"score < {review}", "inspect": f"{review} <= score < {reject}", "reject": f"score >= {reject}"},
        "preprocessing": PREPROCESSING,
        "training": asdict(config),
        "files": [{"path": str(path.relative_to(output)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(files)],
        "license_notice": "MVTec AD is CC BY-NC-SA 4.0; this artifact is for non-commercial research/portfolio use.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def evaluate(trainer: PatchCoreTrainer, root: Path, review: float, reject: float, output: Path) -> None:
    rows: list[dict[str, object]] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(folder.glob("*.png")):
            image = TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0)
            score_map, _ = trainer.score(image)
            score = float(score_map.max().item())
            decision = "REJECT" if score >= reject else "INSPECT" if score >= review else "PASS"
            actual = folder.name != "good"
            rows.append({"case": folder.name, "image": path.name, "score": score, "decision": decision,
                         "actual_anomaly": actual, "correct": actual == (decision != "PASS")})
    with (output / "evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["case"])
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="Directory containing grid/train and grid/test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-percentile", type=float, default=95.0)
    parser.add_argument("--reject-percentile", type=float, default=99.5)
    parser.add_argument("--memory-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    config = TrainConfig(dataset_root=args.dataset_root, output=args.output, seed=args.seed, batch_size=args.batch_size,
                         workers=args.workers, memory_size_per_layer=args.memory_size,
                         review_percentile=args.review_percentile, reject_percentile=args.reject_percentile)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    output = Path(config.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty release directory: {output}")
    normal_paths = sorted((Path(config.dataset_root) / "grid" / "train" / "good").glob("*.png"))
    if len(normal_paths) < 2:
        raise FileNotFoundError("Expected normal training images at <dataset-root>/grid/train/good.")
    train_size = int(len(normal_paths) * (1 - config.validation_fraction))
    train, validation = random_split(ImagePaths(normal_paths), [train_size, len(normal_paths) - train_size],
                                     generator=torch.Generator().manual_seed(config.seed))
    loader_args = {"batch_size": config.batch_size, "num_workers": config.workers, "collate_fn": collate}
    trainer = PatchCoreTrainer(config)
    trainer.fit(DataLoader(train, shuffle=True, **loader_args))
    review, reject = trainer.calibrate(DataLoader(validation, shuffle=False, **loader_args))
    trainer.save(output, review, reject)
    test_root = Path(config.dataset_root) / "grid" / "test"
    if test_root.is_dir():
        evaluate(trainer, test_root, review, reject, output)
    write_manifest(output, config, review, reject)
    print(f"Export complete: {output}\nreview={review:.6f}, reject={reject:.6f}")


if __name__ == "__main__":
    main()
