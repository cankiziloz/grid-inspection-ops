from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from ..domain import DefectRegion, ImageQuality, VisionEvidence


class GridSTN(torch.nn.Module):
    """Batch-safe FFT initialized STN compatible with the trained grid checkpoint."""

    def __init__(self, use_fft_init: bool = True) -> None:
        super().__init__()
        self.use_fft_init = use_fft_init
        self.refinement_net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 7, stride=2, padding=3), torch.nn.BatchNorm2d(16), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 5, stride=2, padding=2), torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)), torch.nn.Flatten(),
            torch.nn.Linear(128 * 4 * 4, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1),
        )
        # The original grid artifact was trained with a zero residual angle.
        # Keep that initialization for reproducible re-training.
        for module in self.refinement_net.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.zeros_(module.weight)
                torch.nn.init.zeros_(module.bias)

    @staticmethod
    def _fft_angle(image: torch.Tensor) -> torch.Tensor:
        gray = image.mean(dim=1, keepdim=True)
        spectrum = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1)).abs()
        _, _, height, width = spectrum.shape
        y = torch.arange(height, device=image.device).view(height, 1) - height // 2
        x = torch.arange(width, device=image.device).view(1, width) - width // 2
        i_xx = (spectrum * x.square()).flatten(2).sum(dim=2)
        i_yy = (spectrum * y.square()).flatten(2).sum(dim=2)
        i_xy = (spectrum * (x * y)).flatten(2).sum(dim=2)
        return (0.5 * torch.atan2(2 * i_xy, i_xx - i_yy + 1e-8)).reshape(image.shape[0], -1).mean(1).abs()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, channels, height, width = image.shape
        angle = self._fft_angle(image) if self.use_fft_init else torch.zeros(batch, device=image.device)
        gray = image.mean(dim=1, keepdim=True) if channels == 3 else image
        angle = angle + self.refinement_net(gray).view(batch)
        cos, sin = torch.cos(angle), torch.sin(angle)
        zero = torch.zeros_like(cos)
        theta = torch.stack((torch.stack((cos, -sin, zero), 1), torch.stack((sin, cos, zero), 1)), 1)
        grid = F.affine_grid(theta, image.shape, align_corners=True)
        corrected = F.grid_sample(image, grid, align_corners=True, mode="bilinear")
        source_mask = torch.ones((batch, 1, height, width), device=image.device)
        valid = F.grid_sample(source_mask, grid, align_corners=True, mode="bilinear") > 0.5
        return corrected, angle, valid.float()


class GridPatchCoreVision:
    """Read-only adapter around the trained STN + two-layer PatchCore artifact."""

    preprocess = transforms.Compose([
        transforms.Resize((512, 512), antialias=True), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, model_dir: str, stn_path: str, output_dir: str = "runtime/masks", model_version: str | None = None, artifact_verified: bool = False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = json.loads((Path(model_dir) / "config.json").read_text())
        self.layers: list[str] = payload["layers"]
        self.review_threshold = float(payload["review_threshold"])
        self.reject_threshold = float(payload["reject_threshold"])
        self.model_version = model_version or f"grid-patchcore-{Path(model_dir).resolve().stat().st_mtime_ns}"
        self.artifact_verified = artifact_verified
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Memory banks are feature vectors, not portable across arbitrary
        # pretrained backbones. The published grid-v1 banks were built with V1;
        # never use DEFAULT here because it can silently change to V2.
        weight_name = payload.get("backbone_weights", "IMAGENET1K_V1")
        if weight_name != "IMAGENET1K_V1":
            raise ValueError(f"Unsupported grid artifact backbone weights: {weight_name}")
        self.backbone = wide_resnet50_2(
            weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1
        ).to(self.device).eval()
        self.features: list[torch.Tensor] = []
        modules = dict(self.backbone.named_modules())
        self.hooks = [modules[name].register_forward_hook(self._hook) for name in self.layers]
        self.memory_banks = [
            torch.load(Path(model_dir) / filename, map_location=self.device, weights_only=True).to(self.device)
            for filename in payload["bank_files"]
        ]
        self.stn = GridSTN().to(self.device)
        stn_payload = torch.load(stn_path, map_location=self.device, weights_only=False)
        self.stn.load_state_dict(stn_payload["state_dict"] if "state_dict" in stn_payload else stn_payload)
        self.stn.eval()

    def _hook(self, _module, _inputs, output: torch.Tensor) -> None:
        self.features.append(F.normalize(output, p=2, dim=1))

    def inspect(self, image_path: Path) -> VisionEvidence:
        image = Image.open(image_path).convert("RGB")
        quality = self._quality(image)
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            corrected, _, valid = self.stn(tensor)
            self.features = []
            _ = self.backbone(corrected)
            maps = []
            for feature, bank in zip(self.features, self.memory_banks):
                _, channels, height, width = feature.shape
                flat = feature.permute(0, 2, 3, 1).reshape(-1, channels)
                score = 1 - torch.mm(flat, bank.T).max(dim=1).values
                feature_map = score.view(1, 1, height, width)
                maps.append(F.interpolate(feature_map, size=tensor.shape[-2:], mode="bilinear", align_corners=True))
            score_map = torch.stack(maps).mean(dim=0) * valid
        mask = (score_map[0, 0] >= self.review_threshold) & valid[0, 0].bool()
        score = float(score_map.max().item())
        mask_uri = self._save_mask(image_path.stem, mask)
        return VisionEvidence(
            model_version=self.model_version,
            score=score,
            review_threshold=self.review_threshold,
            reject_threshold=self.reject_threshold,
            valid_coverage=float(valid.float().mean().item()),
            artifact_verified=self.artifact_verified,
            quality=quality,
            mask_uri=str(mask_uri),
            regions=self._regions(mask.cpu().numpy(), score_map[0, 0].cpu().numpy()),
        )

    @staticmethod
    def _quality(image: Image.Image) -> ImageQuality:
        pixels = np.asarray(image)
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        mean = float(gray.mean() / 255.0)
        contrast = float(gray.std() / 255.0)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        reasons: list[str] = []
        if min(image.size) < 256:
            reasons.append("IMAGE_RESOLUTION_TOO_LOW")
        if not 0.05 <= mean <= 0.95:
            reasons.append("IMAGE_EXPOSURE_OUT_OF_RANGE")
        if contrast < 0.02:
            reasons.append("IMAGE_CONTRAST_TOO_LOW")
        if sharpness < 10.0:
            reasons.append("IMAGE_TOO_BLURRY")
        return ImageQuality(width=image.width, height=image.height, mean_luminance=mean, contrast=contrast, laplacian_variance=sharpness, acceptable=not reasons, failure_reasons=reasons)

    def _save_mask(self, stem: str, mask: torch.Tensor) -> Path:
        path = self.output_dir / f"{stem}-{os.urandom(6).hex()}.png"
        cv2.imwrite(str(path), mask.numpy().astype(np.uint8) * 255)
        return path

    @staticmethod
    def _regions(mask: np.ndarray, scores: np.ndarray) -> list[DefectRegion]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        regions = []
        for x, y, width, height, area in stats[1:count]:
            if area >= 8:
                regions.append(DefectRegion(x=int(x), y=int(y), width=int(width), height=int(height), area_px=int(area), peak_score=float(scores[y:y+height, x:x+width].max())))
        return regions
