---
license: cc-by-nc-sa-4.0
tags:
- anomaly-detection
- defect-segmentation
- patchcore
- mvtec-ad
---

# Grid Inspection PatchCore v1

## Intended use

Non-commercial research and portfolio demonstration: one-class anomaly detection and defect localization for the MVTec AD `grid` category.

## Model and preprocessing

- FFT-initialized spatial transformer for rotation handling
- Wide-ResNet-50-2 features from `layer2` and `layer3`
- PatchCore memory banks built only from normal training images
- Input: RGB, resize to 512×512, ImageNet normalization
- Policy: PASS / INSPECT / REJECT thresholds are recorded in `manifest.json`.

## Evaluation

Include the exported `evaluation.csv`, plus the methodology and seed used to produce it. Report per-defect capture rate and normal-pass rate; do not describe this benchmark result as production accuracy.

## Limitations

The model was evaluated only on MVTec AD grid images. It is not validated for production equipment, new cameras, lighting, rotations outside the observed distribution, or commercial decisions without site-specific validation and human oversight.

## License and attribution

This release is derived from MVTec AD, which is licensed under CC BY-NC-SA 4.0. Cite Bergmann et al., *MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection* (CVPR 2019). Non-commercial use only; see the upstream dataset terms.
