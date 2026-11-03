"""Send a shuffled Grid test corpus through the production-like edge ingress API.

This is a demo device simulator, not a model evaluator. Folder names are shown
only as simulator ground truth; the PatchCore model does not receive them.
"""
from __future__ import annotations

import argparse
import hashlib
import mimetypes
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Grid test directory containing case folders")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--station-id", default="demo-line-2")
    parser.add_argument("--camera-id", default="demo-line-2-camera-1")
    parser.add_argument("--max-events", type=int, default=0, help="0 runs continuously")
    parser.add_argument("--seed", type=int, default=None, help="Use a seed only when reproducing a demo sequence")
    return parser.parse_args()


def discover_images(dataset_root: Path) -> list[Path]:
    images = [path for path in dataset_root.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not images:
        raise ValueError(f"No PNG/JPEG files found below {dataset_root}")
    return images


def multipart_request(url: str, fields: dict[str, str], image_path: Path) -> dict:
    boundary = f"----grid-inspection-{uuid.uuid4().hex}"
    body: list[bytes] = []
    for name, value in fields.items():
        body.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    body.extend(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            image_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    request = Request(
        f"{url.rstrip('/')}/v1/edge/inspections",
        data=b"".join(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(request, timeout=90) as response:  # nosec B310: explicit local/demo API URL
        import json

        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    args = parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be greater than zero")
    randomizer = random.Random(args.seed)
    images = discover_images(args.dataset_root)
    sent = 0
    queue: list[Path] = []
    print(f"Loaded {len(images)} images; sending a shuffled edge feed every {args.interval_seconds:g}s.")

    while args.max_events == 0 or sent < args.max_events:
        if not queue:
            queue = images.copy()
            randomizer.shuffle(queue)
        image_path = queue.pop()
        payload = image_path.read_bytes()
        case = image_path.parent.name
        frame_id = f"sim-{uuid.uuid4()}"
        fields = {
            "part_id": f"demo-{case}-{sent + 1:05d}",
            "station_id": args.station_id,
            "camera_id": args.camera_id,
            "frame_id": frame_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "image_sha256": hashlib.sha256(payload).hexdigest(),
        }
        try:
            result = multipart_request(args.api_url, fields, image_path)
            print(
                f"[{sent + 1}] simulator_case={case} frame={frame_id[-8:]} "
                f"decision={result['decision']} score={result['evidence']['score']:.4f}"
            )
        except (HTTPError, URLError, TimeoutError, KeyError) as error:
            print(f"[{sent + 1}] delivery failed for {image_path.name}: {error}")
        sent += 1
        if args.max_events == 0 or sent < args.max_events:
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
