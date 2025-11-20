"""Batch evaluator for the Dark Channel Prior dehazer on the Kaggle hazing dataset.

Usage:
    python main.py --hazy_dir path/to/dataset/HazyImages \
                   --output_dir outputs/kaggle \
                   --gt_dir path/to/dataset/ClearImages

All dehazed results are written under --output_dir using the same relative
structure as the hazy inputs. If --gt_dir is provided and filenames match,
PSNR and SSIM are reported per image and aggregated in a CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from dark_channel_prior import DarkChannelPriorDehazer, _ensure_float_image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Dark Channel Prior dehazer on a dataset."
    )
    parser.add_argument("--hazy_dir", type=Path, required=True, help="Directory of hazy inputs.")
    parser.add_argument(
        "--gt_dir",
        type=Path,
        default=None,
        help="Optional directory of ground-truth clear images for metrics.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/kaggle_dehazed"),
        help="Directory to store dehazed results.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("outputs/kaggle_metrics.csv"),
        help="CSV path for per-image metrics (only used if gt_dir is set).",
    )
    parser.add_argument("--patch", type=int, default=15)
    parser.add_argument("--omega", type=float, default=0.95)
    parser.add_argument("--t0", type=float, default=0.1)
    parser.add_argument("--top_percent", type=float, default=0.001)
    parser.add_argument("--guided_radius", type=int, default=40)
    parser.add_argument("--guided_eps", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    return parser.parse_args()


def collect_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.suffix.lower() in VALID_EXTS and path.is_file()
        ]
    )


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def compute_metrics(
    dehazed: np.ndarray, gt: np.ndarray
) -> Tuple[float, float]:
    """Return PSNR and SSIM in [float, float]."""
    dehazed = _ensure_float_image(cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB))
    gt = _ensure_float_image(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))
    if dehazed.shape != gt.shape:
        gt = cv2.resize(gt, (dehazed.shape[1], dehazed.shape[0]), interpolation=cv2.INTER_CUBIC)
    psnr_val = peak_signal_noise_ratio(gt, dehazed, data_range=1.0)
    ssim_val = structural_similarity(gt, dehazed, channel_axis=2, data_range=1.0)
    return float(psnr_val), float(ssim_val)


def main() -> None:
    args = parse_args()
    hazy_paths = collect_images(args.hazy_dir)
    if not hazy_paths:
        raise RuntimeError(f"No images found under {args.hazy_dir}")

    dehazer = DarkChannelPriorDehazer(
        patch_size=args.patch,
        omega=args.omega,
        transmission_floor=args.t0,
        top_percent=args.top_percent,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
        depth_beta=args.beta,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: List[Dict[str, float]] = []

    for hazy_path in tqdm(hazy_paths, desc="Dehazing images"):
        image = load_image(hazy_path)
        recovered, _, _ = dehazer.dehaze(image)

        rel_path = hazy_path.relative_to(args.hazy_dir)
        out_path = args.output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), (recovered * 255).astype(np.uint8))

        if args.gt_dir:
            gt_path = args.gt_dir / rel_path
            if gt_path.exists():
                gt_img = load_image(gt_path)
                psnr_val, ssim_val = compute_metrics((recovered * 255).astype(np.uint8), gt_img)
                metrics_rows.append(
                    {"image": str(rel_path).replace("\\", "/"), "psnr": psnr_val, "ssim": ssim_val}
                )
            else:
                print(f"[WARN] Missing ground-truth for {rel_path}; skipping metrics.")

    if metrics_rows and args.gt_dir:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["image", "psnr", "ssim"])
            writer.writeheader()
            writer.writerows(metrics_rows)

        psnr_mean = np.mean([row["psnr"] for row in metrics_rows])
        ssim_mean = np.mean([row["ssim"] for row in metrics_rows])
        print(f"Saved per-image metrics to {args.csv_path}")
        print(f"Mean PSNR: {psnr_mean:.3f}, Mean SSIM: {ssim_mean:.3f}")


if __name__ == "__main__":
    main()

