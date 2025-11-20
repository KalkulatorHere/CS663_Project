"""Dark Channel Prior single-image dehazing implementation (BGR pipeline).

This module implements the full pipeline described in
“Single Image Haze Removal Using Dark Channel Prior”
by Kaiming He, Jian Sun, and Xiaoou Tang (CVPR 2009).

The core steps follow the mathematics of the paper:
1. Atmospheric scattering model: I(x) = J(x) t(x) + A (1 - t(x))
2. Dark channel prior statistics for haze-free images
3. Transmission estimation via normalized dark channel
4. Edge-aware refinement (guided filter approximation to soft matting)
5. Scene radiance recovery with lower-bounded transmission
6. Optional depth-map recovery using t(x) = exp(-β d(x))
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Display/post-processing tuning knobs
# Edit the constants below to adjust brightness and sharpness everywhere.
# ---------------------------------------------------------------------------
GAMMA = 0.85       # >1.0 brightens mid-tones, <1.0 darkens
EXPOSURE_GAIN = 1.3  # Multiplier applied before gamma
APPLY_SHARPEN = False  # Enable unsharp masking post-process

def _ensure_float_image(image: np.ndarray) -> np.ndarray:
    """Return float32 BGR image in [0, 1]."""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if np.issubdtype(image.dtype, np.floating):
        return np.clip(image.astype(np.float32), 0.0, 1.0)
    info = np.iinfo(image.dtype)
    return image.astype(np.float32) / float(info.max)


def _guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Edge-aware smoothing (guided filter) approximation to soft matting."""
    # convert to float32
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)

    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_guide = cv2.boxFilter(guide, -1, ksize, normalize=True)
    mean_src = cv2.boxFilter(src, -1, ksize, normalize=True)
    mean_guide_src = cv2.boxFilter(guide * src, -1, ksize, normalize=True)
    cov_guide_src = mean_guide_src - mean_guide * mean_src

    mean_guide_sq = cv2.boxFilter(guide * guide, -1, ksize, normalize=True)
    var_guide = mean_guide_sq - mean_guide * mean_guide

    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(a, -1, ksize, normalize=True)
    mean_b = cv2.boxFilter(b, -1, ksize, normalize=True)
    return mean_a * guide + mean_b


@dataclass
class DarkChannelPriorDehazer:
    patch_size: int = 15
    omega: float = 0.95
    transmission_floor: float = 0.1
    top_percent: float = 0.001
    guided_radius: int = 40
    guided_eps: float = 1e-3
    depth_beta: float = 1.0
    # Post-processing parameters for brightness/sharpness (edit constants above)
    gamma: float = GAMMA
    exposure_gain: float = EXPOSURE_GAIN
    apply_sharpen: bool = APPLY_SHARPEN

    def _dark_channel(self, image: np.ndarray) -> np.ndarray:
        """Compute J_dark(x) = min_{c} min_{y in Ω(x)} J_c(y)."""
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.patch_size, self.patch_size)
        )
        return cv2.erode(min_channel, kernel)

    def _estimate_atmospheric_light(
        self, image: np.ndarray, dark_channel: np.ndarray
    ) -> np.ndarray:
        """Select brightest input pixel among top dark-channel candidates."""
        flat_dark = dark_channel.ravel()
        num_pixels = flat_dark.size
        num_top = max(int(num_pixels * self.top_percent), 1)
        top_indices = np.argpartition(flat_dark, -num_top)[-num_top:]
        flat_image = image.reshape(-1, 3)
        candidate_pixels = flat_image[top_indices]
        brightness = candidate_pixels.sum(axis=1)
        best_idx = top_indices[np.argmax(brightness)]
        return flat_image[best_idx]

    def _estimate_transmission(
        self, image: np.ndarray, atmospheric_light: np.ndarray
    ) -> np.ndarray:
        """t(x) = 1 - ω * min_c min_y ( I_c(y) / A_c )."""
        eps = 1e-6
        norm_image = image / (atmospheric_light.reshape(1, 1, 3) + eps)
        transmission = 1.0 - self.omega * self._dark_channel(norm_image)
        return np.clip(transmission, 0.0, 1.0)

    def _refine_transmission(
        self, image: np.ndarray, coarse_t: np.ndarray
    ) -> np.ndarray:
        """Refine transmission using guided filtering."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        refined = _guided_filter(
            gray, coarse_t, self.guided_radius, self.guided_eps
        )
        return np.clip(refined, 0.0, 1.0)

    def _recover_radiance(
        self, image: np.ndarray, transmission: np.ndarray, atmospheric_light: np.ndarray
    ) -> np.ndarray:
        """Recover J(x) = (I(x) - A) / max(t(x), t0) + A."""
        t = np.maximum(transmission[..., None], self.transmission_floor)
        recovered = (image - atmospheric_light) / t + atmospheric_light
        return np.clip(recovered, 0.0, 1.0)

    def _estimate_depth(self, transmission: np.ndarray) -> np.ndarray:
        """Depth up to scale from t(x) = exp(-β d(x)) => d(x) = -ln(t)/β."""
        beta = max(self.depth_beta, 1e-6)
        depth = -np.log(np.clip(transmission, 1e-6, 1.0)) / beta
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth_norm

    def _apply_post_processing(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction, exposure adjustment, and optional sharpening."""
        result = image.copy()
        
        # Exposure gain (brightness multiplier)
        if self.exposure_gain != 1.0:
            result = result * self.exposure_gain
        
        # Gamma correction
        if self.gamma != 1.0:
            result = np.power(np.clip(result, 0.0, 1.0), 1.0 / self.gamma)
        
        # Unsharp masking for sharpness (optional)
        if self.apply_sharpen:
            # Convert to uint8 for OpenCV operations
            img_8bit = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            # Gaussian blur for unsharp mask
            blurred = cv2.GaussianBlur(img_8bit, (0, 0), 1.0)
            # Unsharp mask: original + (original - blurred) * amount
            sharpened = cv2.addWeighted(img_8bit, 1.5, blurred, -0.5, 0)
            result = sharpened.astype(np.float32) / 255.0
        
        return np.clip(result, 0.0, 1.0)

    def dehaze(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full pipeline returning (recovered_image, transmission, depth_map)."""
        image = _ensure_float_image(image)

        dark = self._dark_channel(image)
        atmospheric_light = self._estimate_atmospheric_light(image, dark)
        coarse_transmission = self._estimate_transmission(image, atmospheric_light)
        refined_transmission = self._refine_transmission(image, coarse_transmission)
        recovered = self._recover_radiance(image, refined_transmission, atmospheric_light)
        
        # Apply post-processing (gamma, exposure, sharpening)
        recovered = self._apply_post_processing(recovered)
        
        depth = self._estimate_depth(refined_transmission)
        return recovered, refined_transmission, depth


def _save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_8bit = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image_8bit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image haze removal using the Dark Channel Prior."
    )
    parser.add_argument("input", type=Path, help="Path to hazy input image.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dehazed.png"),
        help="Output path for dehazed image.",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        default=Path("outputs/depth.png"),
        help="Output path for relative depth map.",
    )
    parser.add_argument(
        "--transmission",
        type=Path,
        default=Path("outputs/transmission.png"),
        help="Output path for refined transmission map.",
    )
    parser.add_argument("--patch", type=int, default=15, help="Patch size for dark channel.")
    parser.add_argument("--omega", type=float, default=0.88, help="Haze retention factor.")
    parser.add_argument("--t0", type=float, default=0.06, help="Transmission floor.")
    parser.add_argument(
        "--top_percent",
        type=float,
        default=0.001,
        help="Fraction of brightest dark-channel pixels for atmospheric light.",
    )
    parser.add_argument("--guided_radius", type=int, default=40, help="Guided filter radius.")
    parser.add_argument("--guided_eps", type=float, default=1e-3, help="Guided filter epsilon.")
    parser.add_argument("--beta", type=float, default=1.0, help="Depth scaling parameter β.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {args.input}")

    dehazer = DarkChannelPriorDehazer(
        patch_size=args.patch,
        omega=args.omega,
        transmission_floor=args.t0,
        top_percent=args.top_percent,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
        depth_beta=args.beta,
    )

    recovered, transmission, depth = dehazer.dehaze(image)

    _save_image(args.output, recovered)
    _save_image(args.transmission, transmission)
    _save_image(args.depth, depth)


if __name__ == "__main__":
    main()

