# Image Quality Metrics: PSNR and SSIM Explained

## Overview

Your code currently uses two metrics to evaluate dehazing quality:
1. **PSNR** (Peak Signal-to-Noise Ratio)
2. **SSIM** (Structural Similarity Index)

Both compare your **dehazed output** against the **ground truth (GT) clear image**.

---

## 1. PSNR (Peak Signal-to-Noise Ratio)

### What It Measures
PSNR measures **pixel-level accuracy** - how close each pixel value is to the ground truth.

### How It's Calculated

**Mathematical Formula:**
```
MSE = mean((GT - Dehazed)²)  # Mean Squared Error
PSNR = 10 × log₁₀(MAX² / MSE)  # in decibels (dB)
```

Where:
- `MAX` = maximum possible pixel value (1.0 for normalized images, 255 for uint8)
- Your code uses `data_range=1.0`, so MAX = 1.0

**In Your Code:**
```python
psnr_val = peak_signal_noise_ratio(gt, dehazed, data_range=1.0)
```

### What the Values Mean

| PSNR (dB) | Quality Interpretation |
|-----------|------------------------|
| > 30 dB | Excellent (very close to GT) |
| 25-30 dB | Good (minor differences) |
| 20-25 dB | Acceptable (noticeable differences) |
| < 20 dB | Poor (significant differences) |

**For Dehazing:**
- **Typical range:** 15-25 dB (dehazing is challenging!)
- **Good results:** 20-25 dB
- **Excellent results:** > 25 dB

### Strengths
✅ Simple, fast to compute  
✅ Standard metric in image processing  
✅ Good for measuring noise/error magnitude

### Limitations for Dehazing
❌ **Doesn't capture perceptual quality** - two images can have similar PSNR but look very different  
❌ **Penalizes brightness shifts** - if your dehazed image is slightly brighter/darker but otherwise correct, PSNR will be low  
❌ **Ignores structure** - doesn't care about edges, textures, or visual structure  
❌ **Sensitive to misalignment** - even small spatial shifts cause large PSNR drops

---

## 2. SSIM (Structural Similarity Index)

### What It Measures
SSIM measures **perceptual similarity** - how similar the images look to a human observer, focusing on **structure, luminance, and contrast**.

### How It's Calculated

**Mathematical Formula:**
```
SSIM(x, y) = [l(x, y)]^α × [c(x, y)]^β × [s(x, y)]^γ

Where:
- l(x, y) = (2μₓμᵧ + c₁) / (μₓ² + μᵧ² + c₁)  # Luminance
- c(x, y) = (2σₓσᵧ + c₂) / (σₓ² + σᵧ² + c₂)  # Contrast
- s(x, y) = (σₓᵧ + c₃) / (σₓσᵧ + c₃)        # Structure

μ = mean, σ = std dev, σₓᵧ = covariance
c₁, c₂, c₃ = small constants to avoid division by zero
```

**In Your Code:**
```python
ssim_val = structural_similarity(gt, dehazed, channel_axis=2, data_range=1.0)
```

### What the Values Mean

| SSIM | Quality Interpretation |
|------|------------------------|
| 0.9 - 1.0 | Excellent (nearly identical) |
| 0.8 - 0.9 | Good (very similar) |
| 0.7 - 0.8 | Acceptable (similar structure) |
| < 0.7 | Poor (different structure) |

**For Dehazing:**
- **Typical range:** 0.6-0.85
- **Good results:** 0.75-0.85
- **Excellent results:** > 0.85

### Strengths
✅ **Perceptually meaningful** - correlates better with human perception  
✅ **Considers structure** - focuses on edges, textures, patterns  
✅ **More robust to brightness shifts** - less sensitive to global brightness differences  
✅ **Multi-scale** - can be computed at different scales (your code uses single scale)

### Limitations for Dehazing
❌ **Still not perfect** - doesn't capture all aspects of visual quality  
❌ **Can miss color issues** - may not penalize color cast as much as it should  
❌ **Computationally slower** than PSNR (but still fast)

---

## Are These Metrics Useful for Dehazing?

### ✅ **Yes, but with caveats:**

**Useful for:**
- **Comparing different algorithms** on the same dataset
- **Tracking improvements** when tuning parameters
- **Objective baseline** when you can't do visual inspection for all images
- **Research papers** - standard metrics everyone uses

**Not perfect for:**
- **Final quality assessment** - always visually inspect results!
- **Color accuracy** - both metrics may miss color cast issues
- **Artifact detection** - halos, over-enhancement, noise may not be well captured
- **Perceptual quality** - what looks "good" to humans may not score highest

---

## How Your Code Calculates Them

```python
def compute_metrics(dehazed: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    # Convert BGR to RGB for proper color comparison
    dehazed = _ensure_float_image(cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB))
    gt = _ensure_float_image(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))
    
    # Resize if dimensions don't match
    if dehazed.shape != gt.shape:
        gt = cv2.resize(gt, (dehazed.shape[1], dehazed.shape[0]), 
                       interpolation=cv2.INTER_CUBIC)
    
    # Calculate metrics (images in [0, 1] range)
    psnr_val = peak_signal_noise_ratio(gt, dehazed, data_range=1.0)
    ssim_val = structural_similarity(gt, dehazed, channel_axis=2, data_range=1.0)
    
    return float(psnr_val), float(ssim_val)
```

**Key points:**
- ✅ Converts BGR→RGB (important for color accuracy)
- ✅ Normalizes to [0, 1] range
- ✅ Handles size mismatches
- ✅ Uses `channel_axis=2` for SSIM (treats RGB as 3 channels)

---

## Interpreting Your Results

### Typical Values for Dehazing

**Good performance:**
- PSNR: 20-25 dB
- SSIM: 0.75-0.85

**Excellent performance:**
- PSNR: > 25 dB
- SSIM: > 0.85

**Poor performance:**
- PSNR: < 18 dB
- SSIM: < 0.70

### What to Look For

1. **Both metrics agree** → More reliable assessment
2. **High SSIM, low PSNR** → Structure is good, but pixel values differ (maybe brightness/color shift)
3. **Low SSIM, high PSNR** → Rare, but could indicate structural artifacts despite pixel accuracy
4. **Both low** → Significant quality issues

---

## Alternative Metrics (Not Currently Implemented)

### 1. **LPIPS** (Learned Perceptual Image Patch Similarity)
- Uses deep learning to measure perceptual similarity
- **Better** for perceptual quality, but requires a pre-trained model
- **Slower** to compute

### 2. **CIEDE2000** (Color Difference)
- Measures color accuracy specifically
- **Better** for detecting color cast issues
- **Useful** for dehazing where color correction is important

### 3. **BRISQUE** (Blind Image Spatial Quality Evaluator)
- No-reference metric (doesn't need GT)
- Measures naturalness/blur/noise
- **Useful** when you don't have ground truth

### 4. **NIQE** (Natural Image Quality Evaluator)
- Another no-reference metric
- Measures statistical naturalness
- **Useful** for real-world images without GT

---

## Recommendations

### For Your Project:

1. **Keep PSNR and SSIM** - they're standard and useful for comparison
2. **Always visually inspect** - metrics don't tell the whole story
3. **Look at both metrics together** - they complement each other
4. **Consider adding:**
   - **CIEDE2000** if color accuracy is important
   - **No-reference metrics** (BRISQUE/NIQE) for real-world evaluation

### When Metrics Disagree with Visual Quality:

- **Metrics say "good" but looks bad:**
  - Check for artifacts (halos, over-enhancement)
  - Look for color issues (cast, saturation)
  - Metrics may miss these!

- **Metrics say "bad" but looks good:**
  - Check for brightness/color shifts (SSIM handles this better)
  - Minor misalignment can hurt PSNR
  - Visual quality is what matters for applications!

---

## Summary

| Metric | Measures | Good For | Limitations |
|--------|----------|----------|-------------|
| **PSNR** | Pixel-level accuracy | Noise/error magnitude | Not perceptual, sensitive to shifts |
| **SSIM** | Perceptual similarity | Structure, edges, textures | May miss color issues |

**Bottom line:** PSNR and SSIM are useful for objective comparison, but **always validate with visual inspection**. They're standard in research, but real-world quality assessment requires human judgment.

---

## Analysis of Your Current Results

Based on your `metrics.csv`:

### Your Metrics Range:
- **PSNR:** 7.67 - 18.14 dB (average ~12-13 dB)
- **SSIM:** 0.17 - 0.60 (average ~0.35-0.40)

### Interpretation:

**These values are relatively low**, which suggests:

1. **Significant differences from ground truth:**
   - PSNR < 18 dB indicates substantial pixel-level differences
   - SSIM < 0.60 indicates structural differences

2. **Possible reasons:**
   - ✅ **Normal for dehazing** - Dehazing is inherently difficult; perfect reconstruction is rare
   - ✅ **Algorithm limitations** - Dark Channel Prior is a classic method (2009), newer methods may perform better
   - ✅ **Parameter tuning needed** - Your current settings might not be optimal for all images
   - ⚠️ **Color/brightness shifts** - Post-processing (gamma, exposure) may cause metric drops even if visually good
   - ⚠️ **Ground truth quality** - If GT images have artifacts or aren't perfect, metrics will be lower

3. **What to focus on:**
   - **Visual quality matters more** - If images look good to you, metrics are secondary
   - **Relative comparison** - Use metrics to compare different parameter settings
   - **Best performers** - Images with PSNR > 15 dB and SSIM > 0.45 are your better results

### Your Best Results (from metrics.csv):
- **Best PSNR:** Image 22 (18.14 dB)
- **Best SSIM:** Image 55 (0.60), Image 32 (0.58), Image 41 (0.57)

### Recommendations:

1. **Visual inspection is key** - Check if these "best" images actually look best
2. **Parameter tuning** - Try different settings to improve metrics
3. **Compare with hazy input** - Metrics vs. GT may be low, but improvement over hazy image is what matters
4. **Consider no-reference metrics** - BRISQUE/NIQE can evaluate quality without GT

### Is This Normal?

**Yes, for Dark Channel Prior:**
- Classic methods typically achieve:
  - PSNR: 15-22 dB
  - SSIM: 0.60-0.80
- Your results are on the lower end but within expected range
- Modern deep learning methods often achieve higher metrics, but require training data

**Bottom line:** Your metrics indicate room for improvement, but **visual quality assessment is more important** than these numbers alone.

