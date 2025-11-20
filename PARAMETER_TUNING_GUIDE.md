# Parameter Tuning Guide for Dark Channel Prior Dehazing

This guide explains which parameters to adjust to fix **lighting issues** and **blur** in dehazed outputs.  
All appearance-related knobs live directly inside `dark_channel_prior.py` (see the `GAMMA`, `EXPOSURE_GAIN`, and `APPLY_SHARPEN` constants and the `DarkChannelPriorDehazer` defaults).  
Edit those values inside the Python file to change global behavior—no CLI flags are needed.

## Quick Fixes for Common Issues

### 🔆 **LOW LIGHTING / DARK OUTPUTS**

**Primary parameters to adjust:**

1. **`--exposure` (Exposure Gain)**
   - **Default:** 1.0 (no adjustment)
   - **To brighten:** Use 1.2 to 1.5 (e.g., `--exposure 1.3`)
   - **Effect:** Multiplies pixel values to increase overall brightness
   - **Recommended starting value:** 1.2-1.3

2. **`--gamma` (Gamma Correction)**
   - **Default:** 1.0 (no correction)
   - **To brighten:** Use 0.8 to 0.9 (e.g., `--gamma 0.85`)
   - **Effect:** Non-linear brightness adjustment (preserves highlights better)
   - **Recommended starting value:** 0.85-0.9

3. **`--t0` (Transmission Floor)**
   - **Default:** 0.1
   - **To brighten:** Lower to 0.05-0.08 (e.g., `--t0 0.06`)
   - **Effect:** Allows more aggressive haze removal in dense regions
   - **Warning:** Too low (<0.05) can cause noise amplification

4. **`--omega` (Haze Retention Factor)**
   - **Default:** 0.95
   - **To brighten:** Lower to 0.85-0.90 (e.g., `--omega 0.88`)
   - **Effect:** Removes more haze but may make distant objects darker
   - **Trade-off:** Better contrast vs. depth perception

**How to adjust:** open `dark_channel_prior.py` and set  
`EXPOSURE_GAIN = 1.3`, `GAMMA = 0.85`, or tweak `transmission_floor (t0)` and `omega` in `DarkChannelPriorDehazer`.

---

### 🔍 **BLUR / LOSS OF SHARPNESS**

**Primary parameters to adjust:**

1. **`--patch` (Patch Size)**
   - **Default:** 15
   - **To reduce blur:** Lower to 9-11 (e.g., `--patch 9`)
   - **Effect:** Smaller patches preserve fine details
   - **Trade-off:** More detail but potentially noisier

2. **`--guided_radius` (Guided Filter Radius)**
   - **Default:** 40
   - **To reduce blur:** Lower to 20-30 (e.g., `--guided_radius 25`)
   - **Effect:** Less smoothing preserves edges better
   - **Trade-off:** Sharper but may have more artifacts

3. **`--guided_eps` (Guided Filter Epsilon)**
   - **Default:** 0.001 (1e-3)
   - **To reduce blur:** Lower to 0.0005-0.0001 (e.g., `--guided_eps 0.0005`)
   - **Effect:** Preserves more detail in transmission refinement
   - **Warning:** Too low can cause artifacts

4. **`--sharpen` (Unsharp Masking)**
   - **Default:** Off
   - **To reduce blur:** Enable with `--sharpen`
   - **Effect:** Post-processing sharpening filter
   - **Use when:** Other parameters don't fully resolve blur

**How to adjust:** change `patch_size`, `guided_radius`, `guided_eps`, or set `APPLY_SHARPEN = True` at the top of `dark_channel_prior.py`.

---

## Combined Fixes (Lighting + Blur)

**Recommended starting point:**  
Edit `dark_channel_prior.py` to set `EXPOSURE_GAIN = 1.3`, `GAMMA = 0.85`, `APPLY_SHARPEN = True`, `patch_size = 9`, `guided_radius = 25`, `guided_eps = 5e-4`, and lower `transmission_floor` to `0.06`.

---

## Parameter Reference Table

| Parameter | Default | Range | Affects | Direction |
|-----------|---------|-------|---------|-----------|
| **Lighting Parameters** |
| `EXPOSURE_GAIN` | 1.0 | 0.8-2.0 | Brightness | ↑ = brighter |
| `GAMMA` | 1.0 | 0.5-1.5 | Brightness (non-linear) | ↓ = brighter |
| `--t0` | 0.1 | 0.05-0.2 | Haze removal aggressiveness | ↓ = brighter |
| `--omega` | 0.95 | 0.8-1.0 | Haze retention | ↓ = brighter |
| **Sharpness Parameters** |
| `patch_size` | 15 | 5-25 | Detail preservation | ↓ = sharper |
| `guided_radius` | 40 | 10-60 | Edge smoothing | ↓ = sharper |
| `guided_eps` | 0.001 | 0.0001-0.01 | Detail preservation | ↓ = sharper |
| `APPLY_SHARPEN` | False | True/False | Post-processing sharpness | True = sharper |
| **Other Parameters** |
| `--top_percent` | 0.001 | 0.0005-0.01 | Atmospheric light selection | Usually keep default |
| `--beta` | 1.0 | 0.5-2.0 | Depth map scaling | Only affects depth map |

---

## Tuning Strategy

1. **Start with defaults** and identify the main issue (lighting vs blur)

2. **For lighting issues:**
   - First try `--exposure 1.2` (quick fix)
   - If still dark, add `--gamma 0.85`
   - If needed, lower `--t0` to 0.06-0.08

3. **For blur issues:**
   - First try `--sharpen` (easiest)
   - Then lower `--patch` to 9-11
   - Finally adjust `--guided_radius` and `--guided_eps`

4. **Fine-tune iteratively:**
   - Adjust one parameter at a time
   - Test on a few sample images
   - Check for artifacts (noise, halos, over-saturation)

5. **Watch for over-correction:**
   - Too high exposure → washed out highlights
   - Too low patch → noise and artifacts
   - Too low guided_radius → transmission map artifacts

---

## Example: Full Command with All Optimizations

Edit `dark_channel_prior.py` to set:
- `EXPOSURE_GAIN = 1.3`
- `GAMMA = 0.85`
- `APPLY_SHARPEN = True`
- Inside `DarkChannelPriorDehazer(...)`: `patch_size = 9`, `guided_radius = 25`, `guided_eps = 5e-4`, `transmission_floor = 0.06`.

---

## Notes

- **Image-dependent:** Optimal parameters vary by image content (dense haze, light haze, sky regions, etc.)
- **Batch processing:** Use consistent parameters across a dataset for fair comparison
- **Ground truth:** If available, use PSNR/SSIM metrics to guide parameter selection
- **Visual inspection:** Always visually inspect results; metrics don't capture all quality aspects

