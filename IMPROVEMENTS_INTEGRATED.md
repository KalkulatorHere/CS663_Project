# ChatGPT Improvements Integrated

This document summarizes the improvements integrated from ChatGPT's review and testing.

## ✅ Changes Integrated

### 1. **Improved Atmospheric Light Selection**
- **Changed:** Uses `max(channel)` brightness metric instead of `sum(channel)`
- **Why:** Prevents picking colored bright objects (e.g., red car, blue sign) which cause color cast
- **Code:** `_estimate_atmospheric_light()` now uses `candidate_pixels.max(axis=1)`
- **Safety:** Clamps atmospheric light to minimum 1e-6 to avoid divide-by-zero

### 2. **Per-Channel Guided Refinement**
- **Added:** `_refine_transmission_perchannel()` method
- **Why:** Color-guided refinement preserves chromatic edges better than grayscale guide
- **How:** Applies guided filter to each BGR channel separately and averages results
- **Default:** Enabled (`use_perchannel_refinement=True`)
- **Trade-off:** Better edge preservation but may introduce slight color noise

### 3. **Gray-World White Balance**
- **Added:** `_gray_world_balance()` method
- **Why:** Reduces residual color cast after radiance recovery
- **How:** Assumes average color should be neutral gray, scales channels accordingly
- **Default:** Enabled (`apply_gray_world=True`)
- **Applied:** After radiance recovery, before post-processing

### 4. **Improved Default Parameters**
- **omega:** `0.95` (was 0.88) - stronger haze removal
- **t0:** `0.06` (was 0.1) - more aggressive dehazing in dense regions
- **guided_radius:** `12` (was 40) - smaller window reduces blur and haloing

### 5. **Robust Numerics**
- **Already present:** Safe normalization with epsilon in transmission estimation
- **Already present:** Proper float handling for all dtypes
- **Already present:** Correct guided filter kernel sizing `(2*radius+1, 2*radius+1)`

## 📊 Expected Improvements

1. **Reduced Color Cast:** Better atmospheric light selection + Gray-World balance
2. **Sharper Edges:** Per-channel refinement preserves color boundaries
3. **Less Blur:** Smaller guided filter radius reduces over-smoothing
4. **Brighter Outputs:** Lower t0 allows more aggressive haze removal

## 🔧 Configuration

### In Code (dark_channel_prior.py)
```python
@dataclass
class DarkChannelPriorDehazer:
    omega: float = 0.95  # Improved default
    transmission_floor: float = 0.06  # Improved default
    guided_radius: int = 12  # Improved default
    use_perchannel_refinement: bool = True  # NEW: Enable per-channel refinement
    apply_gray_world: bool = True  # NEW: Enable Gray-World white balance
```

### Via CLI
```bash
# Use improved defaults (automatic)
python main.py --hazy_dir data/HazyImages --output_dir outputs

# Disable per-channel refinement (use grayscale guide)
python main.py --hazy_dir data/HazyImages --output_dir outputs --no-perchannel

# Disable Gray-World white balance
python main.py --hazy_dir data/HazyImages --output_dir outputs --no-grayworld
```

## ⚠️ Known Trade-offs

1. **Per-Channel Refinement:**
   - ✅ Better edge preservation
   - ⚠️ May introduce slight color noise/mottling
   - 💡 If artifacts appear, disable with `--no-perchannel`

2. **Lower t0 (0.06):**
   - ✅ More aggressive haze removal
   - ⚠️ May amplify noise in very dense haze regions
   - 💡 Increase to 0.08-0.1 if noise becomes problematic

3. **Smaller Guided Radius (12):**
   - ✅ Less blur, sharper transmission maps
   - ⚠️ May have more artifacts near edges
   - 💡 Increase to 20-30 if haloing appears

## 🧪 Testing Recommendations

1. **Visual Inspection:**
   - Check for color cast reduction (should be more neutral)
   - Verify edge sharpness (should be improved)
   - Look for any new artifacts (color noise, haloing)

2. **Metrics (if GT available):**
   - Compare PSNR/SSIM before/after improvements
   - Per-channel refinement may slightly reduce metrics but improve visual quality

3. **Parameter Tuning:**
   - If color noise appears: `--no-perchannel`
   - If too much noise: increase `--t0` to 0.08
   - If haloing: increase `--guided_radius` to 20-25

## 📝 Files Modified

- `dark_channel_prior.py`: Core improvements integrated
- `main.py`: Updated defaults and added CLI flags
- `IMPROVEMENTS_INTEGRATED.md`: This document

