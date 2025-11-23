### 1. Improved Atmospheric Light Selection
- **Changed:** Uses `max(channel)` brightness metric instead of `sum(channel)`
- **Why:** Prevents selecting colored bright objects that introduce color cast
- **Code:** `_estimate_atmospheric_light()` now uses `candidate_pixels.max(axis=1)`
- **Safety:** Atmospheric light clamped to minimum `1e-6` to avoid divide-by-zero

### 2. Per-Channel Guided Refinement
- **Added:** `_refine_transmission_perchannel()` method
- **Why:** Preserves chromatic edges better than grayscale guidance
- **How:** Guided filter applied independently on each BGR channel and averaged
- **Default:** Enabled (`use_perchannel_refinement=True`)
- **Trade-off:** Better edge preservation but may add slight color noise

### 3. Gray-World White Balance
- **Added:** `_gray_world_balance()` method
- **Why:** Removes residual color cast after radiance recovery
- **How:** Assumes mean color is neutral gray and rescales channels
- **Default:** Enabled (`apply_gray_world=True`)
- **Applied:** After radiance recovery and before post-processing

### 4. Improved Default Parameters
- `omega`: 0.95 (was 0.88)
- `t0`: 0.06 (was 0.1)
- `guided_radius`: 12 (was 40)

### 5. Robust Numerics
- Safe normalization using epsilon in transmission estimation
- Consistent float handling across dtypes
- Guided filter kernel sized using `(2*radius+1, 2*radius+1)`

## Expected Improvements
1. Reduced color cast from improved atmospheric light and Gray-World balancing  
2. Sharper edges due to per-channel guided refinement  
3. Less blur from smaller guided filter radius  
4. Brighter results from lower `t0` enabling stronger haze removal  

## Configuration

### In Code (`dark_channel_prior.py`)
```python
@dataclass
class DarkChannelPriorDehazer:
    omega: float = 0.95
    transmission_floor: float = 0.06
    guided_radius: int = 12
    use_perchannel_refinement: bool = True
    apply_gray_world: bool = True
