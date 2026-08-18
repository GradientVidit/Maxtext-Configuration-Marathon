## 1. Why does `beta_fast` exist?

In YaRN (Yet another RoPE extensioN), rotary dimensions are partitioned into three frequency bands based on their wavelength $\lambda_i$:

1. **High Frequencies (Short Wavelengths)**: Critical for local token ordering. These should **not** be interpolated (extrapolation only, scale $= 1.0$).
2. **Low Frequencies (Long Wavelengths)**: Global position. These should be **fully interpolated** (scale $= s$).
3. **Mid Frequencies**: Blended smoothly between extrapolation and interpolation.

```text
YaRN Frequency Ramp Function:
Wavelength λ_i:    0 ────────────── λ_low ────────────── λ_high ────────────── ∞
Interpolation γ:        0.0 (Extrapolate)    Ramp (0 → 1)       1.0 (Interpolate)
                             ▲                               ▲
                             │                               │
                       beta_fast (32)                  beta_slow (1)
```

`beta_fast` defines the high-frequency cutoff boundary $\beta_{fast}$, determining where the transition ramp begins.

---

## 2. What it actually controls

```yaml
beta_fast: 32
```

In YaRN:
$$\lambda_{\text{low}} = \frac{L_{\text{orig}}}{\beta_{\text{fast}}} = \frac{4096}{32} = 128$$

- For all rotary dimensions with wavelength $\lambda_i < \lambda_{\text{low}}$ (i.e. $\lambda_i < 128$ tokens), the interpolation weight $\gamma_i = 0$.
- These dimensions rotate with their original, uncompressed angular velocities, ensuring zero loss of high-resolution local attention.

---

## 3. Options and Defaults

| Value | Meaning | Transition Cutoff (at $L_{orig}=4096$) |
|---|---|---|
| `32` (default) | Standard YaRN fast cutoff | Wavelengths $< 128$ tokens remain unscaled |
| `16` | Wider unscaled band | Wavelengths $< 256$ tokens remain unscaled |
| `64` | Narrower unscaled band | Wavelengths $< 64$ tokens remain unscaled |

---

## 4. Interactions with `beta_slow`

- **Constraint**: `beta_fast > beta_slow` must always hold.
- Default pairing: `beta_fast: 32`, `beta_slow: 1`.
- The ramp interval spans $[\lambda_{low}, \lambda_{high}] = [\frac{L_{orig}}{\beta_{fast}}, \frac{L_{orig}}{\beta_{slow}}]$.

---

## 5. Practical Scenarios

- **YaRN Fine-Tuning**: Keep default `32` as established in the original YaRN paper (Peng et al., 2023) and DeepSeek models.

---

### One-line intuition

> **`beta_fast` sets the upper boundary for high-frequency rotary dimensions in YaRN, preserving full local resolution below this threshold.**
