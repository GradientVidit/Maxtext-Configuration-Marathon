## 1. Why does `beta_slow` exist?

In YaRN's non-uniform frequency scaling, low-frequency rotary dimensions (long wavelengths) must be completely compressed by the context expansion factor $s$ to ensure that rotary angles stay within the domain observed during pretraining.

```text
YaRN Frequency Ramp Function:
Wavelength λ_i:    0 ────────────── λ_low ────────────── λ_high ────────────── ∞
Interpolation γ:        0.0 (No Scale)       Ramp (0 → 1)       1.0 (Full Scale s)
                                                             ▲
                                                             │
                                                       beta_slow = 1
```

`beta_slow` defines the low-frequency cutoff boundary $\beta_{slow}$, determining the wavelength beyond which dimensions undergo 100% position interpolation.

---

## 2. What it actually controls

```yaml
beta_slow: 1
```

In YaRN:
$$\lambda_{\text{high}} = \frac{L_{\text{orig}}}{\beta_{\text{slow}}} = \frac{4096}{1} = 4096$$

- For all rotary dimensions with wavelength $\lambda_i > \lambda_{\text{high}}$ (i.e. $\lambda_i > 4096$ tokens), the interpolation weight is $\gamma_i = 1.0$.
- These dimensions are fully scaled by $1/s$, guaranteeing that even at maximum extended context $L_{target}$, their rotary angles never exceed the original $2\pi$ phase space.

---

## 3. Options and Defaults

| Value | Meaning | Full Interpolation Threshold (at $L_{orig}=4096$) |
|---|---|---|
| `1` (default) | Standard YaRN slow cutoff | Wavelengths $> 4096$ tokens are 100% scaled |
| `2` | Earlier full scaling | Wavelengths $> 2048$ tokens are 100% scaled |

---

## 4. Interactions

- **`beta_fast`**: Works in tandem with `beta_fast: 32` to define the smooth interpolation ramp between $\lambda_{low}$ and $\lambda_{high}$.
- **`rope_truncate`**: Determines whether the ramp bounds are rounded via floor/ceil operations.

---

## 5. Practical Scenarios

- **Standard YaRN / DeepSeek Context Extension**: Keep `beta_slow: 1`.

---

### One-line intuition

> **`beta_slow` sets the boundary beyond which long-wavelength rotary dimensions are 100% interpolated in YaRN.**
