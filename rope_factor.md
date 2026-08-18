## 1. Why does `rope_factor` exist?

In YaRN and NTK-style positional scaling implementations, the context extension ratio $s$ is often explicitly parameterized as `rope_factor`:

$$s = \text{rope\_factor} \approx \frac{\text{max\_position\_embeddings}}{\text{original\_max\_position\_embeddings}}$$

```text
Context Length: 4,096 ──────( × rope_factor = 40 )──────> 163,840
```

`rope_factor` provides the direct scalar multiplier $s$ used across YaRN's wavelength boundary equations and attention temperature calculation.

---

## 2. What it actually controls

```yaml
rope_factor: 40
```

- When `rope_type: "yarn"`, `rope_factor` is the scaling constant $s$.
- It scales the low-frequency rotary dimensions by $1/s$ (compressing their angular velocity by $s$) while high frequencies remain untouched.

```text
YaRN Frequency Transformation:
Frequency $    heta_i' = (1 - \gamma_i) \cdot \frac{    heta_i}{s} + \gamma_i \cdot     heta_i$
where:
  s = rope_factor (40)
  γ_i is the interpolation weight computed from beta_fast and beta_slow
```

---

## 3. Options and Defaults

| `rope_factor` | Base Context | Resulting Target Context | Use Case |
|---|---|---|---|
| `40` (default) | 4k ($4096$) | 160k ($163{,}840$) | DeepSeek-style 160k YaRN extension |
| `8` | 4k ($4096$) | 32k ($32{,}768$) | 32k context fine-tuning |
| `16` | 4k ($4096$) | 64k ($65{,}536$) | 64k context fine-tuning |
| `32` | 4k ($4096$) | 128k ($131{,}072$) | 128k context fine-tuning |

---

## 4. Interactions and Consistency

- **Consistency Requirement**: `rope_factor` must be mathematically consistent with:
$$\text{rope\_factor} = \frac{\text{max\_position\_embeddings}}{\text{original\_max\_position\_embeddings}}$$
- In base.yml: $163{,}840 / 4{,}096 = 40$, which matches `rope_factor: 40`.

---

## 5. Practical Scenarios

- **Scaling a 4k model to 32k**: Set `rope_factor: 8`, `max_position_embeddings: 32768`, `original_max_position_embeddings: 4096`.
- **Scaling a 4k model to 128k**: Set `rope_factor: 32`, `max_position_embeddings: 131072`, `original_max_position_embeddings: 4096`.

---

### One-line intuition

> **`rope_factor` specifies the context expansion multiplier $s$ used by YaRN to scale rotary wavelengths and attention temperature.**
