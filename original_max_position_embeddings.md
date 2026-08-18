## 1. Why does `original_max_position_embeddings` exist?

When scaling context length using YaRN or NTK-aware interpolation, the algorithm must know the **original context window** $L_{orig}$ that the model saw during initial pretraining.

The ratio between the target extended context $L_{target}$ and the original context $L_{orig}$ determines:
1. Which rotary frequency bands need interpolation vs extrapolation.
2. The attention entropy temperature scaling factor $m$.

```text
Pretrained Base Model (L_orig = 4096) ──> YaRN Scaling (s = L_target / L_orig) ──> Extended Model (L_target = 163840)
```

`original_max_position_embeddings` records this baseline pretraining context length $L_{orig}$.

---

## 2. What it actually controls

```yaml
original_max_position_embeddings: 4096
```

Inside the YaRN mathematical formulation:
- $L_{orig} =     ext{original\_max\_position\_embeddings}$
- Scale factor $s = \frac{    ext{max\_position\_embeddings}}{L_{orig}}$

```text
Wavelength Transition in YaRN:
Wavelength $\lambda_i = 2\pi \cdot b^{2i/d}$
If $\lambda_i < \frac{L_{orig}}{\beta_{fast}}$ ──> Extrapolate (scale = 1.0)
If $\lambda_i > \frac{L_{orig}}{\beta_{slow}}$ ──> Interpolate (scale = s)
```

---

## 3. Options and Common Values

| Base Pretrained Model | `original_max_position_embeddings` | Typical Extended Target |
|---|---|---|
| Llama 2 / Mistral / DeepSeek V1 | `4096` (default) | 32k, 64k, 128k, 160k |
| Llama 1 | `2048` | 16k, 32k |
| Llama 3 Base | `8192` | 128k |

---

## 4. Interactions and Constraints

- **Only used when `rope_type: "yarn"`**: Standard RoPE ignores this parameter.
- **Must match actual checkpoint pretraining length**: If you set `original_max_position_embeddings: 8192` on a checkpoint that was pretrained at `4096`, YaRN will miscalculate the frequency boundaries, leading to degraded perplexity on short and long contexts alike.

---

## 5. Practical Scenarios

- **Extending 4k checkpoints**: Set `original_max_position_embeddings: 4096`.
- **Extending 8k checkpoints**: Set `original_max_position_embeddings: 8192` and adjust `rope_factor` accordingly.

---

### One-line intuition

> **`original_max_position_embeddings` specifies the baseline pretraining context window used by YaRN to calculate context expansion ratios.**
