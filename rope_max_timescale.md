## 1. Why does `rope_max_timescale` exist?

The maximum timescale in RoPE (commonly called **Base Theta** or $b$) defines the longest wavelength in the rotary embedding frequency spectrum:

$$\theta_i = b^{-\frac{2i}{d}} = \left(\text{rope\_max\_timescale}\right)^{-\frac{2i}{d}}$$

```text
Wavelength Progression across Head Dimension:
i = 0          ──> λ_0 = 2π × 1               ──> Fast oscillation (Token syntax)
i = d/4        ──> λ_{mid} = 2π × √b          ──> Medium range
i = d/2 - 1    ──> λ_{max} = 2π × b           ──> Slow oscillation (Global document position)
```

If `rope_max_timescale` ($b$) is too small (e.g. $10{,}000$), the longest rotary wavelength is $\approx 62{,}831$ tokens. Over long contexts, the lowest-frequency dimensions complete multiple full rotations, leading to phase collisions and position ambiguity.

Increasing `rope_max_timescale` (e.g., to $500{,}000$ in Llama 3 or $10{,}000{,}000$ in DeepSeek) stretches the maximum wavelength, preventing phase collisions across hundreds of thousands of tokens.

---

## 2. What it actually controls

```yaml
rope_max_timescale: 10_000
```

Controls the base frequency $b$ for global attention layers.

```text
Comparison of Base Theta Values:
rope_max_timescale: 10_000    ──> Llama 1/2, Mistral 7B (Pretrained at 4k/8k context)
rope_max_timescale: 500_000   ──> Llama 3/3.1 (Pretrained at 8k/128k context)
rope_max_timescale: 1_000_000 ──> Gemma 2, CodeLlama
```

---

## 3. Options and Common Values

| Model Architecture | Default / Standard Value | Context Target |
|---|---|---|
| Llama 1 / Llama 2 / Mistral 7B | `10_000` (1e4) | 2k – 8k |
| Llama 3 / 3.1 / 3.2 / 3.3 | `500_000` (5e5) | 8k – 128k |
| Gemma 2 (Global Attention) | `10_000` or `1_000_000` | 8k |
| Qwen 2 / DeepSeek V2/V3 | `1_000_000` – `10_000_000` | 32k – 128k |

---

## 4. Interactions and Overrides

- **`local_rope_max_timescale`**: For sliding-window or local attention layers, `local_rope_max_timescale` overrides `rope_max_timescale` if set positive.
- **`global_rope_max_timescale`**: In hybrid models (like Gemma 4), `global_rope_max_timescale` can provide a dedicated override for global attention layers.

---

## 5. Practical Scenarios

- **Pretraining standard model**: Set according to target context ($10{,}000$ for $\le 8k$, $500{,}000$ for $\ge 32k$).
- **Fine-tuning / Pretraining Llama 3**: Must be set to `500000.0`.
- **What breaks if wrong**: Loading a Llama 3 checkpoint with `rope_max_timescale: 10000` results in corrupted attention scores and nonsensical text output.

---

### One-line intuition

> **`rope_max_timescale` sets the base theta parameter ($b$), controlling the longest rotary wavelength to prevent positional phase collisions across long contexts.**
