## 1. Why does `global_rope_proportion` exist?

In standard RoPE, 100% of the query and key head dimension ($d_{head}$) is rotated. However, research in partial RoPE (and architectures like Gemma 4, GLM, and DeepSeek) demonstrated that rotating only a fraction $\alpha \in (0, 1]$ of the head dimensions achieves full positional awareness while preserving unrotated feature channels for translation-invariant semantic matching:

```text
Head Dimension d_head (e.g. 128):
┌──────────────────────────────────────┬──────────────────────────────────────┐
│  Rotated Dimensions: α * d_head       │  Unrotated Dimensions: (1-α) * d_head │
│  (Contains RoPE Positional Encoding) │  (Standard unrotated Q, K features)  │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

`global_rope_proportion` defines the fraction $\alpha_{global}$ of head dimensions that undergo rotary transformation in **global attention layers**.

---

## 2. What it actually controls

```yaml
global_rope_proportion: 0.25
```

- Specifies the proportion of the rotary dimension allocated to global attention.
- When `0.25`: For a head dimension of $128$, the first $128 \times 0.25 = 32$ dimensions ($16$ 2D rotation pairs) receive RoPE, while the remaining $96$ dimensions remain unrotated.
- When `1.0`: 100% of head dimensions undergo rotary encoding (standard Llama/Mistral style).

$$\text{Rotary Dimension } d_{rotary} = \lfloor d_{head} \times \text{global\_rope\_proportion} \rfloor$$

---

## 3. Options and Defaults

| Value | Rotated Head Dimension | Common Usage |
|---|---|---|
| `0.25` (default) | 25% of $d_{head}$ rotated, 75% unrotated | Gemma 4 / partial-RoPE global attention |
| `0.5` | 50% rotated | Partial RoPE models (e.g. ChatGLM) |
| `1.0` | 100% rotated (Full RoPE) | Standard Llama 1/2/3, Mistral, Gemma 1/2 |

---

## 4. Interactions and Constraints

- **`head_dim`**: $d_{head} \times \text{global\_rope\_proportion}$ must evaluate to an even integer because RoPE operates on 2D coordinate pairs.
- **`local_rope_proportion`**: Controls the corresponding proportion for local/sliding-window attention layers.

---

## 5. Practical Scenarios

- **Llama 2/3, Mistral**: Set to `1.0` (or model config presets will automatically configure it).
- **Gemma 4**: Uses `0.25` for global attention layers to balance positional encoding and content representation.

---

### One-line intuition

> **`global_rope_proportion` sets the fraction of head dimension rotated by RoPE in global attention layers, leaving the remainder unrotated for translation-invariant features.**
