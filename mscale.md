## 1. Why does `mscale` exist?

When context length is extended from $L_{orig}$ to $s \times L_{orig}$, self-attention averages over $s$ times more key-value tokens:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

As sequence length increases, the distribution of dot-product attention logits widens, causing the softmax distribution to become either overly sharp (concentrated on a few tokens) or overly diffuse (entropy explosion), which degrades model perplexity.

YaRN introduces an empirical **magnitude scale multiplier** ($m$-scale) to rescale attention logits:

$$m = 0.1 \ln(s) + 1.0$$

```text
Extended Context (s = 40):
Raw Attention Logits ──> [ Scale by mscale ] ──> Softmax with Calibrated Attention Entropy
```

`mscale` sets or tunes this attention temperature multiplier.

---

## 2. What it actually controls

```yaml
mscale: 1.0
```

- When `1.0` (default): MaxText either applies the standard baseline magnitude multiplier or derives temperature scaling dynamically from context expansion ratios.
- When explicitly configured (e.g. `1.15`, `1.2`): Overrides the attention logit magnitude scaling factor applied inside rotary attention layers under `rope_type: "yarn"`.

---

## 3. Options and Defaults

| Value | Behavior | Context Scaling Factor ($s$) |
|---|---|---|
| `1.0` (default) | Baseline scale | $s = 1$ (No extension) or auto-scaled |
| `1.07` | Calibrated $m$-scale | $s = 8$ (32k context from 4k) |
| `1.14` | Calibrated $m$-scale | $s = 16$ (64k context from 4k) |
| `1.20` – `1.30` | Calibrated $m$-scale | $s = 40$ (160k context from 4k) |

---

## 4. Interactions and Dependencies

- **`rope_type: "yarn"`**: `mscale` is active during YaRN attention computation.
- **`rope_attention_scaling`**: Toggle that governs whether attention output is scaled in addition to rotary vectors.

---

## 5. Practical Scenarios

- **YaRN Fine-Tuning**: Set `mscale` according to the YaRN formula $1.0 + 0.1 \ln(s)$ for the chosen context extension multiplier $s$.

---

### One-line intuition

> **`mscale` scales attention logit magnitudes to counteract entropy explosion and maintain sharp attention distributions at long context lengths.**
