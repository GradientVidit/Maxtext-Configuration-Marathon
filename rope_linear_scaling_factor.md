## 1. Why does `rope_linear_scaling_factor` exist?

When extending a pretrained Transformer to longer context windows, the model encounters token position indices $m > L_{train}$ that it never observed during pretraining. Directly evaluating at $m > L_{train}$ causes catastrophic attention degradation because rotary angles $\theta_i \cdot m$ venture into unobserved trigonometric regimes.

**Position Interpolation (PI)** (Chen et al., 2023) solves this by linearly compressing sequence position indices by a constant downscaling ratio $s$:

```text
Without Scaling (Extrapolation):
Position Index: 0, 1, 2, ..., 4096, 4097, ..., 16384  ──> (Angles exceed trained domain!)

With Linear Scaling (s = 4.0):
Position Index: 0, 1, 2, ..., 4096, 4097, ..., 16384
       │
       ▼ (Divide position by s)
Scaled Position: 0, 0.25, 0.5, ..., 1024, 1024.25, ..., 4096  ──> (Angles remain within [0, 4096])
```

`rope_linear_scaling_factor` defines this linear scale multiplier $s$.

---

## 2. What it actually controls

```yaml
rope_linear_scaling_factor: 1.0
```

Inside `RotaryEmbedding` (used when `rope_type: "default"`), the position index vector $m$ is divided by `rope_linear_scaling_factor`:

$$\tilde{m} = \frac{m}{\text{rope\_linear\_scaling\_factor}}$$

$$\text{Rotary Angle: } \theta_i \cdot \tilde{m} = \theta_i \cdot \frac{m}{s}$$

- When `1.0` (default): Standard uncompressed RoPE coordinates are used ($s = 1$).
- When `> 1.0` (e.g. `2.0`, `4.0`): The position coordinate space is compressed by factor $s$, allowing a model pretrained at length $L$ to process sequences up to $s \times L$ with brief fine-tuning.

---

## 3. Options and Defaults

| Value | Behavior | Effective Context Extension | Typical Use Case |
|---|---|---|---|
| `1.0` (default) | No scaling; standard 1:1 position coordinates | $1\times$ ($L_{train}$) | Standard pretraining and normal context inference |
| `2.0` | 2x position interpolation | $2\times$ (e.g. 4k $\to$ 8k) | 2x context fine-tuning |
| `4.0` | 4x position interpolation | $4\times$ (e.g. 4k $\to$ 16k) | 4x context fine-tuning |
| `8.0` | 8x position interpolation | $8\times$ (e.g. 4k $\to$ 32k) | 8x context fine-tuning |

---

## 4. Interactions and Limitations

- **Only active when `rope_type: "default"`**: If `rope_type: "llama3.1"` or `"yarn"`, `rope_linear_scaling_factor` is bypassed in favor of their respective non-linear frequency scaling mechanisms.
- **Fine-Tuning Requirement**: Setting `rope_linear_scaling_factor: 4.0` on a pretrained checkpoint without fine-tuning causes performance degradation because all high-frequency positional distinctions are compressed into 1/4th their trained angular velocity. A short fine-tuning stage (1,000–5,000 steps) is required to adapt.

---

## 5. Practical Scenarios

- **Pretraining**: Leave at `1.0`.
- **Fine-tuning a 4k Llama-2 model to 16k**: Set `rope_linear_scaling_factor: 4.0` and `max_target_length: 16384`.

---

### One-line intuition

> **`rope_linear_scaling_factor` divides sequence position coordinates by a constant factor to linearly compress long context sequences into the model's pretrained rotary angle range.**
