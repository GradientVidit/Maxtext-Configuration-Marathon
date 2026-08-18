## 1. Why does `rope_min_timescale` exist?

In Rotary Position Embeddings, the angular velocity $\theta_i$ for each dimension pair $i \in [0, d/2-1]$ is determined by a geometric progression of timescales (wavelengths):

$$\theta_i = \frac{1}{\text{timescale}_i}, \quad \text{timescale}_i = \text{min\_timescale} \cdot \left(\frac{\text{max\_timescale}}{\text{min\_timescale}}\right)^{\frac{2i}{d}}$$

```text
Dimension Index i:   0 ───────> 1 ───────> ... ───────> d/2 - 1
Timescale:           min_timescale                    max_timescale
Angular Velocity θ:  Fastest Rotation                 Slowest Rotation
Wavelength:          Shortest (2π)                    Longest (2π × max_timescale)
```

The fastest-rotating dimension ($i = 0$) corresponds to the minimum timescale. `rope_min_timescale` establishes this lower frequency anchor.

---

## 2. What it actually controls

```yaml
rope_min_timescale: 1
```

- Defines the base timescale for the highest-frequency rotary dimension ($i=0$).
- When set to `1` (standard), the highest-frequency rotary component has an angular frequency of $\theta_0 = 1.0$ radians per token position, completing a full $2\pi$ cycle every $\approx 6.28$ tokens.

---

## 3. Options and Defaults

| Value | Meaning | Usage |
|---|---|---|
| `1` (default) | Standard highest-frequency base timescale ($\theta_0 = 1.0$) | Universal standard across Llama, Gemma, Mistral, GPT-NeoX |
| `> 1` | Slower maximum rotation speed across all dimensions | Rare experimental scaling |

---

## 4. Parameter Interactions

- **`rope_max_timescale`**: Forms the upper bound of the timescale range. The geometric progression spans from `rope_min_timescale` to `rope_max_timescale`.
- **Head Dimension (`head_dim`)**: The number of frequency bands is $d/2 = \text{head\_dim} / 2$. These bands are exponentially spaced between `rope_min_timescale` and `rope_max_timescale`.

---

## 5. Practical Scenarios

- **Almost never modified**: In virtually all standard LLMs (Llama 1/2/3, Mistral, Gemma 1/2, Qwen), `rope_min_timescale` is fixed at `1`. Keep default `1`.

---

### One-line intuition

> **`rope_min_timescale` sets the minimum timescale (fastest rotational frequency) for the first dimension pair in the RoPE geometric progression.**
