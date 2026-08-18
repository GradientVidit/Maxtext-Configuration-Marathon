## 1. Why does `local_rope_proportion` exist?

In hybrid Transformer architectures with mixed local and global attention, local sliding-window layers typically require different positional representation density than global layers. 

Because sliding-window attention operates over a bounded, fixed-size receptive field, rotating a higher proportion of head dimensions (or 100% of dimensions) provides maximal local positional discrimination without suffering from long-range extrapolation degradation.

`local_rope_proportion` defines the fraction $\alpha_{local}$ of head dimensions subjected to RoPE in **local (sliding window) attention layers**.

---

## 2. What it actually controls

```yaml
local_rope_proportion: 1.0
```

- Specifies the fraction of the head dimension rotated inside local/sliding-window attention layers.
- Default is `1.0` (100% of the head dimension is rotated in local layers).

```text
Local Attention Layer:
Head Dimension d_head ──> [ local_rope_proportion = 1.0 ] ──> 100% Rotated by RoPE

Global Attention Layer:
Head Dimension d_head ──> [ global_rope_proportion = 0.25 ] ──> 25% Rotated, 75% Unrotated
```

---

## 3. Options and Defaults

| Value | Behavior in Local Layers | Architecture Alignment |
|---|---|---|
| `1.0` (default) | 100% of head dimension rotated | Gemma 4, hybrid sliding-window architectures |
| `0.5` | 50% of head dimension rotated | Partial-RoPE local layers |
| `0.25` | 25% of head dimension rotated | Matches `global_rope_proportion` |

---

## 4. Interactions and Constraints

- **Even Number Constraint**: The computed rotary dimension $d_{rotary} = d_{head} \times \text{local\_rope\_proportion}$ must be an even integer.
- **`sliding_window_size`**: Applies when sliding window attention is active on the layer.

---

## 5. Practical Scenarios

- **Standard Models (Llama 3, Mistral)**: Both local and global proportions are effectively `1.0`.
- **Gemma 4 Architecture**: Configured with `local_rope_proportion: 1.0` and `global_rope_proportion: 0.25`.

---

### One-line intuition

> **`local_rope_proportion` sets the fraction of head dimension rotated by RoPE in local sliding-window attention layers.**
