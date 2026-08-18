## 1. Why does it exist?

During the backward pass of Flash Attention, computing Query gradients ($dQ$) requires summing partial gradient contributions across all Key/Value sequence blocks:

$$dQ_i = \sum_j dS_{ij} \times K_j$$

In parallel kernel grids, partial $dQ$ sums can either be accumulated continuously across all KV steps or reduced in a multi-stage tree reduction pattern.

`dq_reduction_steps` controls the number of reduction steps used in the $dQ$ gradient reduction schedule.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `0` (default) | Reduces across all KV steps sequentially. |
| `3` | 3-stage reduction schedule for supported Pallas kernel grid layouts. |

Default in `base.yml`:
```yaml
dq_reduction_steps: 0 #the number of reduction steps. For now, only 3 or all the kv steps are supported.
```

---

### One-line intuition

> **`dq_reduction_steps` specifies the reduction schedule stages (default `0` = all KV steps) for accumulating Query gradients in the Splash Attention backward pass.**
