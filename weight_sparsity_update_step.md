
## 1. Why does it exist?

Once N:M sparsity is active, the **sparsity mask** (which weights are zeroed) isn't fixed for all time. As training progresses, gradients shift — a weight that was near-zero at step 1000 might become important at step 5000, and a currently-large weight might shrink. Periodically recomputing the mask lets the model keep the most important weights active.

`weight_sparsity_update_step` controls how often (in training steps) the sparsity mask is recomputed.

---

## 2. The update process

```text
training step K (where K >= weight_sparsity_start_step):

  if K mod weight_sparsity_update_step == 0:
    for each M-block of weights:
      compute |weight| for each element
      keep top-N by magnitude, zero the rest
      update mask

  apply current mask to weights (every step)
  proceed with forward/backward
```

Between mask updates, the mask is frozen. The gradient flows normally through non-zero weights; zero-masked weights don't contribute.

---

## 3. Options

| Value | Behavior |
|---|---|
| Integer > 0 | Mask recomputed every N steps |
| `10` | Default — mask updated every 10 steps |

Default in base.yml:
```yaml
weight_sparsity_update_step: 10
```

---

## 4. Trade-off

```text
Small update_step (e.g., 1):
  Mask changes every step
  → More adaptive to training dynamics
  → More compute overhead (mask recomputation is cheap but not free)
  → Potentially unstable: weights keep getting zeroed as they're learning

Large update_step (e.g., 1000):
  Mask mostly fixed
  → Less adaptive (model may be stuck with suboptimal sparsity pattern)
  → Closer to static pruning

Default 10:
  Moderate — adapts reasonably without excessive mask churn
```

---

## 5. Dependency

Only relevant when sparsity is active:
```yaml
weight_sparsity_n: 2
weight_sparsity_m: 4
```

With `weight_sparsity_n: null`, this parameter has no effect.

---

## 6. Interaction with `weight_sparsity_start_step`

```text
steps 0 → start_step:
  train densely (no mask)

step start_step:
  initial mask computed, sparsity applied

every update_step steps after start_step:
  mask recomputed
```

`start_step` delays when sparsity kicks in; `update_step` controls how often the mask refreshes thereafter.

---

### One-line intuition

> **`weight_sparsity_update_step=10` refreshes which N weights survive in each M-block every 10 training steps — frequent enough to track training dynamics but infrequent enough to avoid constant mask churn.**
