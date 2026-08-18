
## 1. The bias update mechanism

`routed_bias_update_rate` is the step size (learning rate) for the DeepSeek-V3 routing bias feedback update. It controls how aggressively the per-expert routing biases are adjusted each training step to correct load imbalance.

Only relevant when `routed_bias=True`.

---

## 2. The update rule

```text
After each training step, for each expert i:
    actual_load_i = fraction of tokens routed to expert i
    target_load   = 1 / num_experts  (uniform)
    
    if actual_load_i > target_load:
        bias_i -= routed_bias_update_rate  (reduce appeal of overloaded expert)
    else:
        bias_i += routed_bias_update_rate  (increase appeal of underused expert)
```

`routed_bias_update_rate` is the fixed step size for each of these adjustments.

---

## 3. Choosing the right value

**Too large (e.g. 0.1):**
```text
bias overshoots → overloaded expert becomes underloaded in next step
→ oscillation: experts take turns being overloaded
→ routing never stabilizes
```

**Too small (e.g. 1e-6):**
```text
bias responds too slowly → expert collapse persists for many steps
→ load imbalance not corrected in time
```

**Good range:** DeepSeek-V3 paper used γ=0.001 for the main training phase (14.3T tokens), then disabled it (set to 0) for the final cooldown phase (500B tokens). This is a reasonable starting point.

---

## 4. Default

```yaml
routed_bias_update_rate: 0.0
```

Zero — bias update disabled. Correct default since `routed_bias: false` means there's no bias to update.

---

## 5. Must be paired with `routed_bias=True`

```yaml
routed_bias: true
routed_bias_update_rate: 0.001  # DeepSeek-V3 paper value
```

With `routed_bias=False`, `routed_bias_update_rate` has no effect.

---

## 6. Interaction with `load_balance_loss_weight`

Don't set both to non-zero at the same time. Two load-balancing mechanisms will interfere.

| `routed_bias` | `load_balance_loss_weight` | Recommended |
|---|---|---|
| `false` | `0.0` | Dense or no balancing |
| `false` | `> 0.0` | Aux loss balancing (Mixtral style) |
| `true` | `0.0` | Bias balancing (DeepSeek-V3 style) |
| `true` | `> 0.0` | Avoid — two conflicting mechanisms |

---

### One-line intuition

> **`routed_bias_update_rate` is the step size for the DeepSeek-V3 routing bias feedback controller — too large causes oscillation, too small allows collapse; `0.001` is the paper's value.**
