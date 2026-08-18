## 1. Why does it exist?

During the backward pass, Splash Attention computes gradients while overlapping backward communication collectives (e.g. Ring Attention KV shifts or FSDP reduce-scatters).

`cost_estimate_flops_bwd` provides a manual FLOP cost estimate override for the backward Splash Attention kernel to assist XLA in scheduling backward collective overlap.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `-1` (default) | Uses default internal analytical backward FLOP cost model. |
| Float $\ge 0$ | Explicit backward FLOP cost override. |

Default in `base.yml`:
```yaml
cost_estimate_flops_bwd: -1 # -1 means using splash default cost estmiation, any >= 0 value will be used as cost estmiation for splash to overlap for communication (backward)
```

---

### One-line intuition

> **`cost_estimate_flops_bwd` allows manually overriding the backward Splash Attention FLOP cost estimate for fine-tuning backward communication-compute pipelining in XLA.**
