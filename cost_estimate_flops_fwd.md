## 1. Why does it exist?

When Context Parallelism or Pipeline Parallelism executes in tandem with Splash Attention, XLA's async scheduler needs accurate cost models (estimated FLOPs and latency) for each attention operation to decide how to interleave background communication collectives with foreground Matrix Multiply Unit (MXU) compute.

If XLA's internal analytical cost model over- or under-estimates the true execution duration of the forward Splash Attention kernel, it may schedule communication too early (causing buffer bloat) or too late (failing to hide communication latency).

`cost_estimate_flops_fwd` provides a manual FLOP cost estimate override for the forward Splash Attention kernel.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `-1` (default) | Uses Splash Attention's internal analytical FLOP cost model. |
| Float $\ge 0$ | Explicit FLOP cost override used by XLA scheduling heuristics. |

Default in `base.yml`:
```yaml
cost_estimate_flops_fwd: -1 # -1 means using splash default cost estmiation, any >= 0 value will be used as cost estmiation for splash to overlap for communication (forward)
```

---

## 3. Companion Parameter

- **`cost_estimate_flops_bwd`**: FLOP cost estimate override for the backward pass.

---

### One-line intuition

> **`cost_estimate_flops_fwd` allows manually overriding the forward Splash Attention FLOP cost estimate to tune XLA compiler communication-compute overlap scheduling.**
