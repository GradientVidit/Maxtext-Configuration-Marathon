
## 1. Context: the two loops in pipelined MaxText

```text
outer loop: pipeline iterations (microbatches)
    inner loop: layers per stage (the N layers each stage processes sequentially)
```

`scan_layers_per_stage` controls whether `jax.lax.scan` is applied to the **inner loop** — iterating over the `num_layers_per_pipeline_stage` layers that each stage runs through sequentially.

---

## 2. The default is false — why?

```yaml
scan_layers_per_stage: false  # default
```

The outer loop is already scanned (`scan_pipeline_iterations=true`). Scanning both loops simultaneously, combined with attaching the remat policy to both (`set_remat_policy_on_pipeline_iterations` and `set_remat_policy_on_layers_per_stage`), causes **extra unintended rematerialization** — activations are checkpointed/recomputed at both scan boundaries rather than just the outer one.

With the outer scan covering microbatches and the inner loop unrolled, the compile graph size is bounded by `num_layers_per_pipeline_stage` unrolled iterations — which is typically small (1–4).

---

## 3. When to consider enabling

When `num_layers_per_pipeline_stage` is very large (e.g. 8+):
- Unrolled inner loop creates large HLO (N copies of each layer op)
- Compilation becomes slow or memory-intensive
- Scanning the inner loop reduces HLO size at the cost of inner-loop dynamism

MaxText maintainers explicitly note this as a scenario worth trying — with the caveat that the scan/remat recommendations may need to be swapped (scan+remat on inner, not outer) to avoid double-rematerialization.

---

## 4. Options

| Value | Behavior |
|---|---|
| `false` | Default — unroll the inner layers-per-stage loop |
| `true` | `jax.lax.scan` over layers per stage (inner loop) |

Default: `false`.

---

## 5. The remat pairing

```yaml
scan_layers_per_stage: true
set_remat_policy_on_layers_per_stage: true
set_remat_policy_on_pipeline_iterations: false  # flip this when using inner scan+remat
scan_pipeline_iterations: true  # can keep outer scan
```

The key: attach `set_remat_policy` to the scan you actually want to remat from. Don't remat both scans simultaneously.

---

## 6. Full scan matrix reference

| Setting | Default | Recommended flip? |
|---|---|---|
| `scan_pipeline_iterations` | `true` | Keep `true` unless debugging |
| `scan_pipeline_repeats` | `false` | `true` if repeats count is large |
| `scan_layers_per_stage` | `false` | `true` if `num_layers_per_pipeline_stage` is large |
| `set_remat_policy_on_pipeline_iterations` | `true` | Flip to `false` if using `scan_layers_per_stage+remat` |
| `set_remat_policy_on_layers_per_stage` | `false` | Flip to `true` if scanning inner loop with remat |

---

### One-line intuition

> **`scan_layers_per_stage` applies `jax.lax.scan` to the inner per-stage layer loop — off by default because unrolling is fine for small `num_layers_per_pipeline_stage`, and scanning both loops with remat on both causes double-rematerialization; worth enabling only when the inner loop is large.**
