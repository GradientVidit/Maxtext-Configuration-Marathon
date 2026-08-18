
## 1. The two loops in pipelined MaxText

Pipeline parallelism in MaxText has a two-level loop structure:

```text
outer loop: pipeline iterations (microbatches moving through stages)
    inner loop: layers per stage (the layers each stage processes sequentially)
```

`scan_pipeline_iterations` controls whether `jax.lax.scan` is applied to the **outer loop**.

---

## 2. Why scan the outer loop at all?

Without scanning, JAX unrolls the microbatch loop, generating separate HLO operations for each microbatch. For N microbatches this means N copies of each op in the compiled program — larger compile time, larger HLO, harder for XLA to optimize uniformly.

With `jax.lax.scan`, the entire microbatch loop becomes a single compiled operation. This:
- Reduces compilation time dramatically
- Produces a more compact HLO graph
- Enables applying a single remat policy across all iterations (via `set_remat_policy_on_pipeline_iterations`)

---

## 3. The stacked checkpoint implication

When the outer loop is scanned, intermediate activations across microbatches are stacked into a single tensor rather than existing as separate named arrays. This changes the checkpoint structure. When combined with `set_remat_policy_on_pipeline_iterations=true`, the remat policy applies uniformly to all microbatch iterations.

---

## 4. Default and recommendation

```yaml
scan_pipeline_iterations: true  # default and recommended
```

The MaxText maintainers explicitly recommend this setting. The only reason to disable it: debugging scenarios where you want to inspect per-microbatch ops individually, or when a specific XLA optimization only triggers on unrolled code.

---

## 5. Options

| Value | Behavior |
|---|---|
| `true` | `jax.lax.scan` over pipeline iterations (outer loop) — default |
| `false` | Unroll the pipeline iteration loop |

Default: `true`.

---

## 6. Interaction with `set_remat_policy_on_pipeline_iterations`

These are designed to work together:

```yaml
scan_pipeline_iterations: true
set_remat_policy_on_pipeline_iterations: true  # also default
```

The remat policy can only be attached to the scan loop, not to unrolled iterations. If you set `scan_pipeline_iterations=false`, `set_remat_policy_on_pipeline_iterations` has no effect.

---

## 7. Interaction with the inner loop

The recommended configuration scans only the outer loop and does not scan the inner loop:

```yaml
scan_pipeline_iterations: true   ← scan this
scan_layers_per_stage: false     ← don't scan this
set_remat_policy_on_pipeline_iterations: true   ← remat on outer
set_remat_policy_on_layers_per_stage: false     ← not on inner
```

Applying scan+remat to both loops causes extra (unintended) rematerialization. The maintainers note the reverse configuration may be better when `num_layers_per_pipeline_stage` is very large.

---

### One-line intuition

> **`scan_pipeline_iterations` applies `jax.lax.scan` to the outer microbatch loop in pipeline parallelism — the recommended default that reduces compile time and enables uniform remat policy application across all pipeline iterations.**
