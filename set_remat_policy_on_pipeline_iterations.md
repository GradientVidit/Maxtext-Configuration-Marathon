
## 1. Why remat needs a scan to attach to

Rematerialization (activation checkpointing) in JAX works by annotating a computation to discard intermediate activations and recompute them during the backward pass. In the context of `jax.lax.scan`, the remat policy can be applied **at the scan boundary** — meaning activations within each scan iteration are discarded, and only the checkpoint at the end of each iteration is kept.

Without a scan, there's no uniform boundary to attach a remat policy to — you'd need to annotate individual ops. `set_remat_policy_on_pipeline_iterations` says: apply the `remat_policy` to each iteration of the outer pipeline scan.

---

## 2. What this looks like in practice

```text
Outer scan (pipeline iterations, one per microbatch):
  Iteration 0: forward pass for microbatch 0
  Iteration 1: forward pass for microbatch 1
  ...

With set_remat_policy_on_pipeline_iterations=true:
  → activations within each iteration are subject to remat_policy
  → only the per-iteration boundary checkpoint survives
  → backward pass recomputes from iteration boundaries
```

The `remat_policy` setting (e.g. `'full'`, `'minimal'`, etc.) determines which tensors within each iteration to keep or recompute.

---

## 3. Default and recommendation

```yaml
scan_pipeline_iterations: true                 # outer scan enabled
set_remat_policy_on_pipeline_iterations: true  # attach remat to it
```

Both default to `true` and are recommended to stay together. This is the primary location where the remat policy has effect in pipelined MaxText.

---

## 4. The "extra remat" trap

Applying remat policy to **both** the outer and inner scans causes intermediate activations to be discarded at both boundaries — resulting in more recomputation than intended:

```yaml
# AVOID this combination:
set_remat_policy_on_pipeline_iterations: true   ← remat outer
set_remat_policy_on_layers_per_stage: true      ← remat inner too
```

This means activations are checkpointed at the layers-per-stage boundary AND at the pipeline-iteration boundary, so the backward pass recomputes more than a user expecting only one level of checkpointing would anticipate.

---

## 5. Options

| Value | Behavior |
|---|---|
| `true` | Apply `remat_policy` to the outer pipeline-iteration scan — default |
| `false` | No remat policy on outer scan; attach it elsewhere or use no per-scan remat |

Default: `true`.

---

## 6. Requires scan to be active

If `scan_pipeline_iterations=false`, the outer loop is unrolled and there's no scan to attach the policy to. In that case, this setting has no effect.

---

## 7. Interaction with `remat_policy`

`set_remat_policy_on_pipeline_iterations=true` is the **where** (which loop gets rematerialized).
`remat_policy` is the **what** (which tensors within that loop to discard vs. keep).

Both must be configured together:

```yaml
remat_policy: 'full'
set_remat_policy_on_pipeline_iterations: true
```

---

### One-line intuition

> **`set_remat_policy_on_pipeline_iterations` tells MaxText to apply the activation checkpointing (`remat_policy`) at the outer microbatch-scan boundary — this is where the memory savings actually happen in pipelined training, and it should almost always be kept `true` alongside `scan_pipeline_iterations`.**
