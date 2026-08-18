
## 1. Mirror of `set_remat_policy_on_pipeline_iterations` for the inner loop

`set_remat_policy_on_layers_per_stage` does exactly what its counterpart does for the outer loop, but applied to the **inner loop**: iterating over the `num_layers_per_pipeline_stage` layers that each pipeline stage processes sequentially.

```text
Outer scan (pipeline iterations):
    set_remat_policy_on_pipeline_iterations → controls remat here

  Inner scan/unroll (layers per stage):
      set_remat_policy_on_layers_per_stage → controls remat here
```

---

## 2. Default: false — why?

```yaml
set_remat_policy_on_layers_per_stage: false  # default
```

The inner loop is typically **unrolled** (`scan_layers_per_stage=false`), and remat policies only attach to scan loops — not to unrolled code. So with the default configuration, this setting has no mechanical effect even if set to `true`.

Furthermore: enabling remat on the inner loop in addition to the outer loop causes **double-rematerialization** — activations are recomputed at both scan boundaries, which is more expensive than intended.

---

## 3. When to enable

Only meaningful when:
1. `scan_layers_per_stage=true` (inner loop is actually scanned)
2. You want remat at the inner scan boundary
3. You've turned off `set_remat_policy_on_pipeline_iterations=false` to avoid double-remat

This is the "reverse" configuration the maintainers mention for large `num_layers_per_pipeline_stage`:

```yaml
# Alternative for large num_layers_per_pipeline_stage:
scan_pipeline_iterations: true
scan_layers_per_stage: true
set_remat_policy_on_pipeline_iterations: false
set_remat_policy_on_layers_per_stage: true
```

---

## 4. Options

| Value | Behavior |
|---|---|
| `false` | Default — no remat policy on inner layers-per-stage loop |
| `true` | Apply `remat_policy` to the inner layers-per-stage scan |

Default: `false`.

---

## 5. The double-remat anti-pattern

```yaml
# AVOID:
set_remat_policy_on_pipeline_iterations: true   # remat outer
set_remat_policy_on_layers_per_stage: true      # remat inner too → DOUBLE REMAT
```

This causes more recomputation than expected during the backward pass.

---

## 6. Relationship to `remat_policy`

Like its outer-loop counterpart, this is the **where** — `remat_policy` remains the **what**:

```yaml
remat_policy: 'save_dot_except_mlp'
set_remat_policy_on_layers_per_stage: true
scan_layers_per_stage: true
```

---

### One-line intuition

> **`set_remat_policy_on_layers_per_stage` applies the remat policy to the inner per-stage layer scan — off by default because the outer iteration scan handles remat in the standard config, and enabling both simultaneously causes unintended double-rematerialization.**
