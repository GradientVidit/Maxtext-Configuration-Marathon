
## 1. Why does `mlp_activations_limit` exist?

SwiGLU and other gated activations can produce unbounded outputs. While this isn't usually a problem in well-tuned runs, there are specific situations where activation values blow up:

```text
Large model scale
    + aggressive learning rate
    + poor initialization
    ↓
SiLU(x) outputs very large values
    ↓
MLP output explodes
    ↓
Residual stream overflows bf16 range
    ↓
Loss → NaN
```

`mlp_activations_limit` is a hard clip inserted after the activation function to cap these values:

```text
activation_output = clip(silu(x), -limit, +limit)
```

This prevents runaway activations from cascading into a NaN loss without having to reduce the learning rate across the board.

---

## 2. Default

```yaml
mlp_activations_limit: -1.0
```

`-1.0` means **no clipping** — the parameter is disabled. This is the right default: clipping activations in healthy training is unnecessary overhead and can slightly reduce expressiveness.

---

## 3. When to use it

| Situation | Recommendation |
|---|---|
| Stable training, healthy loss curve | Leave at `-1.0` |
| Early training instability (loss spikes in first 100 steps) | Try `100.0` or `50.0` |
| Large model with aggressive LR, occasional NaN | Try `200.0` as a safety net |
| Production run after hyperparams are tuned | Leave at `-1.0` |

Think of this as an emergency brake, not a tuning knob. If you need it constantly, the real fix is the learning rate, initialization, or model scale.

---

## 4. What it clips

The clip applies to **each MLP branch output** (after activation) before the element-wise multiplication:

```text
silu_output = silu(W_gate × x)              → values may be large
clamped     = clip(silu_output, -limit, limit)  ← mlp_activations_limit
linear_out  = W_up × x
gate        = clamped ⊙ linear_out
output      = W_down × gate
```

It does not clip the final MLP output, the pre-activation linear projections, or activations in attention.

---

## 5. Alternative approaches to activation explosions

If you're seeing instability that makes you want to set this:

1. **Gradient clipping** — `max_grad_norm` (a training loop param) is often more appropriate
2. **LR warmup** — longer warmup prevents early large steps from destabilizing
3. **Weight init scale** — `dense_init_scale` affects initial activation magnitudes
4. **Cast logits precision** — `cast_logits_to_fp32` and `float32_qk_product` address precision at specific choke points

---

### One-line intuition

> **`mlp_activations_limit` hard-clips MLP activation values to `[-limit, +limit]` as an emergency brake for training instability — leave at `-1.0` (disabled) in stable runs.**
