## 1. Why does `abort_on_inf_loss` exist?

Similar to `NaN`, floating-point arithmetic overflow (such as division by zero in normalization layers or logit overflow before softmax) produces `+Inf` or `-Inf` values in loss tensors:

```text
Step N:   Loss = 1.74
Step N+1: Loss = Inf ──(If unhandled)──> Produces NaN gradients on next backprop step!
                       │
                       ▼ (abort_on_inf_loss: true)
               IMMEDIATE JOB TERMINATION
```

An infinite loss value indicates complete numerical breakdown in the forward pass. Permitting backpropagation with `Inf` loss causes gradient clipping to fail and corrupts optimizer weights.

`abort_on_inf_loss` acts as the fail-stop trigger for infinity anomalies.

---

## 2. What it actually controls

```yaml
abort_on_inf_loss: true
```

- When `true` (default): MaxText checks `jnp.isinf(loss)` after every training step. If infinite loss is detected, it raises an exception and cleanly shuts down the training process.
- When `false`: Ignores infinite loss and attempts to proceed.

---

## 3. Options and Defaults

| Value | Behavior | Recommended Setting |
|---|---|---|
| `true` (default) | Aborts immediately when loss is `+Inf` or `-Inf` | Recommended for all production and testing runs |
| `false` | Ignores infinite loss | Never recommended in production |

---

## 4. Interactions and Stability Controls

- **`normalization_layer_epsilon`**: If `Inf` occurs frequently, check that `normalization_layer_epsilon` is not too small (e.g. use `1e-5`).
- **`gradient_clipping_threshold`**: Helps prevent gradient explosions that lead to `Inf` loss.

---

## 5. Practical Scenarios

- **Production Rule**: Always leave `abort_on_inf_loss: true`.

---

### One-line intuition

> **`abort_on_inf_loss` halts training immediately if an infinite loss is detected, preventing mathematical overflow from corrupting optimizer state.**
