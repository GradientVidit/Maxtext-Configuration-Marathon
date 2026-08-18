## 1. Why does `abort_on_nan_loss` exist?

During large-scale distributed training, numerical instability (exploding gradients, activation overflow in FP16/BF16, or corrupted data batches) can cause loss computations to return `NaN` (Not a Number):

```text
Step N:   Loss = 1.62
Step N+1: Loss = NaN ──(If ignored)──> NaN propagates to all weights via optimizer state!
                        │
                        ▼ (abort_on_nan_loss: true)
                IMMEDIATE JOB TERMINATION
       (Prevents overwriting valid checkpoints with NaN!)
```

If training continues after a `NaN` step, corrupted gradients immediately overwrite model parameters and Adam momentum buffers with `NaN`s, permanently ruining the run and potentially overwriting valid checkpoints.

`abort_on_nan_loss` is the safety fail-stop that immediately terminates training when a `NaN` loss is encountered.

---

## 2. What it actually controls

```yaml
abort_on_nan_loss: true
```

- When `true` (default): MaxText verifies that the scalar training loss is not `NaN` at each step. If `jnp.isnan(loss)` evaluates to true, MaxText logs a critical error and aborts execution immediately.
- When `false`: Execution continues despite `NaN` values (unsafe; for compiler debugging only).

---

## 3. Options and Defaults

| Value | Behavior | Safety |
|---|---|---|
| `true` (default) | Aborts run immediately on `NaN` loss | Safe; protects optimizer state and checkpoints |
| `false` | Continues training on `NaN` | Unsafe; leads to corrupted model weights |

---

## 4. Interactions and Debugging

- **Checkpoints**: By aborting immediately, the previous valid checkpoint remains intact on GCS, allowing preemption auto-resume or rollback.
- **Pairing with `abort_on_inf_loss`**: Standard runs enable both `abort_on_nan_loss: true` and `abort_on_inf_loss: true`.

---

## 5. Practical Scenarios

- **Always keep `true` in production**: Disabling this setting in real training leads to wasted compute billing on dead runs.

---

### One-line intuition

> **`abort_on_nan_loss` immediately halts training upon detecting a NaN loss to prevent corrupted gradients from destroying model checkpoints.**
