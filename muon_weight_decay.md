## 1. Why does `muon_weight_decay` exist?

Because Muon replaces coordinate-wise adaptive scaling with matrix-level orthogonalization (ensuring spectral norm updates), weight regularization must be applied directly as weight decay scaled by learning rate:

$$W_{t+1} = W_t - \eta_t \cdot (\text{OrthogonalUpdate} + \lambda_{\text{muon}} \cdot W_t)$$

```text
Muon Regularization:
Update = -η * OrthogonalMatrix - (η * muon_weight_decay) * W
```

`muon_weight_decay` controls the regularization shrinkage strength for parameters updated via Muon.

---

## 2. Fundamentals & Mechanics

- Multiplied directly by the learning rate on each step.
- Default `0` means no weight decay is applied to Muon matrix parameters by default.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `0` | No weight decay applied on Muon-managed 2D matrices. |
| Regularized | `0.01` to `0.1` | Adds proportional weight decay to prevent matrix norms from growing. |

---

## 4. Interactions & Dependencies

- Only applies to parameters optimized by Muon (when `opt_type: "muon"`).

---

## 5. Practical Scenarios & Failure Modes

- If matrix norms drift upward over hundreds of thousands of steps when using Muon, set `muon_weight_decay: 0.01`.

---

### One-line intuition

> **`muon_weight_decay` sets the weight decay regularization coefficient multiplied by the learning rate for Muon optimizer updates.**
