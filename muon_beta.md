## 1. Why does `muon_beta` exist?

Muon (Momentum Orthogonalized Optimizer) optimizes 2D weight matrices by computing an exponentially-weighted moving average of gradients and orthogonalizing the resulting matrix via iterative Newton-Schulz steps.

The momentum parameter $eta$ governs how aggressively past gradient matrices are smoothed before orthogonalization:

$$M_t = \beta M_{t-1} + (1 - \beta) G_t$$

```text
Muon Update Pipeline:
Raw Gradients (G_t) ──> EMA with muon_beta (M_t) ──> Newton-Schulz Polar Decomp ──> Orthogonal Update
```

`muon_beta` defines the momentum decay rate for the exponentially weighted gradient matrix in Muon.

---

## 2. Fundamentals & Mechanics

- Matches the functional role of $eta_1$ in AdamW, but operates on full 2D matrix blocks.
- Default `0.95` provides a smooth matrix trajectory that prevents erratic subspace rotations during orthogonalization.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `0.95` | Standard momentum decay rate recommended for Muon. |
| Alternative | `0.90` | Faster response to rapid loss landscape changes. |

---

## 4. Interactions & Dependencies

```text
opt_type: "muon" ──> muon_beta ──> Newton-Schulz Iterations
```

---

## 5. Practical Scenarios & Failure Modes

- Setting `muon_beta` too low (e.g. `0.5`) causes noisy orthogonalized updates, destabilizing attention layers.

---

### One-line intuition

> **`muon_beta` sets the momentum decay rate for the gradient moving average in the Muon optimizer before matrix orthogonalization.**
