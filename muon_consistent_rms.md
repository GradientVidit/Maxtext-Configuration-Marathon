## 1. Why does `muon_consistent_rms` exist?

When orthogonalizing 2D weight matrices of varying aspect ratios ($d_{\text{in}} \times d_{\text{out}}$), the Root Mean Square (RMS) of matrix updates can vary depending on matrix dimensions:

1. **Width-based scaling (`None`):** Scales updates using matrix dimension formulas (e.g. $\sqrt{\max(A, B)}$).
2. **Consistent RMS scaling (Float, e.g. `0.2`):** Forces the update matrix to have a constant, uniform RMS norm across all layers regardless of hidden dimension or aspect ratio.

```text
Update Scaling Strategy:
  muon_consistent_rms: None ──> Scale depends on sqrt(fan_in / fan_out)
  muon_consistent_rms: 0.2  ──> Every matrix update has exact RMS = 0.2 across all layers
```

`muon_consistent_rms` selects between aspect-ratio-dependent scaling and constant RMS scaling for Muon updates.

---

## 2. Fundamentals & Mechanics

- When `None` (default): MaxText applies standard width-dependent scaling factors.
- When set to a float (e.g. `0.2`): The orthogonalized update matrix $U$ is normalized so that $\text{RMS}(U) = \sqrt{\frac{1}{N} \sum U_{ij}^2} = 0.2$.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `None` | Uses standard width-based scaling for matrix updates. |
| Recommended RMS | `0.2` | Forces constant 0.2 RMS update scale across all layers. |

---

## 4. Interactions & Dependencies

```text
opt_type: "muon" ──> muon_consistent_rms (Scaling formulation)
```

---

## 5. Practical Scenarios & Failure Modes

- When scaling up model width (e.g. from 1B to 70B), setting `muon_consistent_rms: 0.2` provides predictable step sizes across different layer shapes.

---

### One-line intuition

> **`muon_consistent_rms` controls whether Muon update steps are scaled by matrix aspect ratios or clamped to a constant RMS magnitude (recommended `0.2`).**
