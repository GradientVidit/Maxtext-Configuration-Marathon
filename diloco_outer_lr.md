## 1. Why does `diloco_outer_lr` exist?

When DiLoCo replicas synchronize, they produce a "pseudo-gradient" representing the total trajectory displacement over $H$ local steps:

$$\Delta W = W_{	ext{base}} - rac{1}{R} \sum_{r=1}^R W_{	ext{local}}^{(r)}$$

A simple direct average ($\Delta W$) does not capture global trajectory velocity. The **outer optimizer** treats $\Delta W$ as a gradient step on the global base weights with its own dedicated learning rate:

$$W_{	ext{base}, t+1} = W_{	ext{base}, t} - \eta_{	ext{outer}} \cdot \Delta W$$

```text
Outer Step Calculation:
Averaged Local Weights ──> Compute Displacement (Pseudo-Grad) ──> Scale by diloco_outer_lr (0.3) ──> Base Model Update
```

`diloco_outer_lr` specifies the step size for the DiLoCo outer optimizer.

---

## 2. Fundamentals & Mechanics

- Default `0.3` provides aggressive outer updates that accelerate convergence towards the consensus manifold.
- Distinct from the inner `learning_rate` used for per-step batch updates.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `0.3` | Standard outer learning rate from DiLoCo research. |
| Conservative | `0.1` | Slower, more conservative outer updates. |
| Aggressive | `0.7` to `1.0` | Direct extrapolation along consensus direction. |

---

## 4. Interactions & Dependencies

```text
diloco_outer_lr (0.3) + diloco_outer_momentum (0.9) ──> DiLoCo Outer Optimizer
```

---

## 5. Practical Scenarios & Failure Modes

- Setting `diloco_outer_lr` too high ($>1.5$) can cause outer trajectory overshoot and loss spikes after sync steps.

---

### One-line intuition

> **`diloco_outer_lr` sets the learning rate for the outer optimizer that applies synchronized replica updates to the base model weights.**
