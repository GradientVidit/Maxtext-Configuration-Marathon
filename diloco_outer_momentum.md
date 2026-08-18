## 1. Why does `diloco_outer_momentum` exist?

Pseudo-gradients generated across $H$ local training steps can exhibit direction variance between successive sync rounds.

Applying Nesterov or standard momentum to the outer optimizer smooths the trajectory of the base model across synchronization intervals:

$$V_{k+1} = eta_{	ext{outer}} V_k + \Delta W_{k+1}$$
$$W_{	ext{base}, k+1} = W_{	ext{base}, k} - \eta_{	ext{outer}} V_{k+1}$$

```text
Outer Momentum Smoothing:
Sync Round k-1 Pseudo-Grad ──┐
                             ├──> Exponential Momentum (diloco_outer_momentum = 0.9) ──> Consistent Global Trajectory
Sync Round k   Pseudo-Grad ──┘
```

`diloco_outer_momentum` sets the momentum coefficient for the DiLoCo outer optimizer.

---

## 2. Fundamentals & Mechanics

- Default `0.9` maintains a running momentum vector across outer synchronization rounds.
- Accelerates progress along persistent gradient directions across long training horizons.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `0.9` | Standard momentum coefficient for outer optimization. |
| Disabled Momentum | `0.0` | Pure outer gradient descent (no historical outer momentum). |

---

## 4. Interactions & Dependencies

- Paired directly with `diloco_outer_lr`.

---

## 5. Practical Scenarios & Failure Modes

- Setting `0.9` is critical for matching standard single-cluster pretraining loss curves when using large sync periods ($H \ge 36$).

---

### One-line intuition

> **`diloco_outer_momentum` sets the momentum coefficient for the outer optimizer to maintain a consistent trajectory across synchronization rounds.**
