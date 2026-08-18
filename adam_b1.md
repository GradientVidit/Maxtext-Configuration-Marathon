## 1. Why does `adam_b1` exist?

First-order stochastic gradients contain substantial high-frequency noise from mini-batch sampling.

AdamW uses an Exponential Moving Average (EMA) of gradients—the first moment $m_t$—to estimate the true gradient direction and maintain momentum across steps:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

```text
Gradient Smoothing via adam_b1 (β1):

Raw Gradients:   /\  /\    /\  /\   (High-frequency stochastic noise)
                /  \/  \  /  \/  Smoothed (β1):    . ─ ─ ─ ─ ─ .     (Consistent velocity towards minimum)
```

`adam_b1` ($eta_1$) sets the exponential decay rate for tracking the first moment (momentum) of past gradients.

---

## 2. Fundamentals & Mechanics

- Controls the effective memory window of past gradients: roughly $\frac{1}{1 - \beta_1}$ steps.
- At default `adam_b1: 0.9`, momentum averages gradients across approximately the last 10 steps.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `0.9` | Standard first-moment momentum across Transformer training. |
| High Momentum | `0.95` | Smoother updates; useful for very noisy data mixtures. |
| Low Momentum | `0.8` | More responsive to abrupt gradient changes (often used in RL / RLHF). |

---

## 4. Interactions & Dependencies

```text
adam_b1 (First Moment β1) ──┐
                            ├──> Parameter Update Vector
adam_b2 (Second Moment β2) ──┘
```

---

## 5. Practical Scenarios & Failure Modes

- Deviating from `0.9` is rare in LLM pretraining; values $>0.98$ cause excessive momentum inertia, overshooting sharp minima.

---

### One-line intuition

> **`adam_b1` sets the exponential decay rate ($eta_1$) for gradient momentum, smoothing out per-batch stochastic gradient noise.**
