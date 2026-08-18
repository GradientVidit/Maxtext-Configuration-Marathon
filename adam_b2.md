## 1. Why does `adam_b2` exist?

Different weights in a deep Transformer experience dramatically different gradient scales: embeddings and normalization layers often receive sparse or massive gradients, while deep residual layers receive smaller signals.

AdamW tracks the second uncentered moment $v_t$ (the variance proxy) to normalize parameter updates per-coordinate:

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

```text
Per-Coordinate Scaling:
Large Gradients (v_t large)  ──> Scaled DOWN by 1 / sqrt(v_t)
Small Gradients (v_t small)  ──> Scaled UP   by 1 / sqrt(v_t)
Equalized update velocity across all network layers
```

`adam_b2` ($eta_2$) sets the exponential decay rate for tracking the second moment of squared gradients.

---

## 2. Fundamentals & Mechanics

- Effective memory horizon: roughly $\frac{1}{1 - \beta_2}$ steps.
- At default `adam_b2: 0.95` (Llama 2 / modern LLM standard), the memory spans $\approx 20$ steps.
- Older deep learning conventions used `0.999` (1,000 steps), but modern LLM training uses `0.95` or `0.98` to rapidly forget outdated variance estimates under dynamic learning rates.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `0.95` | Modern standard for LLMs (Llama 2, Mistral, Gemma). |
| Conservative | `0.98` or `0.99` | Longer variance memory for large-batch pretraining runs. |
| Classic | `0.999` | Classic Adam baseline (slower variance adaptation). |

---

## 4. Interactions & Dependencies

```text
adam_b2 (0.95) ──> Rapidly adapts coordinate scales to changing loss curvature
```

---

## 5. Practical Scenarios & Failure Modes

- Using `adam_b2: 0.999` with fast learning rate decay causes the optimizer to retain stale variance statistics from earlier high-LR steps, slowing late-stage convergence.

---

### One-line intuition

> **`adam_b2` sets the exponential decay rate ($eta_2$) for tracking squared gradient variance, adapting update step sizes coordinate-by-coordinate.**
