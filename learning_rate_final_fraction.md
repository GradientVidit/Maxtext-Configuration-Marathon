## 1. Why does `learning_rate_final_fraction` exist?

Decaying the learning rate all the way to absolute zero ($0.0$) at the end of training is often undesirable:
1. True zero LR prevents the model from making any parameter adjustments in final gradient batches.
2. In continual pretraining or domain adaptation, keeping a small non-zero terminal learning rate preserves plasticity.

```text
Final Learning Rate Floor:

Peak LR (1.0x) ──────────────────┐
                                  \
                                   \
                                    \
Final LR Floor ──────────────────────┴────────> (learning_rate * final_fraction)
                                                (e.g., 0.1x of peak)
Zero Floor ───────────────────────────────────> 0.0 (Traditional zero-decay)
```

`learning_rate_final_fraction` defines the minimum learning rate floor at the end of the schedule as a proportion of the peak `learning_rate`.

---

## 2. Fundamentals & Mechanics

The terminal learning rate $\eta_{\text{final}}$ is calculated as:

$$\eta_{\text{final}} = \text{learning\_rate} \times \text{learning\_rate\_final\_fraction}$$

- Applies to both `'cosine'` and `'wsd'` schedule types.
- With default `learning_rate = 3.e-5` and `learning_rate_final_fraction = 0.1`, the schedule decays to $3.\text{e-}6$.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `0.1` | Final LR decays to 10% of peak LR (standard in Llama 2 / 3). |
| Zero Decay | `0.0` | Decays completely to zero at the final scheduled step. |
| Conservative Floor | `0.2` or `0.05` | Custom floor for specialized annealing runs. |

---

## 4. Interactions & Dependencies

```text
learning_rate ──┐
                ├──> Terminal LR = learning_rate * learning_rate_final_fraction
learning_rate_  │
final_fraction ─┘
```

- **`learning_rate_schedule_steps`:** If `steps > learning_rate_schedule_steps`, the learning rate reaches `learning_rate * learning_rate_final_fraction` at `learning_rate_schedule_steps`, and drops to `0.0` for any remaining overshoot steps.

---

## 5. Practical Scenarios & Failure Modes

- **Matching Published Baselines:** Modern LLMs (Llama, Mistral, Gemma) consistently use `0.1` (10% of peak) rather than decaying to `0.0`, ensuring stable late-stage loss dynamics.
- **Accidental Zero Floor:** Setting `0.0` with cosine decay can cause the last 2–3% of training steps to have negligible gradient impact.

---

### One-line intuition

> **`learning_rate_final_fraction` sets the minimum terminal learning rate as a fraction of peak LR, preventing the schedule from decaying to absolute zero.**
