## 1. Why does `wsd_decay_steps_fraction` exist?

Under the Warmup-Stable-Decay (`wsd`) learning rate schedule, the model spends the vast majority of training at the peak learning rate (the "stable" phase) and only executes an aggressive cooldown (the "decay" phase) near the end of training.

Without a dedicated fraction parameter, MaxText would not know how many steps to allocate to this terminal annealing phase:

```text
Total Schedule Steps (learning_rate_schedule_steps)
├───────────────┼────────────────────────────────────────┼──────────────┤
│ Warmup Phase  │              Stable Phase              │ Decay Phase  │
│  (e.g., 10%)  │              (e.g., 80%)               │ (e.g., 10%)  │
└───────────────┴────────────────────────────────────────┴──────────────┘
                                                         ▲
                                             wsd_decay_steps_fraction
```

`wsd_decay_steps_fraction` specifies what percentage of the total schedule is dedicated to the terminal decay phase.

---

## 2. Fundamentals & Mechanics

When `lr_schedule_type: 'wsd'`:
- **Decay Step Count:** $N_{\text{decay}} = \text{learning\_rate\_schedule\_steps} \times \text{wsd\_decay\_steps_fraction}$.
- **Stable Step Count:** $N_{\text{stable}} = \text{learning\_rate\_schedule\_steps} \times (1 - \text{warmup\_steps\_fraction} - \text{wsd\_decay\_steps\_fraction})$.

During this decay phase, the learning rate transitions from peak `learning_rate` down to `learning_rate * learning_rate_final_fraction`.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `0.1` | 10% of the schedule is spent in the final decay phase. |
| Fast Cooldown | `0.05` | 5% aggressive decay (useful for rapid evaluation branching). |
| Extended Annealing | `0.15` to `0.20` | 15–20% gradual decay for smoother convergence on multi-dataset mixtures. |

---

## 4. Interactions & Dependencies

```text
                      lr_schedule_type: 'wsd'
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
wsd_decay_steps_fraction                        wsd_decay_style
 (Duration of decay)                          (Shape: linear/cosine)
```

- **`wsd_decay_style`:** Determines whether the decay over this duration follows a linear slope or a half-cosine curve.
- **Validity Constraint:** $\text{warmup\_steps\_fraction} + \text{wsd\_decay\_steps\_fraction} < 1.0$.

---

## 5. Practical Scenarios & Failure Modes

- **Branching Cooldowns from Checkpoints:** In WSD pretraining workflows, teams save stable checkpoints and launch small runs with `wsd_decay_steps_fraction: 0.1` to evaluate benchmark performance without committing the main run to decay.
- **Decay Phase Too Short:** Setting `wsd_decay_steps_fraction: 0.01` (1%) can create gradient shocks and instability during rapid parameter shrinkage.

---

### One-line intuition

> **`wsd_decay_steps_fraction` defines the proportion of total schedule steps allocated to the terminal cooling/decay phase in Warmup-Stable-Decay training.**
