## 1. Why does `lr_schedule_type` exist?

The learning rate should not remain static throughout training. Early in training, large step sizes help discover broad basins in the loss landscape; later, smaller step sizes allow fine-grained convergence into sharp minima.

Different training paradigms demand different schedule shapes:
1. **Fixed-Horizon Pretraining (`cosine`):** Designed when the exact total token/step count is known in advance (e.g. Llama 2).
2. **Continual Pretraining & Flexibility (`wsd`):** Warmup-Stable-Decay maintains a high learning rate across the entire run and only decays when you decide to finish, allowing arbitrary training extensions.

```text
Cosine Schedule:
LR ^        Peak
   |         /\
   |        /  \
   |       /    \  (Continuous smooth decay)
   |      /      \
   |_____/________\______> Steps

WSD (Warmup-Stable-Decay):
LR ^        Peak ─────────────────────┐ (Stable phase: max learning)
   |         /                        \
   |        /                          \ (Rapid final decay)
   |       /                            \
   |______/______________________________\______> Steps
```

`lr_schedule_type` selects the mathematical trajectory governing learning rate decay over time.

---

## 2. Fundamentals & Mechanics

MaxText supports two schedule topologies:

### `'cosine'` (Default)
- **Warmup:** Linear increase from $0$ to `learning_rate`.
- **Decay:** Smooth half-cosine decay from peak down to `learning_rate * learning_rate_final_fraction` across all remaining schedule steps.
- **Trailing:** Constant zero LR for any steps beyond `learning_rate_schedule_steps`.

### `'wsd'` (Warmup-Stable-Decay)
- **Warmup:** Linear increase to peak.
- **Stable Phase:** Constant peak `learning_rate` for $(1 - \text{warmup\_fraction} - \text{wsd\_decay\_fraction})$ of the schedule.
- **Decay Phase:** Fast decay (linear or cosine, chosen via `wsd_decay_style`) over the final fraction of steps.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `'cosine'` | Standard smooth cosine decay over the entire training horizon. |
| Alternative | `'wsd'` | Warmup-Stable-Decay schedule for continual training and flexible checkpoints. |

---

## 4. Interactions & Dependencies

```text
                          lr_schedule_type
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
      ['cosine']                                    ['wsd']
           │                                           │
  warmup_steps_fraction                      wsd_decay_steps_fraction
learning_rate_final_fraction                 wsd_decay_style
                                             learning_rate_final_fraction
```

---

## 5. Practical Scenarios & Failure Modes

- **Choosing `cosine` for fixed runs:** When training a known model on a fixed token budget (e.g. Chinchilla optimal 300B tokens), `'cosine'` offers predictable, proven convergence.
- **Choosing `'wsd'` for open-ended runs:** If you are unsure whether you will train for 100k steps or 300k steps, `'wsd'` lets you branch off checkpoints from the stable phase and run a quick decay whenever desired.

---

### One-line intuition

> **`lr_schedule_type` selects the learning rate curve shape—either smooth fixed-horizon `'cosine'` decay or flexible `'wsd'` (Warmup-Stable-Decay).**
