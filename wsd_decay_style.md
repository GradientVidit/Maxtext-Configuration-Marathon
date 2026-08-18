## 1. Why does `wsd_decay_style` exist?

When annealing the learning rate in the decay phase of a Warmup-Stable-Decay (`wsd`) schedule, the transition from peak learning rate to final learning rate can follow different mathematical curves:

1. **Linear Decay (`'linear'`):** A straight-line constant deceleration, which is simple, predictable, and historically standard in WSD research (e.g. MiniCPM).
2. **Cosine Decay (`'cosine'`):** A smooth S-curve transition that avoids abrupt derivative kinks at the start and end of the decay window.

```text
WSD Decay Styles (Zoomed on Decay Window):

LR ^  Peak
   │  ┌──────\ 
   │  │       \  <-- 'linear' (Constant slope)
   │  │        \
   │  │         \
   │  │          \
   │  │  . - - .  \
   │  │ (       )  \ <-- 'cosine' (Smooth S-curve)
   │  │          ` .
   └──┴─────────────┴────────> Steps
      Decay Start   Decay End
```

`wsd_decay_style` selects the functional form of the learning rate reduction during the WSD decay phase.

---

## 2. Fundamentals & Mechanics

- **`'linear'` (Default):**
  $$\eta_t = \eta_{\text{peak}} - (\eta_{\text{peak}} - \eta_{\text{final}}) \cdot \frac{t - t_{\text{decay\_start}}}{N_{\text{decay}}}$$
- **`'cosine'`:**
  $$\eta_t = \eta_{\text{final}} + \frac{1}{2}(\eta_{\text{peak}} - \eta_{\text{final}})\left(1 + \cos\left(\pi \frac{t - t_{\text{decay\_start}}}{N_{\text{decay}}}\right)\right)$$

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `'linear'` | Linear straight-line decay during the WSD cooldown phase. |
| Alternative | `'cosine'` | Half-period cosine curve decay during the WSD cooldown phase. |

---

## 4. Interactions & Dependencies

```text
lr_schedule_type: 'wsd' ──> wsd_decay_steps_fraction ──> wsd_decay_style
```

- This parameter has **no effect** when `lr_schedule_type: 'cosine'`.

---

## 5. Practical Scenarios & Failure Modes

- **Empirical Preference:** Linear decay is standard in most WSD literature, but cosine decay provides smoother loss curves when the decay window is relatively short ($\le 5\%$ of training).

---

### One-line intuition

> **`wsd_decay_style` selects whether the terminal decay phase in WSD follows a straight `'linear'` ramp or a smooth `'cosine'` curve.**
