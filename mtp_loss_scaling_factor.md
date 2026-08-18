
## 1. Why does `mtp_loss_scaling_factor` exist?

When MTP is enabled (`mtp_num_layers > 0`), the training loss combines the main next-token prediction loss with the auxiliary MTP losses:

```text
final_loss = main_loss + λ × mean(mtp_aux_losses)
```

Without a scaling factor, the auxiliary losses could dominate training — especially if the model is bad at predicting 2+ steps ahead early in training, making those losses large relative to the main loss. This would cause the optimizer to prioritize improving 2-step prediction over 1-step prediction, inverting the intended priority.

`mtp_loss_scaling_factor` (λ) is the weight that keeps the auxiliary signal informative without overwhelming the main objective.

---

## 2. Default

```yaml
mtp_loss_scaling_factor: 0.1
```

0.1 is taken from the DeepSeek-V3 paper — the MTP auxiliary loss contributes 10% of the gradient signal relative to the main loss. This is the published, empirically-validated default.

---

## 3. How it's used in the loss calculation

```text
With mtp_num_layers=1:
    main_loss = cross_entropy(predictions_step1, targets_step1)
    mtp_loss_1 = cross_entropy(predictions_step2, targets_step2)
    
    final_loss = main_loss + 0.1 × mtp_loss_1

With mtp_num_layers=2:
    mtp_loss_1 = cross_entropy(predictions_step2, targets_step2)
    mtp_loss_2 = cross_entropy(predictions_step3, targets_step3)
    avg_mtp_loss = (mtp_loss_1 + mtp_loss_2) / 2
    
    final_loss = main_loss + 0.1 × avg_mtp_loss
```

The MTP losses are averaged before scaling, so adding more MTP layers doesn't linearly increase the auxiliary loss contribution.

---

## 4. Tuning guidance

| Value | Effect |
|---|---|
| `0.0` | Effectively disables MTP loss (why have MTP at all?) |
| `0.1` | Default — balanced; matches DeepSeek-V3 |
| `0.3–0.5` | Stronger auxiliary signal; may help if MTP is primary objective |
| `1.0+` | Auxiliary loss dominates — likely degrades main task quality |

In practice, 0.1 is well-calibrated. Only change it if you have a specific reason from ablations.

---

## 5. Dependency: only active when `mtp_num_layers > 0`

When `mtp_num_layers: 0`, this parameter has no effect — there are no auxiliary losses to scale. Setting `mtp_loss_scaling_factor` without enabling MTP changes nothing.

---

## 6. Gradient signal perspective

From a gradient flow standpoint:
```text
λ = 0.1 means:
    For every 1 unit of gradient from main_loss,
    MTP contributes 0.1 units of gradient
    → MTP gradient is a minority contributor
    → Model primarily trained on main task, with MTP as regularizer
```

This is the intended behavior: MTP improves representations, not replaces the main objective.

---

### One-line intuition

> **`mtp_loss_scaling_factor` (λ) weights the auxiliary MTP prediction losses at training time — `0.1` (the DeepSeek-V3 default) keeps the multi-token objective as a regularizing signal without overwhelming the main next-token loss.**
