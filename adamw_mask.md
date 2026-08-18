## 1. Why does `adamw_mask` exist?

Decaying 1D biases, RMSNorm/LayerNorm scale gains, or embedding vectors degrades representation quality and creates numerical instability:
- Normalization scales (e.g. $\gamma$ in RMSNorm) need to scale freely to preserve layer variance.
- Biases have minimal parameter count and do not contribute to overfitting.

```text
Parameter Classification for Weight Decay:
┌─────────────────────────────────┬─────────────────────────────────┐
│     2D Weight Matrices (MLP/Attn) │    Norm Scales, Biases, Embeds  │
└─────────────────────────────────┴─────────────────────────────────┘
                │                                   │
       [NOT in adamw_mask]                    [MATCHES adamw_mask]
                │                                   │
                ▼                                   ▼
   Apply adam_weight_decay (0.1)           ZERO Weight Decay (0.0)
```

`adamw_mask` specifies regex patterns for parameter names that should be excluded from weight decay.

---

## 2. Fundamentals & Mechanics

During optimizer creation:
- Each parameter path is checked against the regex list `adamw_mask`.
- Parameters that match (e.g. `['bias', '.*norm', '.*ln.*']`) receive a weight decay factor of `0.0`.
- All other parameters receive `adam_weight_decay`.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `[]` | Empty list: all parameters receive weight decay unless customized. |
| Recommended Best Practice | `['bias', '.*norm', '.*ln.*']` | Excludes biases and normalization scale parameters from decay. |

---

## 4. Interactions & Dependencies

```text
adam_weight_decay (0.1) ───[ Filtered through adamw_mask ]───> Parameter Update
```

---

## 5. Practical Scenarios & Failure Modes

- Decaying RMSNorm parameters can cause scale shrinkage that artificially inflates hidden layer activations, leading to overflow in fp16/bf16 dot products.

---

### One-line intuition

> **`adamw_mask` lists regex patterns to exempt sensitive parameters (such as normalization scales and biases) from weight decay.**
