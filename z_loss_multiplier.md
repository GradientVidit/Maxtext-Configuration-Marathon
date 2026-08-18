## 1. Why does `z_loss_multiplier` exist?

The standard categorical cross-entropy loss with softmax is **shift-invariant**: adding an arbitrary constant scalar $C$ to every logit $z_i$ does not change the resulting probabilities $p_i$:

$$\text{softmax}(z_i + C) = rac{e^{z_i + C}}{\sum_j e^{z_j + C}} = rac{e^C e^{z_i}}{e^C \sum_j e^{z_j}} = \text{softmax}(z_i)$$

Because the loss function does not penalize uniform logit shifts, neural networks during long training runs can drift toward assigning enormous absolute values to all logits (e.g. $z_i pprox 10{,}000$). While mathematically equivalent in infinite precision, in float32 / bfloat16 computation this causes:
- Log-sum-exp overflow and NaN gradients.
- Severe rounding errors during backpropagation.

**Z-Loss** (introduced in **ST-MoE**, Zoph et al., 2022; also applied in PaLM and other large-scale pretraining) introduces an auxiliary regularization term that penalizes the square of the softmax partition function $\log Z = \log \sum_i \exp(z_i)$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cross-entropy}} + \lambda_z \cdot \left(\log \sum_{i=1}^V \exp(z_i)
ight)^2$$

```text
Without Z-Loss:
  Logits drift: [1000.2, 998.4, 1005.1] --> Softmax output is valid, but log-sum-exp ≈ 1005 (numerical risk)

With Z-Loss (z_loss_multiplier > 0):
  Auxiliary gradient pulls log-sum-exp toward 0 --> Logits center around 0: [1.2, -0.6, 6.1]
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `0.0` | Disabled. Standard cross-entropy loss without partition function penalty. | **Default**. Standard for LLaMA. |
| Any float $> 0.0$ (e.g. `1e-4`) | Adds $\lambda_z \cdot (\log \sum \exp(z_i))^2$ to the training loss. | PaLM and ST-MoE use $10^{-4}$ ($0.0001$). |

Default in `base.yml`: `0.0`

---

## 3. Z-Loss vs. Soft-Capping vs. QK-Norm

```text
Technique                 Mechanism                                    Location
────────────────────────────────────────────────────────────────────────────────────────────
z_loss_multiplier         Auxiliary loss penalty $\lambda (\log Z)^2$    Loss computation (PaLM)
final_logits_soft_cap     Activation function $c 	anh(z/c)$            Forward logits (Gemma 2)
qk_norm_with_scale        LayerNorm on Q and K vectors                  Attention heads (ViT/Qwen)
```

Z-loss preserves the exact linear property of output logits without introducing nonlinear distortion, relying purely on gradient descent to keep logits near zero.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[final_logits_soft_cap]] | Alternative logit stabilization. Generally, models use either Z-loss (PaLM) or Soft-Capping (Gemma 2), not both. |
| [[cast_logits_to_fp32]] | Z-loss computation ($\log \sum \exp$) should always be performed in FP32 to prevent exponent overflow before the penalty takes effect. |

---

## 5. Practical Scenarios

- **PaLM / ST-MoE Style Pretraining:** Set `z_loss_multiplier: 1.e-4` to prevent logit drift across hundreds of billions of tokens.
- **Large Vocabulary Scaling ($V \ge 128\text{K}$):** Larger vocabularies naturally produce larger log-sum-exp values; Z-loss keeps partition function magnitudes bounded.

---

### One-line intuition

> **`z_loss_multiplier` adds an auxiliary penalty $\lambda (\log \sum e^{z_i})^2$ to the loss, preventing output logits from drifting to large absolute values and ensuring FP32/BF16 stability throughout long training runs.**
