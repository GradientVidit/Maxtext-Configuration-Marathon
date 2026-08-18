## 1. Why does `final_logits_soft_cap` exist?

At the final output layer of a language model, hidden states are projected to vocabulary logits through the unembedding matrix $W_{vocab}$:

$$\text{logits} = h_{final} \cdot W_{vocab}^T \quad (\text{logits} \in \mathbb{R}^{V})$$

When training on long corpora with cross-entropy loss, models can become overconfident on frequent tokens, driving output logits toward $\pm \infty$. This causes:
1. **Numerical instability** during FP16/BF16 softmax and cross-entropy exponentiation.
2. **Probability collapse**, where temperature sampling becomes brittle and generated text degenerates into repetitive loops.
3. **Loss explosion** if an outlier token prediction receives zero probability mass due to numerical underflow.

**Final Logit Soft-Capping** (used in **Gemma 2**) passes vocabulary logits through a $	anh$ transfer function before computing cross-entropy:

$$\text{logits}_{\text{capped}} = c \cdot 	anh\left(rac{\text{logits}}{c}
ight)$$

```text
Hidden State (h) ──> Matmul(W_vocab) ──> Raw Logits [B, S, V]
                                                │
                                                ▼
                                [ final_logits_soft_cap > 0? ]
                                       ├── Yes ──> logits = 30.0 · tanh(logits / 30.0)
                                       └── No  ──> logits = logits
                                                │
                                                ▼
                                    Softmax / Cross-Entropy Loss
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `0.0` | Disabled. Output logits are unbounded. | **Default**. Standard for LLaMA, Mistral, GPT. |
| Any float $> 0.0$ (e.g. `30.0`) | Bounds final vocabulary logits to $[-c, c]$. | `30.0` is the exact Gemma 2 architecture value. |

Default in `base.yml`: `0.0`

---

## 3. Differences between `final_logits_soft_cap` and `attn_logits_soft_cap`

```text
Parameter                 Target Tensor            Location                     Gemma 2 Value
─────────────────────────────────────────────────────────────────────────────────────────────
attn_logits_soft_cap      QKᵀ attention scores     Inside every attention layer 50.0
final_logits_soft_cap     Vocabulary output logits Unembedding layer (Loss)     30.0
```

`attn_logits_soft_cap` protects the internal routing entropy of attention heads, while `final_logits_soft_cap` protects the prediction entropy over the vocabulary distribution.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[attn_logits_soft_cap]] | Sister parameter applied to attention heads. Typically paired together in Gemma 2 configs. |
| [[z_loss_multiplier]] | PaLM-style auxiliary loss penalty $\mathcal{L}_z = \lambda (\log \sum e^{z_i})^2$ addressing the same logit inflation problem via loss regularization rather than a nonlinear activation cap. |
| [[cast_logits_to_fp32]] | Casting logits to FP32 before soft-capping and softmax preserves numerical precision. |

---

## 5. Practical Scenarios

- **Gemma 2 Pretraining & Fine-tuning:** Set `final_logits_soft_cap: 30.0`.
- **Mitigating Repetitive Text & Overconfidence:** If a model exhibits high calibration error or overconfidence on common tokens, introducing final logit soft-capping keeps the output distribution well-calibrated.

---

### One-line intuition

> **`final_logits_soft_cap` constrains output vocabulary logits to $[-c, c]$ using $c \cdot 	anh(\text{logits}/c)$, preventing model overconfidence and numerical underflow in the loss function.**
