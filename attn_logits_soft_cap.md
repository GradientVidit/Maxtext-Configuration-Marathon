## 1. Why does `attn_logits_soft_cap` exist?

During large-scale autoregressive training, raw dot-product attention logits ($rac{q_i k_j^T}{\sqrt{d}}$) can occasionally grow very large in magnitude. When attention logits reach values like $50$ to $100+$, the subsequent softmax function becomes extremely peaked (one-hot), collapsing entropy and zeroing out gradients for all other tokens:

$$rac{\partial \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i (\delta_{ij} - \text{softmax}(z)_j) 	o 0$$

Hard clipping ($\text{clamp}(z, -c, c)$) bounds magnitudes but causes non-differentiable sharp kinks with zero gradient at the boundary.

**Logit Soft-Capping** (pioneered in **Gemma 2**) squashes logits smoothly into $(-c, c)$ using the hyperbolic tangent ($	anh$) function:

$$\text{logits}_{\text{capped}} = c \cdot 	anh\left(rac{\text{logits}}{c}
ight)$$

```text
Hard Clipping vs. Soft-Capping (c = 50.0):

Logit Value
  ▲
60│              / (Unbounded raw logit)
50│─────────────/─────── Hard Clip (clamp to 50, zero gradient for z > 50)
  │            /  . · · ·  Soft Cap: c · tanh(z/c) (asymptotically approaches 50)
40│           / ·
30│          /·
20│         /·
10│       ·/
 0└──────/────────────────► Raw Logit Input
```

Because $rac{d}{dz}[c 	anh(z/c)] = \text{sech}^2(z/c) > 0$ everywhere, non-zero gradients flow back even for extreme outlier activations.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `0.0` | Disabled. Standard unconstrained attention logits. | **Default**. Standard for LLaMA 1/2/3, Mistral, DeepSeek. |
| Any float $> 0.0$ (e.g. `50.0`) | Soft-caps attention logits to $[-c, c]$ before the attention softmax. | `50.0` is the exact Gemma 2 architecture value. |

Default in `base.yml`: `0.0`

---

## 3. Where Soft-Capping occurs in the Attention Forward Pass

```text
Q, K Projections ──> Dot Product (QKᵀ / √d)
                            │
                            ▼
              [ attn_logits_soft_cap > 0? ]
                     ├── Yes ──> logits = c · tanh(logits / c)
                     └── No  ──> logits = logits
                            │
                            ▼
                     [ Softmax Layer ]
                            │
                            ▼
                  Weighted Sum with V
```

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[final_logits_soft_cap]] | Companion parameter for vocabulary output logits ($c=30.0$ in Gemma 2). |
| [[z_loss_multiplier]] | Alternative loss-based logit stabilization technique (PaLM). Soft-capping modifies activations directly, while z-loss adds an auxiliary penalty. |
| [[qk_norm_with_scale]] | Alternative stabilization: normalizes $Q$ and $K$ vectors to unit norm before the dot product. |
| [[use_qk_clip]] | Alternative: hard-clips $Q$ and $K$ vector norms to $	au$ (MuonClip). |

---

## 5. Practical Scenarios & Inference Considerations

- **Reproducing Gemma 2:** Set `attn_logits_soft_cap: 50.0` and `final_logits_soft_cap: 30.0`.
- **Training Stability on Large Learning Rates:** If attention entropy collapses or loss spikes occur during pretraining, setting `attn_logits_soft_cap: 50.0` prevents logit runaway without destabilizing optimization.
- **Inference Kernel Compatibility Note:** FlashAttention v2 kernels and older do not support internal $\tanh$ soft-capping natively. Newer kernels (FlashAttention v3, Google's Pallas/Splash attention) do support it. If using an older kernel, soft-capping is often omitted at inference with negligible quality degradation, or evaluated using custom Pallas / Triton kernels.

---

### One-line intuition

> **`attn_logits_soft_cap` smoothly confines attention logits to $[-c, c]$ via $c \cdot \tanh(\text{logits}/c)$, eliminating attention entropy collapse and loss spikes without cutting off gradient flow.**
