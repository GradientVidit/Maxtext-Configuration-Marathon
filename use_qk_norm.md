## 1. Why does `use_qk_norm` exist?

As transformer models scale in depth and sequence length, attention logits ($Q K^T / \sqrt{d_k}$) frequently suffer from numerical instability, attention entropy collapse, or loss spikes caused by unbounded growth in the norm of Query and Key vectors. 

Query-Key normalization bounds the magnitude of $Q$ and $K$ projections, ensuring stable gradient flow and preventing large logit outliers during training.

```text
Standard Attention:
Q, K projections ───► [ RoPE ] ───► Dot Product Q K^T / sqrt(d) ───► Softmax
                                      ▲ (Logits can explode)

With QK-Norm (Standard / Non-Llama4):
Q, K projections ───► [ RMSNorm ] ───► [ RoPE ] ───► Dot Product ───► Softmax

With QK-Norm (Llama4 Style):
Q, K projections ───► [ RoPE ] ───► [ L2Norm ] ───► Dot Product ───► Softmax
```

`use_qk_norm` toggles Query and Key normalization across the model's attention heads.

---

## 2. Architectural Mechanics: Pre-RoPE vs Post-RoPE

MaxText implements two distinct QK-norm behaviors depending on the target model family:

1. **Non-Llama4 Models (e.g. Qwen2, Gemma, ViT)**: Applies **RMSNorm** directly to $Q$ and $K$ *before* applying Rotary Position Embeddings (RoPE).
2. **Llama4 Models**: Applies **L2Norm** directly to $Q$ and $K$ *after* applying RoPE. This preserves rotary phase geometry while strictly constraining vector lengths on the unit hypersphere.

```text
Llama4 Branch:
  Q = apply_rope(Q, positions)
  K = apply_rope(K, positions)
  Q = Q / (norm(Q, ord=2, axis=-1, keepdims=True) + eps)
  K = K / (norm(K, ord=2, axis=-1, keepdims=True) + eps)
```

---

## 3. Options and Defaults

| Value | Behavior | Model Architecture |
|---|---|---|
| `false` (Default) | Standard unnormalized $Q$ and $K$ projections | Llama 2/3, Mistral, classic GPT models |
| `true` | Enables Q/K normalization | Llama4, Gemma 2, Qwen2.5 / Qwen3, ViT encoders |

---

## 4. Key Parameter Interactions

- **`qk_norm_with_scale`**: When enabled, applies a learnable gain multiplier ($\gamma$) to the normalized $Q$ and $K$ vectors.
- **`rope_type`**: Determines whether RoPE is applied before or after normalization.
- **`model_name`**: Preset configs automatically configure `use_qk_norm: true` for architectures requiring it (e.g. `llama4`, `gemma2-9b`).

---

## 5. What happens if misconfigured?

- **Enabling on pretrained models without QK-norm**: Utterly scrambles attention activations, resulting in gibberish text and infinite perplexity.
- **Disabling when fine-tuning Llama4/Gemma2**: The attention logits will scale completely out of expected dynamic range, immediately causing `NaN` gradients.

---

### One-line intuition
> **`use_qk_norm` constrains Query and Key vector magnitudes (via RMSNorm before RoPE or L2Norm after RoPE in Llama4) to prevent attention logit explosion and stabilize long-sequence training.**
