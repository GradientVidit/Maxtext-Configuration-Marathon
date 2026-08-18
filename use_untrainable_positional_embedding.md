## 1. Why does `use_untrainable_positional_embedding` exist?

Transformers have no inherent notion of sequence order because self-attention is permutation-equivariant:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

To inject order, models must encode token positions. Early architectures (like the original Transformer in "Attention Is All You Need") used fixed, non-learned sinusoidal position embeddings added directly to the input embeddings:

```text
Token IDs ──> [ Embedding Table ] ──> Input Vectors [B, T, D]
                                             │
                                             ▼ (+)
Positions ──> [ Fixed Sinusoid Table ] ──────┘
                                             │
                                             ▼
                                     Transformer Layers
```

Modern autoregressive LLMs (Llama, Gemma, DeepSeek) use Rotary Position Embeddings (RoPE) inside the attention layers rather than absolute embeddings at the input. 

`use_untrainable_positional_embedding` exists as an architectural switch to enable classic fixed sinusoidal absolute positional embeddings at the input layer for models that require non-learned sinusoidal coordinate representations.

---

## 2. What it actually controls

```yaml
use_untrainable_positional_embedding: false
```

- When `false` (default): MaxText relies on the primary positional mechanism configured (typically RoPE via `rope_type`, or learned embeddings if `trainable_position_size > 0`).
- When `true`: MaxText generates deterministic sinusoidal positional encodings using geometric frequency progressions and adds them to token embeddings before the first Transformer block.

```text
Fixed Sinusoidal Encoding Scheme:
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

Sequence Positions: [0, 1, 2, ..., T-1]
           │
           ▼
[ Fixed Sin/Cos Matrix ] (No gradients, no trainable weights)
           │
           ▼
Added to Token Embeddings prior to Layer 0
```

---

## 3. Options and Defaults

| Value | Behavior | Trainable Parameters | Common Models |
|---|---|---|---|
| `false` (default) | No fixed input sinusoidal embedding added; uses RoPE or learned embeddings | 0 | Llama 2/3, Gemma 1/2, Mistral, Qwen |
| `true` | Adds deterministic sinusoidal position vectors to input tokens | 0 | Original Attention Is All You Need (Vanilla Transformer) |

---

## 4. Interactions and Architectural Conflicts

```text
                         Positional Encoding Selection
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
`use_untrainable_positional_embedding: true`      `trainable_position_size > 0`
   (Fixed Vaswani Sinusoids at Input)             (Learned GPT-style Table at Input)
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      │
                                      ▼
                        `rope_type != "none"`
             (Rotary embeddings inside Attention Q/K heads)
```

- **Conflict with RoPE**: If using RoPE (`rope_type="default"` or `"llama3.1"`), `use_untrainable_positional_embedding` should remain `false`. Applying both adds redundant and conflicting position signals (absolute at input + relative in attention).
- **Conflict with `trainable_position_size`**: Do not enable both fixed untrainable positional embeddings and learned positional tables simultaneously.

---

## 5. Practical Scenarios

- **Standard LLM Pretraining / Fine-Tuning**: Keep `false`. Modern architectures use RoPE.
- **Original Transformer / Encoder-Decoder Replication**: Set `true` if reproducing the exact 2017 Vaswani et al. architecture with fixed sinusoidal embeddings.

---

### One-line intuition

> **`use_untrainable_positional_embedding` injects fixed, non-learned sinusoidal coordinate embeddings into input tokens before layer 0 for classic vanilla Transformer architectures.**
