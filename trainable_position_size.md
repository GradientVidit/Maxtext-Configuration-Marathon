## 1. Why does `trainable_position_size` exist?

Before Rotary Position Embeddings (RoPE) became standard, models like GPT-2, GPT-3, and BERT used **learned (trainable) absolute positional embeddings**. 

Instead of computing fixed sinusoidal functions, the model allocates a dedicated parameter matrix $W_{pos} \in \mathbb{R}^{L_{max} \times D}$ where each position index $i \in [0, L_{max}-1]$ has its own gradient-updated embedding vector:

```text
Position Indices [0, 1, ..., T-1] ──> [ Trainable Table: W_pos (L_max, D) ] ──> Pos Embeddings [B, T, D]
                                                                                        │
                                                                                        ▼ (+)
Token IDs [B, T] ───────────────────> [ Token Embedding: W_tok (V, D) ] ─────> Token Embeddings [B, T, D]
                                                                                        │
                                                                                        ▼
                                                                                Layer 0 Input
```

`trainable_position_size` exists to define the maximum sequence length $L_{max}$ of this learned positional embedding table.

---

## 2. What it actually controls

```yaml
trainable_position_size: -1
```

- When `-1` (default): Trainable absolute positional embeddings are **disabled**.
- When `> 0` (e.g. `2048`): MaxText instantiates a trainable parameter array of shape `[trainable_position_size, base_emb_dim]` that is learned via backpropagation alongside token embeddings.

```text
trainable_position_size: -1    ──> No position weight matrix instantiated (Standard for RoPE models)
trainable_position_size: 2048  ──> Creates parameters: params['position_embed']['embedding'] of shape [2048, D]
```

---

## 3. Options and Defaults

| Value | Behavior | Memory Impact | Architecture Match |
|---|---|---|---|
| `-1` (default) | Disabled; relies on RoPE / relative embeddings | 0 extra parameters | Llama 2/3, Mistral, Gemma, DeepSeek |
| `> 0` (e.g., `2048`, `4096`) | Allocates learned position embedding table for positions $0 \dots \text{size}-1$ | Adds $\text{size} \times D$ parameters | GPT-2, GPT-3, OPT, BERT |

---

## 4. Interactions and Constraints

- **Sequence Length Limit (`max_target_length`)**: If `trainable_position_size > 0`, `max_target_length` must not exceed `trainable_position_size`. Any input token at index $i \ge \text{trainable\_position\_size}$ triggers an out-of-bounds indexing error during table lookup.
- **RoPE (`rope_type`)**: Modern models use `rope_type="default"` or `"llama3.1"` and keep `trainable_position_size: -1`. Combining learned absolute embeddings with rotary relative embeddings degrades extrapolation performance.
- **Sharding (`mesh_axes`, `logical_axis_rules`)**: Trainable position embedding tables are sharded across logical mesh axes according to `embed` rules, requiring parameter synchronization during gradient all-reduce.

---

## 5. Practical Scenarios

- **Training Llama / Mistral / Gemma / Qwen**: Keep `trainable_position_size: -1`.
- **Reproducing GPT-3 / OPT**: Set `trainable_position_size: 2048` (matching GPT-3's context length) and disable RoPE.
- **What breaks if misconfigured**: Setting `trainable_position_size: 1024` while passing input sequences of length `2048` causes runtime array out-of-bounds exceptions on TPU/GPU during the embedding gather.

---

### One-line intuition

> **`trainable_position_size` allocates and sizes a learned, gradient-updated absolute position embedding table for GPT-style architectures when set to a positive integer.**
