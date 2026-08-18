## 1. Why does `vocab_size` exist?

Neural networks do not compute on raw text characters or subwords; they compute over discrete integer token IDs indexed into an embedding matrix $E \in \mathbb{R}^{V  imes D}$ and projected back through an unembedding / output projection matrix $W_{ ext{head}} \in \mathbb{R}^{D  imes V}$.

```text
Raw Text ──[Tokenizer]──> Token IDs [0 .. V-1]
                                 │
                                 ▼
                     Embedding Matrix (V x D)
                                 │
                            Transformer
                                 │
                                 ▼
                      Logits Matrix (B*S x V)
                                 │
                        Cross-Entropy Loss
```

`vocab_size` ($V$) defines the total number of discrete token IDs the model recognizes. In distributed training on TPUs/GPUs with tensor parallelism or FSDP, the embedding and logit matrices are frequently sharded along the vocabulary dimension (`vocab` mesh axis). 

If `vocab_size` is not configured to match the tokenizer or is incompatible with hardware sharding rules (e.g. not divisible by the number of tensor parallel shards or a power of 2), training will either encounter index out-of-bounds CUDA/TPU crashes or severe XLA padding overhead.

---

## 2. Core Mechanics and Sharding Alignment

In MaxText, `vocab_size` determines the dimension of:
1. `shared_embedding/embedding` (or separate `logits_dense`)
2. Output projection and Softmax cross-entropy loss computation
3. `num_vocab_tiling` slicing chunks

```text
Full Vocab Dimension: V = 32,768
Mesh TP=4:
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Shard 0: 8k  │ Shard 1: 8k  │ Shard 2: 8k  │ Shard 3: 8k  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

When `vocab_size` is an exact power of 2 (e.g., 32,000, 32,768, 65,536, 128,256, 256,000 padded to 256,128), XLA can tile the matrix multiplications evenly across TPU Matrix Multiply Units (MXUs) with optimal 128x128 systolic tiling and clean SPMD tensor sharding without tail-padding penalties.

---

## 3. Options & Default

| Parameter | Type | Default | Valid Range |
| :--- | :--- | :--- | :--- |
| `vocab_size` | `int` | `32_000` | Positive integer (strongly recommended multiple of 128 / power of 2) |

---

## 4. Interactions with Related Parameters

```text
tokenizer_path / tokenizer_type ──> Defines actual token vocabulary
                                         │
                               Must match or be <=
                                         ▼
                                     vocab_size
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
          num_vocab_tiling                        logical_axis_rules ('vocab')
   (Chunks cross-entropy over V)                 (Shards V across TPU/GPU mesh)
```

- **`tokenizer_path` / `tokenizer_type`**: If the tokenizer emits an ID $\ge  ext{vocab\_size}$, embedding lookup throws an out-of-bounds error during execution.
- **`num_vocab_tiling`**: When `vocab_size` is huge (e.g., Gemma's 256,000 or Llama 3's 128,256), the un-sharded or un-tiled logit matrix ($B  imes S  imes V$) dominates HBM. `num_vocab_tiling` splits cross-entropy loss along the vocab axis.
- **`logits_via_embedding`**: If `true`, the input embedding table and output logit projection share weights of shape `(vocab_size, base_emb_dim)`.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Using Llama 3 Tokenizer with default config** | OOB index exception: Token ID 128000 >= 32000 | Set `vocab_size: 128256` in config or override CLI. |
| **Vocab size not divisible by TP mesh** | XLA compilation failure or inefficient padding on tensor parallel axis | Pad `vocab_size` to the next multiple of $(128  imes  ext{tensor\_parallel\_degree})$. |
| **Gemma 2 / Gemma 3 pretraining** | Out of Memory (OOM) during backprop on Softmax | Set `vocab_size: 256000` and enable `num_vocab_tiling: 8`. |

---

### One-line intuition

> `vocab_size` establishes the discrete token dimension for embeddings and logit cross-entropy, requiring strict power-of-two / mesh divisibility to ensure seamless XLA tensor sharding and avoid OOB memory faults.
