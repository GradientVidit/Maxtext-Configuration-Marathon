## 1. Why does `max_target_length` exist?

Accelerators (especially TPUs) achieve peak systolic execution performance by compiling static computation graphs where tensor shapes are fixed at compile time:

```text
Static Shape Allocation on TPU:
Input Batch: [ per_device_batch_size, max_target_length ]
KV Cache:    [ per_device_batch_size, max_target_length, num_heads, head_dim ]
```

Dynamic sequence lengths cause recompilations on XLA. Therefore, MaxText fixes the maximum sequence length (context window) across data loaders, attention masks, positional embeddings, and KV caches using a single unified parameter.

`max_target_length` defines the maximum sequence length (in tokens) for training, evaluation, and inference.

---

## 2. What it actually controls

```yaml
max_target_length: 2048
```

- Sets the token sequence dimension across all model inputs, targets, attention masks, and loss calculations.
- Determines the padding and packing horizon in Grain / TFDS data iterators.
- Allocates static buffer memory for activation rematerialization and KV cache.

```text
max_target_length = 2048
Token Sequence:  [ tok_0, tok_1, ..., tok_2047 ] ──> Shape: [B, 2048]
Attention Mask:  [ 2048 × 2048 ] Causal Matrix
Loss Mask:       [B, 2048] Boolean Weights
```

---

## 3. Options and Common Values

| Architecture / Task | `max_target_length` | Context Capability |
|---|---|---|
| Standard Test / Debug | `512` – `2048` | Minimal memory, fast compilation |
| Llama 2 Pretraining | `4096` | 4k context |
| Llama 3 / Gemma 2 Pretraining | `8192` | 8k context |
| Long Context Fine-Tuning | `32768`, `65536`, `131072` | 32k – 128k long-context |

---

## 4. Interactions and Scaling Laws

- **HBM Activation Memory**: Self-attention activation memory scales with $\mathcal{O}(    ext{max\_target\_length}^2)$ for full attention, or $\mathcal{O}(    ext{max\_target\_length})$ when using FlashAttention / chunked attention.
- **`per_device_batch_size`**: Increasing `max_target_length` by $4\times$ generally requires reducing `per_device_batch_size` or increasing Context Parallelism (`ici_context_parallelism`).
- **`max_prefill_predict_length`**: In `decode.py`, `max_prefill_predict_length` must be $\le     ext{max\_target\_length}$.

---

## 5. Practical Scenarios

- **Out of Memory (OOM) on TPU**: If scaling up `max_target_length` to $32k$ or $128k$ causes OOM, activate `attention_type: "flash"`, set `context_sharding: true`, and adjust rematerialization (`remat_policy`).
- **Data Packing**: When `packing: true`, `max_target_length` is the length of the packed sequence containing multiple concatenated examples.

---

### One-line intuition

> **`max_target_length` defines the static maximum token sequence length for training, evaluation, and KV cache allocation.**
