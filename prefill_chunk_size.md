## 1. Why does `prefill_chunk_size` exist?

During LLM inference serving, prompt processing (prefill) scales quadratically in compute and linearly in memory with sequence length $S$. Processing massive prompts (e.g., 32k-128k tokens) in a single monolithic forward pass causes massive activation spikes, blocks accelerator execution, and prevents concurrently scheduled token generation (decode) steps from meeting strict Service Level Objectives (SLOs).

```text
Monolithic Prefill:
Prompt [----------------- 32k tokens -----------------]
Memory: █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ (Peak HBM spike)
TPU Engine: [============ Prefill Locked ============] (Decode requests wait)

Chunked Prefill (prefill_chunk_size = 256):
Chunk 0: [256] -> Compute KV -> Cache -> Yield
Chunk 1: [256] -> Attend to [0..255] + [256] -> Cache -> Yield
...
Chunk N: [256] -> Final logits -> Start Decode
Memory:  ██ (Flat, bounded activation memory)
```

`prefill_chunk_size` defines the granularity (token chunk dimension) used to slice long input prompts into iterative sequential steps.

---

## 2. What does it actually control?

When `use_chunked_prefill: true`, `prefill_chunk_size` fixes the sequence tile length processed by the attention and feed-forward layers per sub-step.

```text
Input Prompt (Length S)
        │
        ▼
Slice into chunks of size K = prefill_chunk_size
        │
        ├── Chunk 0: tokens [0 : K]
        │     └── Forward pass -> write KV to cache [0 : K]
        ├── Chunk 1: tokens [K : 2K]
        │     └── Attend over KV[0 : 2K] -> write KV to cache [K : 2K]
        └── Chunk i: tokens [i*K : (i+1)*K]
              └── Causal self-attention + cross-chunk KV cache accumulation
```

1. **Activation Footprint**: Peak temporary intermediate tensors scale with `prefill_chunk_size` rather than full prompt length $S$.
2. **Decode Interleaving**: Allows high-priority autoregressive decode steps to be co-scheduled between chunks, slashing Time-To-First-Token (TTFT) and decode jitter.

---

## 3. Options and Defaults

| Value | Meaning | Use Case |
|---|---|---|
| `256` (Default) | Default chunk tile dimension | Balanced latency/throughput compromise for TPU v4/v5e/v6e |
| `512` / `1024` | Larger chunk size | Maximizes hardware Matrix Multiply Units (MXU) FLOP utilization on large TPUs |
| `64` / `128` | Smaller chunk size | Ultra-low TTFT jitter in latency-critical serving deployments |

MaxText config default: `256` in `base.yml`.

---

## 4. Interactions and Dependencies

- **`use_chunked_prefill`**: Master switch. If `false`, `prefill_chunk_size` is completely ignored and full prompt sequences are evaluated in one shot.
- **`max_prefill_predict_length`**: Determines total prefill capacity. Must be an integer multiple of `prefill_chunk_size` or padded accordingly.
- **`attention` backend**: Chunked prefill requires causal KV cache concatenation support across scan iterations.

---

## 5. Practical Scenarios & Pitfalls

- **Setting it too small (e.g. 16 or 32)**: TPU MXU hardware utilization plummets due to memory bandwidth boundedness on small matrix multiplications.
- **Setting it too large (e.g. 8192)**: Reintroduces memory spikes and monopolizes accelerator execution, defeating the purpose of chunking.
- **Optimal tuning**: On TPU v5e/v6e, `256` or `512` aligns well with native TPU systolic array vector/matrix register tile dimensions.

---

### One-line intuition
> **`prefill_chunk_size` sets the token slice size for chunked prefill, bounding peak activation memory and enabling fine-grained interleaving of decode steps.**
