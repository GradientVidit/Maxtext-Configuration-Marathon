## 1. Why does `use_chunked_prefill` exist?

Standard transformer prefill calculates self-attention for all prompt tokens simultaneously. While efficient for short contexts, long-context serving suffers from memory exhaustion (OOM from massive attention logits) and severe Head-of-Line (HoL) blocking where running decode requests are starved while a massive prompt finishes computing.

```text
Standard Monolithic Prefill (use_chunked_prefill: false):
Request A (Prefill 16k): [========================================] 
Request B (Decode Step):                                           [Wait...] -> [Token]

Chunked Prefill (use_chunked_prefill: true):
Request A (Chunk 0):     [====]
Request B (Decode Step):       [Token]
Request A (Chunk 1):                  [====]
Request B (Decode Step):                    [Token]
```

`use_chunked_prefill` is the master boolean toggle enabling iterative, chunk-based prefill execution in MaxText inference pipelines.

---

## 2. Mechanics and Execution Flow

When enabled, the forward pass during inference switches from standard batch matrix multiply across the full sequence to a JAX `lax.scan` loop or JetStream chunked iteration:

```text
use_chunked_prefill: true
         │
         ▼
Check Prompt Length S vs prefill_chunk_size K
         │
    ┌────┴────────────────────────┐
    │ S <= K                      │ S > K
    ▼                             ▼
Single forward pass          Loop (i = 0 .. S/K - 1):
                             - Read slice [i*K : (i+1)*K]
                             - Query attends to past KV cache [0 : (i+1)*K]
                             - Update KV cache slice
                             - Output intermediate activation / final logits
```

---

## 3. Options Table

| Value | Behavior | Practical Context |
|---|---|---|
| `false` (Default) | Full sequence processed in a single pass | Training runs, offline batch inference, short-context benchmarking |
| `true` | Prompt is sliced into `prefill_chunk_size` blocks | Online production serving, long-context serving via JetStream |

---

## 4. Interactions

- **`prefill_chunk_size`**: Determines the exact token length of each chunk when `use_chunked_prefill: true`.
- **`enable_prefix_caching`**: Works symbiotically with chunked prefill; cached prefix chunks can be skipped entirely, feeding straight into the first novel chunk.
- **`inference_server`**: When running MaxText behind JetStream/gRPC server loops, chunked prefill ensures smooth streaming token delivery.

---

## 5. Failure Modes

- Enabling during standard pre-training will introduce unnecessary loop overheads and slow down step time. Keep `false` during training.
- If KV cache allocation is misconfigured or lacks causal masking across loop steps, attention will produce corrupted context states.

---

### One-line intuition
> **`use_chunked_prefill` toggles whether long prompts are computed in incremental token chunks to avoid peak memory spikes and prevent decode starvation during serving.**
