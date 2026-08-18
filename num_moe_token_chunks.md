
## 1. The communication-compute overlap problem in expert parallelism

Expert parallelism (EP) shards the experts across devices. Each device holds a subset of experts. During a MoE forward pass:

```text
Step 1: Dispatch — all-gather/all-to-all to route tokens to the device holding their expert
Step 2: Compute  — each device runs its expert MLPs on the routed tokens
Step 3: Combine  — reduce-scatter/all-to-all to return results to originating devices
```

In the baseline (unchunked) approach:
```text
dispatch (communication)
     ↓  [devices idle during communication]
compute (GMM)
     ↓  [network idle during compute]
combine (communication)
```

Communication and compute are sequential — one idles the other.

`num_moe_token_chunks` enables pipelining: split the token batch into N chunks and overlap chunk k's communication with chunk (k-1)'s GMM compute.

---

## 2. The pipelining mechanic

With `num_moe_token_chunks=4`:

```text
Time →

chunk_0:  [dispatch] → [GMM] → [combine]
chunk_1:             [dispatch] → [GMM] → [combine]
chunk_2:                        [dispatch] → [GMM] → [combine]
chunk_3:                                   [dispatch] → [GMM] → [combine]

overlap:  chunk_1 dispatch ↔ chunk_0 GMM
          chunk_2 dispatch ↔ chunk_1 GMM
          ...
```

Each chunk's all-gather/reduce-scatter is overlapped with the previous chunk's expert computation. At high chunk counts, communication latency is hidden behind compute.

---

## 3. Prerequisites

This is **only** useful when:
1. `use_ring_of_experts=True` — the ring-of-experts path provides the structure for this overlap
2. Expert parallelism degree > 1 — you need inter-device communication to overlap

At `num_moe_token_chunks=1` (default), behavior is identical to the unchunked baseline.

---

## 4. Options

| Value | Meaning |
|---|---|
| `1` (default) | Disabled — no chunking, no pipeline overlap |
| `N > 1` | Split tokens into N chunks; requires `use_ring_of_experts=True` |

Typical values: 2, 4, 8. More chunks = more overlap opportunity but more XLA scheduling complexity and overhead per chunk.

---

## 5. Interaction with `moe_chunk_barrier`

With `num_moe_token_chunks > 1`, XLA may interleave chunk iterations in unexpected ways. `moe_chunk_barrier=True` forces strictly sequential execution by fencing each chunk on the previous — useful for debugging whether an overlap is causing correctness issues.

```text
num_moe_token_chunks=4, moe_chunk_barrier=false  →  chunks may interleave (desired for performance)
num_moe_token_chunks=4, moe_chunk_barrier=true   →  chunks execute strictly in order (for debugging)
```

---

## 6. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `use_ring_of_experts` | Must be `True` for chunking to provide overlap |
| `moe_chunk_barrier` | Disables overlap within chunks for debugging |
| `num_moe_emb_chunks` | Separate chunking along embedding dim; both can be used together |
| `capacity_factor` / `ragged_buffer_factor` | Chunk sizing still subject to capacity limits |

---

## 7. When to tune this

**Single-host training (no EP):** leave at `1` — no communication to overlap.

**Multi-host EP with `use_ring_of_experts=True` and communication-bound profile:** start with `num_moe_token_chunks=4`, benchmark.

**XLA failing to overlap automatically:** chunks give explicit structure for overlap scheduling.

---

### One-line intuition

> **`num_moe_token_chunks` splits the MoE token batch into N pipeline chunks so each chunk's expert-parallel all-gather/reduce-scatter overlaps the previous chunk's GMM compute — a latency-hiding trick that only matters with `use_ring_of_experts=True` and expert parallelism.**
