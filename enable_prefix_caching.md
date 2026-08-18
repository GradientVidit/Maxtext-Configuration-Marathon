## 1. Why does `enable_prefix_caching` exist?

In conversational AI, multi-turn chat, system prompts, few-shot prompting, and agentic workflows, large portions of the prompt context are identical across requests. Without caching, the model re-computes identical Key-Value (KV) activations repeatedly, wasting huge amounts of accelerator compute.

```text
Request 1: [System Prompt (2048 tokens)] + [User Query 1]
           └───────── Compute KV ───────┘ └─ Compute KV ─┘

Request 2: [System Prompt (2048 tokens)] + [User Query 2]
           Without Prefix Caching:
           └──────── Re-Compute KV ─────┘ └─ Compute KV ─┘ (Wasted Compute!)

           With Prefix Caching (enable_prefix_caching: true):
           └──────── Cache HIT (0 ms) ──┘ └─ Compute KV ─┘
```

`enable_prefix_caching` activates a dedicated KV-cache reuse subsystem (specifically within JetStream serving integrations) that stores computed KV prefixes in memory and retrieves them using prefix hash trees.

---

## 2. Architecture & Tiered Storage

Prefix caching in MaxText operates across a tiered memory hierarchy:

```text
Incoming Prompt Tokens
         │
         ▼
Compute Prefix Hash (Radix / Hash Tree)
         │
    ┌────┴──────────────────────────────┐
    │ Cache Hit in HBM                  │ Cache Miss / Evicted
    ▼                                   ▼
Load KV directly from HBM          Check Host DRAM Cache Tier
(Instant zero-compute start)            │
                                   ┌────┴────────────────┐
                                   │ Hit in DRAM         │ Miss
                                   ▼                     ▼
                             DMA Transfer to HBM   Compute KV on TPU MXU
                             (Fast host-to-device) & Store in Cache Tree
```

---

## 3. Options and Defaults

| Value | Behavior | Recommendation |
|---|---|---|
| `false` (Default) | Prefix caching disabled; all prompt tokens recomputed | Standard training and standalone benchmarking |
| `true` | Hierarchical prefix caching enabled in JetStream serving | High-throughput multi-turn chat, coding assistants, agent workloads |

---

## 4. Key Parameter Interactions

- **`prefix_caching_hbm_byte`**: Defines the size of the fast on-device TPU High Bandwidth Memory cache pool.
- **`prefix_caching_dram_byte`**: Defines the overflow storage pool in Host CPU DRAM.
- **`use_chunked_prefill`**: Complements prefix caching by allowing cache misses to be chunked smoothly after the hit boundary.

---

## 5. What breaks if misconfigured?

- **Enabling during training**: Unnecessary overhead; prefix caching is strictly an inference-serving feature.
- **Under-allocating HBM/DRAM**: Frequent evictions (thrashing) degrade cache hit rates, nullifying latency gains.
- **Over-allocating HBM**: Steals vital HBM space needed for the primary KV cache and model weights, triggering Out-Of-Memory (OOM) errors during heavy concurrency.

---

### One-line intuition
> **`enable_prefix_caching` enables JetStream KV prefix reuse across requests, skipping redundant prompt computations for shared prefixes like system prompts and chat history.**
