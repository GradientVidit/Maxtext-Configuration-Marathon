## 1. Why does `prefix_caching_dram_byte` exist?

While TPU HBM offers extreme bandwidth, its capacity per chip is small and expensive. Conversely, host CPU DRAM is abundant (hundreds of gigabytes to terabytes per host). 

When high numbers of unique system prompts or document prefixes are used across thousands of user sessions, the fast HBM cache fills up quickly. Instead of completely discarding evicted KV prefixes (forcing costly recomputation on future hits), MaxText and JetStream spill them over into Host CPU DRAM.

```text
Two-Tier Cache Hierarchy:
┌─────────────────────────────────────────────────────────────┐
│ Tier 1: Fast HBM Cache (e.g., 10 GB)                        │
│ -> 0 latency, direct MXU access                             │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Evict when full)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Tier 2: Overflow DRAM Cache (e.g., 100 GB)                  │
│ -> Low latency DMA transfer over PCIe to HBM on cache hit   │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Evict when full)
                               ▼
                        [Drop / Recompute]
```

`prefix_caching_dram_byte` sets the byte allocation budget for this secondary host DRAM cache pool.

---

## 2. Data Flow on Cache Lookup

```text
Incoming Prefix Hash
         │
         ├── 1. Check HBM Cache -> Hit? -> Run immediately
         │
         └── 2. Miss in HBM -> Check Host DRAM Cache
                     │
              ┌──────┴───────────────────────┐
              │ DRAM Hit                     │ DRAM Miss
              ▼                              ▼
        DMA Host-to-Device (PCIe)       Recompute on TPU MXU
        Promote back to HBM Cache       Store in HBM Cache
```

---

## 3. Options and Defaults

| Value | Storage Size | Context |
|---|---|---|
| `100_000_000_000` (Default: 100 GB) | 100 Gigabytes | Standard Cloud TPU VM hosts with 128GB–512GB host RAM |
| `20_000_000_000` (20 GB) | 20 Gigabytes | Constrained CPU host environments |
| `500_000_000_000` (500 GB) | 500 Gigabytes | Heavy document/RAG agent serving with huge prompt catalogs |

---

## 4. Interactions

- **`enable_prefix_caching`**: Must be `true`.
- **`prefix_caching_hbm_byte`**: Primary Tier 1 cache. DRAM acts as Tier 2 overflow.
- Host system memory: Setting this higher than physical host free RAM will cause Linux OOM-killer to terminate the Python runtime.

---

### One-line intuition
> **`prefix_caching_dram_byte` sets the host CPU memory ceiling for the secondary prefix cache tier, storing evicted KV caches to avoid full recomputation.**
