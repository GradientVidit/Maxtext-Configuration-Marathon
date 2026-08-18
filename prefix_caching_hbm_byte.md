## 1. Why does `prefix_caching_hbm_byte` exist?

When prefix caching is active, KV activations must be retained in memory so future requests can skip prompt evaluation. On-chip TPU High Bandwidth Memory (HBM) offers terabytes-per-second memory bandwidth, allowing near-instantaneous KV loading with zero PCIe/host transfer latency.

However, HBM is strictly limited (e.g., 16GB–32GB on TPU v5e/v6e, 32GB–128GB on TPU v4/v5p). Without an explicit memory ceiling, prefix caches could grow unbounded and trigger fatal accelerator Out-Of-Memory (OOM) errors.

```text
TPU HBM Memory Map:
┌─────────────────────────────────────────────────────────────┐
│ Model Weights & Graph Executables                           │ (~60-70%)
├─────────────────────────────────────────────────────────────┤
│ Active Request Dynamic KV Cache                             │ (~20-25%)
├─────────────────────────────────────────────────────────────┤
│ Prefix Cache HBM Pool [prefix_caching_hbm_byte = 10GB]      │ (~10-15%)
└─────────────────────────────────────────────────────────────┘
```

`prefix_caching_hbm_byte` sets the strict upper memory bound in bytes allocated directly in TPU HBM for storing reusable prefix KV caches.

---

## 2. Mechanics & Eviction

```text
New Prefix KV Computed
         │
         ▼
Check current HBM cache usage
         │
    ┌────┴────────────────────────┐
    │ Usage + Size <= Limit       │ Usage + Size > Limit
    ▼                             ▼
Store in HBM Pool            Evict LRU entry to DRAM (prefix_caching_dram_byte)
                             or drop if DRAM full -> Store new KV in HBM
```

---

## 3. Options and Values

| Value | Storage Size | Target Hardware & Topology |
|---|---|---|
| `10_000_000_000` (Default: 10 GB) | 10 Gigabytes | Standard multi-chip slices (TPU v4/v5p pods with ample HBM) |
| `2_000_000_000` (2 GB) | 2 Gigabytes | Low-HBM single-chip setups (e.g. TPU v5e-1x with 16GB total) |
| `30_000_000_000` (30 GB) | 30 Gigabytes | Dedicated serving nodes on TPU v5p/v6e with large system prompt catalogs |

---

## 4. Parameter Interactions

- **`enable_prefix_caching`**: Must be `true` for this parameter to have any effect.
- **`prefix_caching_dram_byte`**: Acts as the secondary spillover tier when this HBM budget is exceeded.
- **`per_device_batch_size` / `max_target_length`**: Dictates the primary dynamic KV cache pool; HBM prefix caching must fit in the remaining headroom.

---

## 5. Practical Tuning Guidelines

- Calculate your total available HBM per chip: `Total_HBM - Weights_Per_Chip - Dynamic_KV_Buffer - 2GB_Safety_Margin`.
- Allocate the remaining headroom to `prefix_caching_hbm_byte`. If set too high, concurrent large-batch requests will crash with `ResourceExhaustedError: Out of memory`.

---

### One-line intuition
> **`prefix_caching_hbm_byte` defines the maximum HBM memory budget in bytes reserved for caching reusable KV prefixes directly on TPU accelerator memory.**
