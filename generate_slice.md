## 1. Why it exists: memory-capacity optimization for the autoregressive decode phase

During large language model inference, the **token generation (decode) phase** is fundamentally constrained by **memory bandwidth and total HBM capacity**:

```text
Autoregressive Step Computational Profile:
┌────────────────────────────────────────────────────────────────────────┐
│ For each new token:                                                    │
│ 1. Load model weights from HBM (e.g. 70GB for 70B model in INT8/FP8)   │
│ 2. Load KV Cache for ALL active sequences from HBM                     │
│ 3. Execute light Matrix-Vector multiplications (GEMV)                  │
│ 4. Emit next token probability distribution                            │
└────────────────────────────────────────────────────────────────────────┘
```

Because decode executes a single token step at a time across multiple batched requests, the arithmetic intensity (FLOPs per byte loaded) is extremely low compared to prefill. The primary limiting factors are:
1. **Total HBM capacity**: Determines the maximum concurrent context tokens the slice can hold before rejecting new requests.
2. **Memory bandwidth**: Determines the per-token latency (Inter-Token Latency, ITL).

In a **disaggregated serving** architecture, `generate_slice` defines the TPU slice accelerator type (e.g., `"v5e-16"`, `"v5e-64"`, `"v6e-16"`) provisioned specifically to host the KV cache and execute autoregressive decoding.

---

## 2. Mechanics: decode worker pool lifecycle

In disaggregated serving, nodes configured under `generate_slice` form a dedicated generation pool:

```text
               Incoming KV Cache from `prefill_slice`
                                 │
                                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │               Decode Worker Pool (`generate_slice`)           │
 │                                                               │
 │  ┌─────────────────────────────────────────────────────────┐  │
 │  │ Persistent Dynamic KV Cache Manager                     │  │
 │  │ Slots: [Req 1 Cache] [Req 2 Cache] ... [Req K Cache]   │  │
 │  └────────────────────────────┬────────────────────────────┘  │
 │                               │                               │
 │                               ▼                               │
 │  ┌─────────────────────────────────────────────────────────┐  │
 │  │ Autoregressive Step Graph (JAX/XLA)                     │  │
 │  │ - Multi-query / Grouped-query attention step kernel     │  │
 │  │ - Continuous batching scheduler                         │  │
 │  │ - Token sampler (Greedy / Top-p / Nucleus)              │  │
 │  └────────────────────────────┬────────────────────────────┘  │
 └───────────────────────────────┼───────────────────────────────┘
                                 │
                                 ▼
                     Streamed Output Tokens to Client
```

1. The decode workers initialize the model weights across the device mesh defined by `generate_slice`.
2. When a prompt is completed on `prefill_slice`, the materialized KV cache is received and placed into an available cache slot in the generate slice's HBM.
3. The generate slice executes continuous decoding steps until an `<EOS>` token is emitted or `max_target_length` is reached.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
generate_slice: "v5e-16"
```

| Value | Hardware Characteristics | Recommended Deployment |
|---|---|---|
| `"v5e-16"` (default) | 16 TPU v5e chips, 256 GB aggregate HBM | Cost-efficient serving for moderate batch sizes on small models (7B–8B). |
| `"v5e-64"` / `"v5e-128"` | Wide TPU v5e pod slice (1TB–2TB aggregate HBM) | Massive concurrent batching for enterprise serving, maximizing throughput per dollar. |
| `"v6e-16"` / `"v6e-32"` | TPU v6e (Trillium) slice with high-bandwidth HBM | Low inter-token latency (ITL) serving for interactive agent applications. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                      generate_slice                       │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│prefill_slice              │   │ar_cache_axis_order        │
│Defines the upstream       │   │Controls physical memory   │
│prefill hardware pool.     │   │layout for the decode cache│
└───────────────────────────┘   └───────────────────────────┘
              │
              ▼
┌───────────────────────────┐
│multi_sampling /           │
│return_log_prob            │
│Executed on generate slice │
│during token generation.   │
└───────────────────────────┘
```

- **`prefill_slice`**: Symmetrically defines the prompt processing slice type.
- **`ar_cache_axis_order`**: Configures the tensor layout of the autoregressive KV cache stored in the generate slice's HBM.
- **`multi_sampling` & `return_log_prob`**: Post-processing operations executed during decode steps on the generate slice.

---

## 5. Practical Scenarios & Failure Modes

### Scaling Prefill vs Decode Horizontally
In disaggregated clusters, traffic with long generations (e.g. coding agents generating 2000 output tokens) requires significantly more decode capacity than prefill capacity. You can configure:
- **1 Prefill Pool** (`prefill_slice: "v5p-16"`)
- **4 Decode Pools** (`generate_slice: "v5e-64"`)
This matches the 1:4 workload ratio without paying for expensive v5p chips during the memory-bound decode phase.

### What breaks if misconfigured:
- **HBM Exhaustion under High Concurrency**: If `generate_slice` is sized too small for the configured maximum concurrency, KV cache allocations will fail with an out-of-memory error when traffic spikes.
- **Sharding Incompatibilities**: Ensure the mesh dimension on `generate_slice` is compatible with the model's head count ($N_{\text{heads}}$ or $N_{\text{kv\_heads}}$ for GQA).

---

### One-line intuition

> **`generate_slice` specifies the TPU accelerator topology dedicated to autoregressive token generation in disaggregated serving, sized to maximize HBM capacity and continuous batching concurrency.**
