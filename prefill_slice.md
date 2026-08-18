## 1. Why it exists: hardware specialization in disaggregated serving

In modern LLM inference systems, the **prefill stage** has fundamentally different hardware resource requirements than the **decode stage**:

```text
Prefill Stage Requirements:
┌─────────────────────────────────────────────────────────┐
│ High Compute Density (TFLOP/s)                          │
│ Large Matrix-Matrix Multiplications (GEMMs)             │
│ Short execution burst per request                       │
│ ──> Optimized for larger chips or compute-dense slices  │
└─────────────────────────────────────────────────────────┘

Decode Stage Requirements:
┌─────────────────────────────────────────────────────────┐
│ High Memory Bandwidth (GB/s per chip) & High HBM Size   │
│ Matrix-Vector Multiplications (GEMVs)                   │
│ Long-running step-by-step iteration                     │
│ ──> Optimized for distributed slices with wide memory   │
└─────────────────────────────────────────────────────────┘
```

In a **disaggregated serving** topology, rather than forcing both stages onto the same TPU hardware type, the cluster splits prompt processing and token generation across distinct physical slices. 

`prefill_slice` defines the TPU slice accelerator type (e.g., `"v5e-16"`, `"v5p-32"`, `"v6e-16"`) provisioned specifically to execute the prefill phase.

---

## 2. Mechanics: disaggregated pipeline routing

When MaxText runs with a disaggregated inference server, the cluster topology is partitioned into two pools:

```text
 Client Request: [Prompt Tokens]
               │
               ▼
 ┌───────────────────────────────────────────────────────────┐
 │               Prefill Pool (`prefill_slice`)              │
 │                                                           │
 │  - TPU Type: e.g. "v5p-32" or "v6e-16"                    │
 │  - Compute: Full prompt parallel attention                │
 │  - Produces: Initial Logits + Populated KV Cache          │
 └─────────────────────────────┬─────────────────────────────┘
                               │
                KV Cache Transfer (via DCN/gRPC)
                               │
                               ▼
 ┌───────────────────────────────────────────────────────────┐
 │              Decode Pool (`generate_slice`)               │
 │                                                           │
 │  - TPU Type: e.g. "v5e-16"                                │
 │  - Memory: Large batch size KV Cache retention            │
 │  - Executes: Iterative autoregressive token generation    │
 └───────────────────────────────────────────────────────────┘
```

1. The gateway routes incoming prompts directly to the prefill workers configured on `prefill_slice`.
2. The prefill slice compiles the prompt ingestion graph matched to its local accelerator mesh.
3. Once the initial token is sampled and the KV cache is materialized, the prefill worker serializes the KV cache state and transmits it over data center networking (DCN) to a decode worker in the `generate_slice` pool.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
prefill_slice: "v5e-16"
```

| Value | Accelerator Hardware Sizing | Use Case |
|---|---|---|
| `"v5e-16"` (default) | 16 TPU v5e TensorCores (2 hosts) | Cost-optimized prefill for small-to-medium models (e.g. 7B/8B). |
| `"v5p-16"` / `"v5p-32"` | 16–32 TPU v5p chips (high-power MXUs) | High-performance enterprise serving where Time-To-First-Token (TTFT) SLA must stay sub-50ms for long prompts. |
| `"v6e-16"` / `"v6e-32"` | Next-gen Trillium (TPU v6e) chips | High-throughput, compute-dense prefill serving with updated matrix engines. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                       prefill_slice                       │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│generate_slice             │   │inference_server           │
│Defines the companion      │   │Must be set to a           │
│decode slice type.         │   │disaggregated server class.│
└───────────────────────────┘   └───────────────────────────┘
              │
              ▼
┌───────────────────────────┐
│stack_prefill_result_cache │
│Optimizes KV cache layout  │
│for inter-slice transfer.  │
└───────────────────────────┘
```

- **`generate_slice`**: Symmetrically defines the target TPU slice type for autoregressive decoding.
- **`inference_server`**: Must be configured to a disaggregated serving engine (e.g., `ExperimentalMaxtextDisaggregatedServer_8`).
- **`stack_prefill_result_cache`**: Strongly recommended (`true`) to concatenate layer caches into a contiguous memory buffer before transferring across slices.
- **`prefill_cache_axis_order`**: Dictates the physical layout of the cache tensors generated on the prefill slice.

---

## 5. Practical Scenarios & Failure Modes

### Designing Asymmetric Slices for Maximum ROI
A common production architecture is pairing **compute-dense prefill slices** with **cost-effective decode slices**:
- **Prefill Slice**: `"v5p-32"` (Maximum TFLOPs to chew through 4k–32k context prompts instantly).
- **Generate Slice**: `"v5e-64"` (Wider aggregate memory capacity to sustain large concurrent generation batches at low cost).

### What breaks if misconfigured:
- **Topology Mismatch**: If `prefill_slice` specifies a slice shape that cannot accommodate the model's tensor/pipeline parallelism degree, JAX will fail with mesh creation and tensor partition errors during initialization.
- **DCN Bandwidth Bottlenecks**: If the prefill slice generates large KV caches faster than the network can transmit them to the generate slice, prefill workers will block on I/O.

---

### One-line intuition

> **`prefill_slice` defines the physical TPU accelerator topology allocated for prompt ingestion in disaggregated serving, enabling compute-dense hardware selection tailored for low Time-To-First-Token.**
