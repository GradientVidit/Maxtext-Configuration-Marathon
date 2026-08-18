## 1. Why it exists: reducing Python dispatch latency and PyTree overhead

In a standard transformer implementation, the Key-Value (KV) cache is structured as a **nested PyTree of per-layer tensors**:

```text
Unstacked KV Cache Layout (stack_prefill_result_cache: false):
PyTree List with N_layers distinct arrays:
  Layer 0:  Array [Batch, Seq, Heads, D_kv]  (separate memory buffer)
  Layer 1:  Array [Batch, Seq, Heads, D_kv]  (separate memory buffer)
  ...
  Layer 79: Array [Batch, Seq, Heads, D_kv]  (separate memory buffer)

──> Total: 80 independent JAX array handles, 80 buffer allocations, 80 network transfers.

Stacked KV Cache Layout (stack_prefill_result_cache: true):
Single unified tensor along a new "layers" axis:
  Unified Cache: Array [Layers, Batch, Seq, Heads, D_kv] (1 contiguous memory buffer)

──> Total: 1 JAX array handle, 1 memory allocation, 1 bulk network transfer.
```

When running deep models (e.g., 32 to 80 layers) in high-throughput serving or **disaggregated inference**:
1. Managing 80 individual layer arrays incurs significant Python runtime overhead (PyTree traversing, unflattening, and garbage collection).
2. Transferring the KV cache from the `prefill_slice` to the `generate_slice` over DCN requires initiating 80 individual RPC buffer transfers rather than a single contiguous DMA bulk copy.

`stack_prefill_result_cache` packs the output KV caches across all transformer layers into a single stacked tensor during prefill, eliminating Python dispatch latency and streamlining inter-slice network transfers.

---

## 2. Mechanics: tensor aggregation and slicing

When `stack_prefill_result_cache: true`:

```text
                       Prompt Prefill Execution
                                   │
                                   ▼
                Per-layer KV Cache Generated During Scan
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │   Stack across layers dimension:                    │
        │   stacked_cache = jnp.stack(layer_caches, axis=0)   │
        │   Shape: [N_layers, Batch, Seq, Heads, D_kv]        │
        └──────────────────────────┬──────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   Single-Slice Serving                      Disaggregated Serving
┌─────────────────────────────┐           ┌─────────────────────────────┐
│ Fast Python PyTree return   │           │ Single zero-copy DMA packet │
│ Minimal runtime overhead    │           │ transmitted over DCN        │
└─────────────────────────────┘           └─────────────────────────────┘
```

1. During the prompt forward pass (typically inside `jax.lax.scan` over layers), the generated Key and Value activations are retained in a stacked layout.
2. In disaggregated mode, the prefill worker transfers the single contiguous `[Layers, ...]` buffer across the high-speed network.
3. The receiving decode worker unpacks or indexes into the stacked buffer during autoregressive decoding.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
stack_prefill_result_cache: false
```

| Value | KV Cache Structure | PyTree Array Count | Inter-Slice Transfer Overhead | Recommended For |
|---|---|---|---|---|
| `false` (default) | List / Dict of per-layer tensors | $2 \times N_{\text{layers}}$ arrays | High (multiple roundtrip network descriptors) | Standard training loops, simple single-instance debugging. |
| `true` | Single stacked multi-layer array | 2 arrays (stacked K, stacked V) | Minimal (single bulk DMA transfer) | **Disaggregated inference serving**, high-concurrency production deployments on GKE / JetStream. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                stack_prefill_result_cache                 │
└─────────────┬───────────────────────────────┬─────────────┘
              │ (when true)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Interacts directly with:                                  │
│ - prefill_cache_axis_order / ar_cache_axis_order          │
│ - inference_server (disaggregated servers)                │
│ - prefill_slice & generate_slice                          │
│ - scan_layers                                             │
└───────────────────────────────────────────────────────────┘
```

- **`prefill_slice` & `generate_slice`**: Stacking is the cornerstone optimization for cross-slice KV cache transfer latency between prefill and decode TPU pods.
- **`scan_layers`**: When layers are scanned via `jax.lax.scan`, layer activations are naturally stacked along the scan leading axis, making `stack_prefill_result_cache: true` a zero-overhead alignment.
- **`prefill_cache_axis_order`**: Controls the ordering of the inner dimensions within each layer slice of the stacked tensor.

---

## 5. Practical Scenarios & Failure Modes

### Disaggregated Serving with JetStream
When configuring high-throughput disaggregated serving across TPU slices:
```yaml
inference_server: "ExperimentalMaxtextDisaggregatedServer_8"
prefill_slice: "v5p-32"
generate_slice: "v5e-64"
stack_prefill_result_cache: true
```
Enabling cache stacking drops the end-to-end handoff latency between the prefill cluster and the generate cluster by over 60%, removing Python GIL bottlenecks and network fragmentation.

### What breaks if misconfigured:
- **Custom decoding loop shape mismatches**: If downstream custom code expects a Python list of layer dicts instead of an indexed multi-dimensional tensor, indexing operations will raise `KeyError` or `IndexError`.

---

### One-line intuition

> **`stack_prefill_result_cache` aggregates the KV caches of all layers into a single contiguous multi-dimensional tensor, slashing Python dispatch overhead and accelerating cross-slice network transfers in disaggregated serving.**
