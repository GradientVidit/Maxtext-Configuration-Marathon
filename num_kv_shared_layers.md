## 1. Why does `num_kv_shared_layers` exist?

In deep autoregressive transformers, empirical analysis shows that trailing decoder layers (those near the output of the network) develop highly similar attention patterns. Computing independent Key ($K$) and Value ($V$) projections in these final layers yields diminishing returns while consuming significant parameter storage and inference KV cache memory.

**Trailing Layer KV Sharing** allows the final $N$ decoder layers of a model to skip instantiating their own $W_k, W_v$ projections. Instead, they **reuse the $K$ and $V$ activations from the preceding non-shared layer of the same attention type**:

```text
Example: 8-Layer Model with num_kv_shared_layers = 4 (Layers 0-3 Unshared, Layers 4-7 Shared)

Layer 0 (Global Attn)  ──> Has own K, V projections ──> [Produces K0, V0]
Layer 1 (Global Attn)  ──> Has own K, V projections ──> [Produces K1, V1]
Layer 2 (Global Attn)  ──> Has own K, V projections ──> [Produces K2, V2]
Layer 3 (Global Attn)  ──> Has own K, V projections ──> [Produces K3, V3] <── Last Non-Shared
──────────────────────────────────────────────────────────────────────────
Layer 4 (Global Attn)  ──> No K/V weights! Reuses K3, V3 from Layer 3
Layer 5 (Global Attn)  ──> No K/V weights! Reuses K3, V3 from Layer 3
Layer 6 (Global Attn)  ──> No K/V weights! Reuses K3, V3 from Layer 3
Layer 7 (Global Attn)  ──> No K/V weights! Reuses K3, V3 from Layer 3
```

---

## 2. Options & Defaults

| Value | Behavior | Parameter & KV Cache Impact |
|---|---|---|
| `0` | Disabled. Every layer computes its own $K, V$ projections. | **Default**. Standard transformer design. |
| Any integer $> 0$ (e.g. `4`, `8`) | The last $N$ layers reuse $K, V$ from the last unshared layer of matching attention type. | Saves $2 	imes d_{model} 	imes d_{kv}$ parameters per shared layer; eliminates new KV cache allocation for those layers. |

Default in `base.yml`: `0`

---

## 3. The "Same Attention Type" Matching Constraint

In hybrid models alternating between **sliding window** and **global** attention (e.g. Gemma 4), trailing shared layers must reuse KVs from a layer with the **identical attention pattern**:

```text
Layer 4 (Sliding) ──> Reuses KV from Layer 2 (Sliding)  <-- NOT Layer 3!
Layer 5 (Global)  ──> Reuses KV from Layer 3 (Global)
```

MaxText enforces this type-matching constraint automatically so that sliding-window layers never accidentally inherit unbounded global KV representations.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[use_double_wide_mlp]] | Direct companion: when `use_double_wide_mlp: true`, KV-shared layers double their MLP hidden dimension to maintain parameter parity. |
| [[scan_layers]] | Must be set to `scan_layers: false` because shared layers have heterogeneous weight structures and inter-layer tensor dependencies. |
| [[attention_type]] | Respects attention type boundaries during KV resolution. |

---

## 5. Practical Scenarios

- **Gemma 4 Edge Architecture Pretraining:** Set `num_kv_shared_layers` to halve the KV cache memory footprint during on-device inference without sacrificing early-layer representational depth.
- **Inference Optimization on Memory-Constrained Accelerators:** Reduces the number of active KV cache tensors by $N$, allowing larger batch sizes during generation.

---

### One-line intuition

> **`num_kv_shared_layers` eliminates $K/V$ projection weights in the trailing $N$ decoder layers by reusing cached Keys and Values from previous layers of matching attention type.**
