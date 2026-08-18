
## 1. The MoE expert GEMM sharding ambiguity

In MaxText's mesh-sharded execution, each tensor dimension maps to a mesh axis. For MoE expert GEMMs in the dense matmul path, there's an ambiguity: where does the `expert` mesh axis belong?

Two options:

**Option A — expert on batch dimension:**
```text
expert GEMM batch dim = [data_batch, expert]
→ "activation_batch_moe" tensor dimension
→ expert axis is treated like another batch dimension
```

**Option B — expert peeled off (sharding preserved):**
```text
expert GEMM batch dim = [data_batch]
expert axis is separate → true expert parallelism
→ AllToAll communication between expert shards
```

`moe_dispatch_no_expert_sharding` chooses between these.

---

## 2. What it controls

```yaml
moe_dispatch_no_expert_sharding: false  # (default) expert on batch dim (activation_batch_moe)
moe_dispatch_no_expert_sharding: true   # peel expert axis off, use AllToAll-based expert-parallel GEMM
```

When `false`, the `expert` mesh axis stays on the batch dimension of the activation tensor. The GEMM sees a combined `[batch × expert]` batch.

When `true`, the `expert` axis is kept separate from the batch dim. Each device holds a subset of experts and processes only those — true expert parallelism with AllToAll.

---

## 3. This only affects the dense path

**`moe_dispatch_no_expert_sharding` has NO effect on:**
- The sparse matmul path (`sparse_matmul=True, megablox=True`)
- The `shard_map`-based dispatch

It only applies to the `dense_matmul` MoE path (`sparse_matmul=False`).

```text
sparse_matmul=True   → shard_map path → unaffected
sparse_matmul=False  → dense_matmul path → this flag matters
```

---

## 4. When `true` makes sense

Use `moe_dispatch_no_expert_sharding=True` when:
- Running the dense matmul path (`sparse_matmul=False`)
- Expert parallelism is the bottleneck and you want true EP-style AllToAll dispatch
- The `expert` dimension needs to be properly parallel, not merged into the batch

---

## 5. Default

```yaml
moe_dispatch_no_expert_sharding: false
```

Most production MoE runs use `sparse_matmul=True`, where this flag is irrelevant anyway.

---

## 6. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `sparse_matmul` | Must be `False` for this flag to have any effect |
| `megablox` | Same — only dense path is affected |
| `use_ring_of_experts` | Ring path is also unaffected; only dense path |

---

### One-line intuition

> **`moe_dispatch_no_expert_sharding` controls whether the expert mesh axis is peeled off the GEMM batch dimension (enabling AllToAll-based true expert parallelism) or kept merged with batch — only applies to the dense matmul path and irrelevant when using sparse matmul.**
