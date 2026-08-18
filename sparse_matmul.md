
## 1. Why this exists

MoE's fundamental challenge: after routing, you have a collection of expert MLPs that each see different subsets of tokens. Two fundamentally different compute strategies exist:

**Dense (padded) strategy:**
```text
for each expert:
    pad its token batch to some fixed capacity
    run a standard dense matmul
```
Wastes FLOPs on padding, but uses standard dense kernels.

**Sparse strategy:**
```text
pack all expert-token pairs into a ragged structure
run a single grouped/sparse matmul across all experts simultaneously
```
No padding waste, but requires specialized sparse kernels.

`sparse_matmul` selects between these two strategies.

---

## 2. What it controls

```yaml
sparse_matmul: true   # use sparse matmul for MoE expert computation
sparse_matmul: false  # use dense (padded) matmul
```

When `true`, MaxText uses the sparse matmul path (block-sparse GMM via MegaBlocks or equivalent). When `false`, falls back to dense padded execution.

---

## 3. Relationship to `megablox`

`sparse_matmul` is the higher-level "sparse vs dense" switch; `megablox` controls which specific sparse implementation is used.

```text
sparse_matmul=false     →  dense padded path (megablox irrelevant)
sparse_matmul=true      →  sparse path
  megablox=true         →    via MegaBlocks block-sparse GMM kernel
  megablox=false        →    via alternative sparse implementation
```

In production you almost always want both `true`.

---

## 4. Under the hood: `shard_map` vs `dense_matmul`

When `sparse_matmul=true`, MaxText uses a `shard_map`-based dispatch to route tokens through expert MLPs in a data-parallel fashion. When `sparse_matmul=false`, it uses the `dense_matmul` path, which is also what `moe_dispatch_no_expert_sharding` affects.

Note: `moe_dispatch_no_expert_sharding` only affects the dense path (`sparse_matmul=false`). The sparse/`shard_map` path is unaffected by that flag.

---

## 5. Options

| Value | Behavior |
|---|---|
| `true` (default) | Sparse matmul — no padding, handles ragged expert loads efficiently |
| `false` | Dense padded matmul — simpler, easier to debug, but wastes compute on token padding |

---

## 6. When to change it

**Leave at `true`** in production — padding overhead on imbalanced routing is real and can be substantial.

**Set to `false` for:**
- Debugging: dense path is more transparent, no ragged buffers, easier to profile
- Correctness verification: compare sparse and dense outputs
- Hardware where dense matmuls outperform sparse alternatives (rare, but possible on some GPU shapes)

---

## 7. Default

```yaml
sparse_matmul: true
```

Confirmed in `base.yml`.

---

### One-line intuition

> **`sparse_matmul` switches MoE expert compute between an efficient sparse (no-padding) grouped matmul and a simpler but wasteful dense padded matmul — keep it `true` unless debugging.**
