
## 1. Why GMM tiling params exist

The MoE expert MLP computation is a **Grouped Matrix Multiply (GMM)** — a batched matmul where each group (expert) has a different number of rows (tokens), and the groups are processed together.

GMM kernels are tiled: they process the computation in blocks to exploit hardware cache hierarchies and vector units. The tile sizes directly affect:
- **Cache utilization**: tiles too large spill to HBM; too small leave hardware underutilized
- **Arithmetic intensity**: the ratio of FLOPs to bytes accessed
- **Parallelism**: smaller tiles = more parallel chunks; larger tiles = more work per chunk

The `wi_tile_fwd_*` and `wo_tile_fwd_*` parameters control these tile dimensions for the two expert projections:
- `wi` = the "up" projection (input → hidden dimension)
- `wo` = the "down" projection (hidden → output dimension)
- `fwd` = forward pass

---

## 2. The three tile dimensions

Each projection's tiling is specified along three axes:

```text
batch_seq    →  the number of tokens dimension (batch × sequence length routed to an expert)
embed_dim    →  the embedding/input dimension axis
mlp_dim      →  the MLP hidden (intermediate) dimension axis
```

For `wi` (up projection): shape is `[moe_expert_input_dim, base_moe_mlp_dim]`
- `embed_dim` tile = tile along `moe_expert_input_dim`
- `mlp_dim` tile = tile along `base_moe_mlp_dim`

For `wo` (down projection): shape is `[base_moe_mlp_dim, moe_expert_input_dim]`
- `embed_dim` tile = tile along `moe_expert_input_dim`
- `mlp_dim` tile = tile along `base_moe_mlp_dim`

---

## 3. Defaults (from base.yml)

```yaml
wi_tile_fwd_batch_seq: 512
wi_tile_fwd_embed_dim: 1024
wi_tile_fwd_mlp_dim:   1024

wo_tile_fwd_batch_seq: 512
wo_tile_fwd_embed_dim: 1024
wo_tile_fwd_mlp_dim:   1024
```

These are the same for both wi and wo. Reasonable defaults for large embedding dimensions.

---

## 4. Backend support

```text
megablox/JAX ragged-dot backend:
    → supports 6 configs: wi_tile_fwd_* and wo_tile_fwd_* only
    → backward-pass tiling (dlhs/drhs) is NOT tunable on this backend

Tokamax ragged-dot backend (use_tokamax_gmm=True):
    → supports all 18 configs (fwd + dlhs + drhs for both wi and wo)
```

The dlhs/drhs variants (backward-pass tiling) only matter when `use_tokamax_gmm=True`.

---

## 5. The 18-parameter tiling grid

```text
                wi_tile_          wo_tile_
             fwd  dlhs drhs     fwd  dlhs drhs
batch_seq  [ 512  512  512 ]  [ 512  512  512 ]
embed_dim  [1024 1024 1024 ]  [1024 1024 1024 ]
mlp_dim    [1024 1024 1024 ]  [1024 1024 1024 ]
```

- `fwd`  = forward pass matmul
- `dlhs` = backward pass, gradient w.r.t. left-hand-side operand
- `drhs` = backward pass, gradient w.r.t. right-hand-side operand

---

## 6. How to tune these

This is a benchmarking exercise, not a hand-tuning exercise. The right tile sizes depend on:
- Your hardware (TPU generation, HBM bandwidth)
- Your model shape (`base_moe_mlp_dim`, `moe_expert_input_dim`, batch × seq)
- Number of experts and token distribution per expert

Approach:
1. Profile the GMM kernel: is it compute-bound or memory-bound?
2. If memory-bound: increase tile sizes to improve arithmetic intensity
3. If compute-bound: tile sizes may be fine; look at parallelism
4. Sweep over `{256, 512, 1024}` per dimension and pick the fastest

---

## 7. When NOT to tune these

In most training runs, the default `512/1024/1024` tiles are reasonable. Only invest tuning effort when:
- GMM is measured as the bottleneck in a profile
- You're running with a non-standard model shape (very small/large `base_moe_mlp_dim`)
- You've switched to the Tokamax backend and want to tune backward-pass tiles

---

### One-line intuition

> **The `wi_tile_fwd_*` and `wo_tile_fwd_*` params control tile dimensions for the MoE GMM kernel's forward pass — profile-first, then sweep; the defaults are reasonable and these only matter when GMM is the measured bottleneck at your specific model and hardware shape.**
