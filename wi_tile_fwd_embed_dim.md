
## GMM tiling parameters — embed dim tile for wi (up-projection) forward pass

`wi_tile_fwd_embed_dim` sets the tile size along the embedding/input-feature axis of the MoE up-projection matmul in the forward pass.

See [[gmm_tiling_params]] for full context on why GMM tiling exists, how all 18 tile params relate, and how to tune.

---

## Specific role of `embed_dim` tile

For the wi (up) projection, shape is `[moe_expert_input_dim, base_moe_mlp_dim]`. The `embed_dim` tile controls how much of the `moe_expert_input_dim` (input feature) axis is processed per tile:

```text
Larger embed_dim tile → more registers needed → may evict from L1 cache
                      → but higher arithmetic intensity per tile
Smaller tile → more tiles → more overhead, but better cache fit
```

---

## Default

```yaml
wi_tile_fwd_embed_dim: 1024
```

Confirmed in `base.yml`.

---

### One-line intuition

> **`wi_tile_fwd_embed_dim` tiles the embedding-axis of the MoE up-projection forward GMM — part of the 18-parameter GMM tiling grid; the default `1024` works well for standard embedding sizes.**
