
## GMM tiling parameters — MLP dim tile for wi (up-projection) forward pass

`wi_tile_fwd_mlp_dim` sets the tile size along the MLP hidden dimension axis of the MoE up-projection matmul in the forward pass.

See [[gmm_tiling_params]] for full context on why GMM tiling exists, how all 18 tile params relate, and how to tune.

---

## Specific role of `mlp_dim` tile

For the wi (up) projection, shape is `[moe_expert_input_dim, base_moe_mlp_dim]`. The `mlp_dim` tile controls how much of the output `base_moe_mlp_dim` axis is computed per tile:

```text
Larger mlp_dim tile → more output elements per tile → better reuse of input activations
                    → but more HBM pressure for the output buffer
Smaller tile → more tiles needed → lower HBM pressure, but more compute overhead
```

---

## Default

```yaml
wi_tile_fwd_mlp_dim: 1024
```

Confirmed in `base.yml`.

---

### One-line intuition

> **`wi_tile_fwd_mlp_dim` tiles the MLP hidden-dimension axis of the MoE up-projection forward GMM — controls output buffer size per tile; default `1024` is suitable for most expert sizes.**
