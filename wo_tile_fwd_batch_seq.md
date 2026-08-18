
## GMM tiling parameters — wo (down-projection) forward pass tiles

`wo_tile_fwd_batch_seq`, `wo_tile_fwd_embed_dim`, and `wo_tile_fwd_mlp_dim` control tile sizes for the MoE down-projection (wo) matmul in the forward pass.

See [[gmm_tiling_params]] for full context on all 18 GMM tiling params and how to approach tuning.

---

## The wo (down) projection

```text
wo GMM shape: [tokens_per_expert, base_moe_mlp_dim] × [base_moe_mlp_dim, moe_expert_input_dim]
```

This is the inverse of the wi (up) projection: maps from MLP hidden dimension back to embedding dimension.

- `wo_tile_fwd_batch_seq`: tokens per tile (same role as wi equivalent)
- `wo_tile_fwd_embed_dim`: tile along output `moe_expert_input_dim` axis
- `wo_tile_fwd_mlp_dim`: tile along input `base_moe_mlp_dim` axis

---

## Defaults

```yaml
wo_tile_fwd_batch_seq: 512
wo_tile_fwd_embed_dim: 1024
wo_tile_fwd_mlp_dim:   1024
```

Confirmed in `base.yml`. Same as wi defaults.

---

## Why wi and wo may need different tuning

In SwiGLU-style experts, wo is the down-projection and processes the full MLP hidden dimension. If `base_moe_mlp_dim ≠ moe_expert_input_dim`, the optimal tile shapes for wi and wo may differ. In practice, tuning them together as a pair is usually sufficient.

---

### One-line intuition

> **`wo_tile_fwd_*` controls the three tiling dimensions of the MoE down-projection forward GMM — identical concept to the wi equivalents but for the wo matmul; tune as a pair with the wi tiles.**
