
## GMM tiling parameters — batch/sequence tile for wi (up-projection) forward pass

`wi_tile_fwd_batch_seq` sets the tile size along the token/batch-sequence axis of the MoE up-projection matmul in the forward pass.

See [[gmm_tiling_params]] for full context on why GMM tiling exists, how all 18 tile params relate to each other, and how to approach tuning.

---

## Specific role of `batch_seq` tile

The MoE GMM operates on a batch of tokens routed to each expert. The `batch_seq` dimension = number of tokens per expert. Tiling along this axis controls how many tokens are processed together in each tile:

```text
wi GMM shape: [tokens_per_expert, moe_expert_input_dim] × [moe_expert_input_dim, base_moe_mlp_dim]

batch_seq tile: T tokens processed per tile
→ smaller T: more tiles, more potential parallelism but more overhead
→ larger T: fewer tiles, higher arithmetic intensity per tile
```

---

## Default

```yaml
wi_tile_fwd_batch_seq: 512
```

Confirmed in `base.yml`.

---

### One-line intuition

> **`wi_tile_fwd_batch_seq` sets the token-count tile size for the MoE up-projection forward-pass GMM — part of the 18-parameter GMM tiling grid; profile first, then sweep against `{256, 512, 1024}`.**
