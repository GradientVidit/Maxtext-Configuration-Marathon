
## 1. Why does `base_num_decoder_layers` exist?

A transformer's total parameter count and computational cost scales linearly with the number of layers. Depth and width are the two primary knobs for scaling:

```text
total_params ≈ num_layers × params_per_layer
            ≈ num_layers × (attn_params + mlp_params + norm_params)
```

`base_num_decoder_layers` controls **depth** — how many transformer blocks are stacked sequentially. More layers = more compute per forward pass, more parameters, and typically better reasoning at the cost of:
- Higher memory per step (more activations to store for backward)
- More pipeline stages (if using pipeline parallelism)
- Longer critical path for sequential execution

---

## 2. Default

```yaml
base_num_decoder_layers: 16
```

Combined with `base_emb_dim: 2048` and `base_mlp_dim: 7168`, 16 layers gives ~117M parameters total — a sanity-check/debugging scale.

---

## 3. Layer count by model scale

| Model | base_num_decoder_layers | Notes |
|---|---|---|
| Default (~117M) | 16 | Debugging scale |
| ~1B | 24–32 | Small production model |
| LLaMA 2 7B | 32 | Standard 7B |
| LLaMA 2 13B | 40 | |
| LLaMA 2 70B | 80 | |
| GPT-3 175B | 96 | |

---

## 4. Interaction with pipeline parallelism

When using pipeline parallelism, `base_num_decoder_layers` must be divisible by `num_pipeline_stages × num_layers_per_pipeline_stage`:

```text
num_decoder_layers = num_pipeline_stages
                   × num_layers_per_pipeline_stage
                   × num_pipeline_repeats
```

If your layer count doesn't divide cleanly, use `pipeline_parallel_layers` to pipeline only a subset of layers, with the remainder acting as data-parallel.

---

## 5. Interaction with `scan_layers`

MaxText uses `jax.lax.scan` to loop over layers instead of unrolling them (when `scan_layers: true`). This:
- Reduces compile time dramatically for large layer counts
- Uses O(1) HBM for the scan (layers reuse the same compiled code)
- Is the recommended default

Setting `scan_layers: false` unrolls all layers — valid for debugging or pipeline parallelism where each stage needs independent code.

---

## 6. Interaction with rematerialization

`remat_policy` determines which activations are saved vs. recomputed during backward pass. With more layers, the total activation memory scales proportionally — this is why deep models almost always need rematerialization (`remat_policy: "full"` recomputes everything, the default).

---

## 7. The depth vs. width tradeoff

Empirically (Chinchilla and related work):
- More layers → better reasoning and in-context learning
- More width (emb_dim/mlp_dim) → more factual knowledge storage
- Both are needed; pure width or pure depth scaling eventually plateaus

Most LLM scaling recipes keep the aspect ratio (depth/width) roughly constant as they scale.

---

### One-line intuition

> **`base_num_decoder_layers` is the depth of the transformer — it scales parameters, compute, and memory linearly, and must divide cleanly across pipeline stages if using pipeline parallelism.**
