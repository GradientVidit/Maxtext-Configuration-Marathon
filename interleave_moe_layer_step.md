## 1. Why does `interleave_moe_layer_step` exist?

Mixture-of-Experts (MoE) models can either make every feed-forward block sparse or interleave dense MLP layers with sparse MoE layers. Pure MoE architectures consume massive parameter memory and communication bandwidth, whereas hybrid dense/sparse architectures maintain high knowledge consolidation in dense layers while scaling capacity efficiently in sparse layers.

While MaxText supports arbitrary irregular layer patterns via `inhomogeneous_layer_cycle_interval`, `interleave_moe_layer_step` provides a straightforward periodic stride for Llama4 and related hybrid models.

```text
All MoE (interleave_moe_layer_step = 1):
[MoE] -> [MoE] -> [MoE] -> [MoE] -> [MoE] -> [MoE]

Interleaved MoE/Dense (interleave_moe_layer_step = 2):
[MoE] -> [Dense MLP] -> [MoE] -> [Dense MLP] -> [MoE] -> [Dense MLP]
```

`interleave_moe_layer_step` controls the stride at which MoE layers occur in the decoder stack.

---

## 2. Layer Construction Flow

```text
For layer_idx in range(base_num_decoder_layers):
    if (layer_idx % interleave_moe_layer_step == 0) and num_experts > 1:
        Instantiate Sparse MoE Layer (Routed Experts)
    else:
        Instantiate Standard Dense MLP Layer
```

---

## 3. Options and Defaults

| Value | Behavior | Architecture Example |
|---|---|---|
| `1` (Default) | Every layer is MoE (if `num_experts > 1`) | Mixtral 8x7B, DeepSeek-V2/V3, Qwen-MoE |
| `2` | Alternating layers: MoE -> Dense -> MoE -> Dense | Llama4 hybrid MoE, DBRX |
| `4` | 1 MoE layer followed by 3 Dense MLP layers | Compute-bound architectures prioritizing FLOP/memory balance |

---

## 4. Parameter Interactions

- **`num_experts`**: Must be $> 1$. If `num_experts: 1`, all layers remain dense regardless of this setting.
- **`base_moe_mlp_dim` vs `base_mlp_dim`**: MoE layers utilize `base_moe_mlp_dim`, while interleaved dense layers utilize `base_mlp_dim`.
- **`inhomogeneous_layer_cycle_interval`**: Overrides or generalizes this stepping pattern for complex non-uniform schedules.

---

## 5. Practical Pitfalls

- When training with pipeline parallelism (`ici_pipeline_parallelism > 1`), uneven interleaving can cause compute/parameter imbalances across pipeline stages if stages don't contain equal counts of MoE layers.

---

### One-line intuition
> **`interleave_moe_layer_step` defines the periodic layer interval for MoE blocks, allowing straightforward alternating mixtures of dense and sparse feed-forward layers.**
