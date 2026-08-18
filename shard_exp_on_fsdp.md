
## 1. The expert weight memory problem

MoE expert weight matrices have shape `[num_experts, moe_expert_input_dim, base_moe_mlp_dim]`. With many experts, this is a lot of parameters. They need to be sharded across the device mesh to fit in memory.

Standard FSDP shards the parameter across the FSDP axis (usually the data-parallel axis). For MoE experts, there's an alternative: shard along the **expert dimension** itself using FSDP — meaning each FSDP shard holds a subset of experts rather than a slice of each expert's weights.

`shard_exp_on_fsdp` enables this expert-dimension sharding on the FSDP axis.

---

## 2. What it does

```yaml
shard_exp_on_fsdp: false  # (default) standard FSDP sharding (slices each expert's weights)
shard_exp_on_fsdp: true   # shard expert dimension on FSDP axis (each shard owns complete experts)
```

When `true`, each FSDP worker owns a complete subset of experts rather than a slice of all experts' weights.

```text
standard FSDP (shard_exp_on_fsdp=false):
  device_0: owns columns 0:dim/4 of ALL 64 experts
  device_1: owns columns dim/4:dim/2 of ALL 64 experts
  ...

expert-dimension FSDP (shard_exp_on_fsdp=true):
  device_0: owns ALL weights of experts 0-15
  device_1: owns ALL weights of experts 16-31
  ...
```

---

## 3. The constraint

```yaml
# Recommended only when num_experts is a multiple of FSDP parallelism degree
shard_exp_on_fsdp: true
```

MaxText explicitly documents this: `num_experts` must be divisible by the FSDP degree for clean sharding. Odd ratios lead to load imbalance.

---

## 4. Why this can be better

When each device owns complete expert weight sets:
- All-gather for FSDP reconstruction is along a well-structured axis
- Expert-parallel operations naturally align with which experts each device holds
- May reduce cross-device communication vs. the standard slice-per-device approach

---

## 5. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `use_2d_fsdp_sharding` | 2D sharding can combine `shard_exp_on_fsdp` with FSDP-transpose sharding |
| `moe_fsdp_use_two_stage_all_gather` | Two-stage gather is more relevant when `shard_exp_on_fsdp=True` + `use_2d_fsdp_sharding=True` |
| `num_experts` | Must be divisible by FSDP degree when this is `True` |

---

### One-line intuition

> **`shard_exp_on_fsdp` shards the expert dimension (not weight slices) across FSDP workers, so each device owns complete experts — cleaner than slicing each expert's weights, but requires `num_experts` to be divisible by the FSDP degree.**
