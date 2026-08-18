
## 1. What 2D FSDP sharding means

Standard FSDP (1D) shards MoE weight matrices across one mesh axis:

```text
weight shape: [num_experts, embed_dim, mlp_dim]
1D FSDP: shard along FSDP axis only
→ each device holds a slice of every weight along one dimension
```

2D FSDP extends this to **two mesh axes** — both the FSDP axis and the FSDP-transpose axis:

```text
2D FSDP: shard along (FSDP, FSDP_transpose) axes
→ each device holds a 2D tile of the weight matrix
→ more memory savings but requires two all-gathers to reconstruct
```

`use_2d_fsdp_sharding` enables this 2D sharding for MoE weight matrices.

---

## 2. What it controls

```yaml
use_2d_fsdp_sharding: false  # (default) 1D FSDP sharding
use_2d_fsdp_sharding: true   # 2D FSDP sharding on both FSDP and FSDP-transpose axes
```

---

## 3. The memory-communication tradeoff

```text
1D FSDP:
  memory: each device holds 1/N of each weight (N = FSDP degree)
  communication: one all-gather per forward pass

2D FSDP:
  memory: each device holds 1/(N × M) of each weight (M = FSDP-transpose degree)
  communication: two all-gathers per forward pass
```

2D sharding enables much larger effective model sizes per device at the cost of more communication collectives.

---

## 4. Interaction with `moe_fsdp_use_two_stage_all_gather`

When `use_2d_fsdp_sharding=True`, the weight reconstruction requires gathering over both axes. `moe_fsdp_use_two_stage_all_gather` controls whether this is done as:
- One combined all-gather (`moe_fsdp_use_two_stage_all_gather=False`)
- Two sequential all-gathers (`moe_fsdp_use_two_stage_all_gather=True`)

---

## 5. Default

```yaml
use_2d_fsdp_sharding: false
```

1D sharding is the baseline. Enable 2D when fitting the model in memory requires more aggressive parameter sharding.

---

## 6. When to use it

**Fitting very large MoE models:** when 1D FSDP isn't enough to fit the expert weights across the device mesh.

**High degree of FSDP-transpose parallelism available:** when the topology provides a strong FSDP-transpose axis to shard along.

**Not for small models:** 2D sharding adds communication overhead that isn't worth it when 1D suffices.

---

### One-line intuition

> **`use_2d_fsdp_sharding` shards MoE expert weights across both FSDP and FSDP-transpose mesh axes, reducing per-device memory by an additional factor at the cost of two all-gathers per forward pass — for models where 1D FSDP alone can't fit the expert weights.**
