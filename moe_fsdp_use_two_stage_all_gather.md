
## 1. The FSDP + expert sharding problem

In large-scale training, MoE expert weight matrices are often sharded across FSDP axes to fit in memory. When weights are sharded on **both** the FSDP and FSDP-transpose axes simultaneously (2D FSDP sharding), an all-gather is needed to reconstruct the full weight before the expert GEMM.

Two ways to execute this all-gather:

**Option A — Single combined all-gather:**
```text
shard on (FSDP, FSDP_transpose)
→ one all-gather over the combined sharding
→ single collective, but may be large
```

**Option B — Two sequential all-gathers:**
```text
shard on (FSDP, FSDP_transpose)
→ all-gather over FSDP axis first
→ all-gather over FSDP_transpose axis second
→ two smaller collectives
```

`moe_fsdp_use_two_stage_all_gather` selects Option B.

---

## 2. What it controls

```yaml
moe_fsdp_use_two_stage_all_gather: false  # (default) single combined all-gather
moe_fsdp_use_two_stage_all_gather: true   # two separate all-gather calls
```

---

## 3. Precondition

Only relevant when:
- `use_2d_fsdp_sharding: true` (weights sharded on both FSDP and FSDP-transpose axes)
- MoE is actually enabled (`num_experts > 1`)

With 1D FSDP sharding, only one all-gather is needed anyway and this flag has no effect.

---

## 4. Why two stages might be better

- **Network topology:** some interconnects have higher bandwidth between certain axis pairs; two smaller gathers over the right topology can outperform one large gather
- **Pipeline overlap:** two sequential gathers create natural pipeline points between them that XLA may use for overlap
- **Memory peak:** a single large gather materializes the full weight at once; two-stage may keep peak memory lower if the second gather can be deferred

---

## 5. Default

```yaml
moe_fsdp_use_two_stage_all_gather: false
```

Single gather is the stable default. Enable when profiling shows the all-gather is the bottleneck with 2D sharding.

---

### One-line intuition

> **`moe_fsdp_use_two_stage_all_gather` replaces a single combined FSDP all-gather for 2D-sharded MoE weights with two sequential smaller all-gathers — a network-topology optimization for `use_2d_fsdp_sharding=True` configurations.**
