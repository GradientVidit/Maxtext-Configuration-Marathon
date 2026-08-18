## 1. Why does it exist?

Fully Sharded Data Parallelism (FSDP) shards model weights, gradients, and optimizer states across devices, requiring an **All-Gather** before each layer's forward/backward compute and a **Reduce-Scatter** for gradients.

While FSDP is usually confined to fast intra-slice interconnects (`ici_fsdp_parallelism`), extremely large models (e.g. 500B+ to 1T+ parameters) might exceed the High Bandwidth Memory (HBM) capacity of a single slice even after activation rematerialization and ICI sharding.

```text
Cross-Slice FSDP:
  Slice 0 (1/2 Weights) ───[ DCN All-Gather per layer ]─── Slice 1 (1/2 Weights)
```

`dcn_fsdp_parallelism` extends the FSDP sharding axis across multiple TPU slices over the Data Center Network.

---

## 2. Options & Trade-offs

| Value | Behavior |
|---|---|
| `1` (default) | FSDP sharding is confined entirely within slices (over ICI); no cross-slice FSDP weight gathers over DCN. |
| Integer $> 1$ | Extends FSDP across `N` slices. Reduces per-device HBM footprint further, but introduces cross-slice All-Gather latency. |

Default in `base.yml`:
```yaml
dcn_fsdp_parallelism: 1
```

---

## 3. Communication vs. Memory Trade-Off

```text
dcn_fsdp_parallelism: 1  (Recommended for almost all runs)
  - Zero cross-slice weight all-gather overhead.
  - Slices communicate only during gradient sync.

dcn_fsdp_parallelism: > 1 (Only for giant models that cannot fit in a single slice)
  - Saves HBM by partitioning weights across slices.
  - Cost: Every transformer layer must wait for cross-slice DCN All-Gather.
```

---

## 4. Interactions with Related Parameters

- **`pipeline_fsdp_ag_once`**: When combined with pipeline parallelism, all-gathers weights once per repeat to mitigate DCN all-gather latency.
- **`ici_fsdp_parallelism`**: Configures intra-slice FSDP.

---

### One-line intuition

> **`dcn_fsdp_parallelism` extends parameter sharding across separate TPU slices over the datacenter network, trading cross-slice communication latency for expanded collective HBM capacity.**
