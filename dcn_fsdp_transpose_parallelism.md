## 1. Why does it exist?

2D-FSDP shards weight matrices along both their input and output dimensions across two orthogonal mesh axes: `fsdp` and `fsdp_transpose`.

When scaling 2D-FSDP across multi-slice clusters, `dcn_fsdp_transpose_parallelism` sets the degree of the transpose FSDP axis spanning over the Data Center Network.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | No 2D-FSDP transpose sharding over DCN. |
| Integer $> 1$ | Extends the transpose FSDP axis across `N` slices. |

Default in `base.yml`:
```yaml
dcn_fsdp_transpose_parallelism: 1
```

---

## 3. Interactions with Related Parameters

- **`use_2d_fsdp_sharding`**: Enables 2D-FSDP sharding logic.
- **`dense_fsdp_use_two_stage_all_gather`**: When both `fsdp` and `fsdp_transpose` participate, issues two sequential all-gather calls.
- **`ici_fsdp_transpose_parallelism`**: Companion intra-slice transpose axis.

---

### One-line intuition

> **`dcn_fsdp_transpose_parallelism` defines the cross-slice datacenter network degree for the orthogonal transpose axis in 2D-FSDP weight partitioning.**
