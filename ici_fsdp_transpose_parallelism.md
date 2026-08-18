## 1. Why does it exist?

Standard 1D-FSDP shards weight matrices along a single dimension (e.g. rows). For massive weight matrices in large models, 2D-FSDP shards matrices along both input and output dimensions (rows and columns) across two orthogonal mesh axes: `fsdp` and `fsdp_transpose`.

```text
2D Weight Matrix Sharding:
                   Columns (fsdp_transpose axis)
                 ┌──────────────┬──────────────┐
  Rows           │   Shard 0    │   Shard 1    │
  (fsdp axis)    ├──────────────┼──────────────┤
                 │   Shard 2    │   Shard 3    │
                 └──────────────┴──────────────┘
```

`ici_fsdp_transpose_parallelism` sets the degree of the orthogonal transpose FSDP axis within a single TPU slice over the fast Inter-Chip Interconnect.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | 2D transpose sharding disabled; 1D-FSDP is used. |
| Integer $> 1$ (e.g. `2`, `4`, `8`) | Activates 2D-FSDP weight partitioning within the slice. |

Default in `base.yml`:
```yaml
ici_fsdp_transpose_parallelism: 1
```

---

## 3. Interactions with Related Parameters

- **`use_2d_fsdp_sharding`**: Master switch for 2D MoE/dense weight sharding.
- **`dense_fsdp_use_two_stage_all_gather`**: When `ici_fsdp_transpose_parallelism > 1`, setting this flag splits weight gathering into two sequential stages to optimize communication ring paths.
- **`replicate_quant_scale`**: Avoids inefficient XLA fusion when 2D sharding is active with quantization.

---

### One-line intuition

> **`ici_fsdp_transpose_parallelism` configures the second orthogonal dimension for 2D-FSDP parameter partitioning across the fast chip interconnect.**
