## 1. Why does it exist?

When training foundation models across multi-slice TPU supercomputers (e.g. 4x `v5p-1024` slices connected over standard Data Center Network ethernet links), communication latency between separate pod slices is significantly higher than the high-speed Inter-Chip Interconnect (ICI) within a single slice.

Pure **Data Parallelism** (DP) requires communication only during gradient synchronization at the end of the backward pass (All-Reduce / Reduce-Scatter), which is easily overlapped with backward compute and requires no fine-grained activation exchange.

```text
       Slice 0 (v5p-1024)                 Slice 1 (v5p-1024)
   ┌───────────────────────┐          ┌───────────────────────┐
   │ Full Model Replica    │          │ Full Model Replica    │
   │ (Batches 0..N)        │          │ (Batches N+1..2N)     │
   └───────────┬───────────┘          └───────────┬───────────┘
               │                                  │
               └───────────┬──────────────────────┘
                           │ Data Center Network (DCN)
                           ↓
               Gradient Sync (dcn_data_parallelism)
```

`dcn_data_parallelism` sets the degree of data parallelism operating across multi-slice Data Center Networks, and is the **recommended auto-sharding axis for DCN**.

---

## 2. Fundamentals & Auto-Derivation (`-1`)

In MaxText, exactly one axis in the DCN family may be set to `-1` to auto-calculate its value from the total number of available slices (`num_slices`):

$$\prod \text{all DCN axes} = \text{num\_slices}$$

When `dcn_data_parallelism: -1` and all other DCN axes are `1`, `dcn_data_parallelism` is automatically set to `num_slices`.

```text
Cluster: 4 TPU Pod Slices (num_slices = 4)
Config:
  dcn_data_parallelism: -1  ──→ Auto-resolves to 4
  dcn_fsdp_parallelism: 1
  dcn_tensor_parallelism: 1
```

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `-1` (default) | Auto-derived from `num_slices` (Recommended setting for multi-slice runs). |
| `1` | Disables DCN data parallelism (e.g. single slice run, or all DCN allocated to pipeline/FSDP). |
| Positive integer $> 1$ | Explicit number of data-parallel slices. |

Default in `base.yml`:
```yaml
dcn_data_parallelism: -1  # recommended DCN axis to be auto-sharded
```

---

## 4. Interactions with Related Parameters

- **`num_slices`**: In multi-slice training, `num_slices` defines the total slice multiplier.
- **`ici_fsdp_parallelism`**: The standard production paradigm combines **ICI FSDP** (within each slice) with **DCN Data Parallelism** (across slices).
- **`shard_optimizer_over_data`**: When `true`, optimizer states are partitioned across data-parallel replicas (ZeRO-1).

---

## 5. Practical Recommendations

- **Standard Multi-Slice Recipe**: Keep `dcn_data_parallelism: -1`. It delivers maximum scaling efficiency because DCN bandwidth limits do not throttle inner forward/backward attention or MLP layers.

---

### One-line intuition

> **`dcn_data_parallelism` scales training across multi-slice clusters by replicating model instances across separate TPU slices over the datacenter network, serving as MaxText's recommended auto-shard DCN axis.**
