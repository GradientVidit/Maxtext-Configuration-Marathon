## 1. Why does it exist?

Pipeline Parallelism (PP) partitions the transformer's sequential decoder layers across different stages. Communication in pipeline parallelism consists solely of passing activation tensors from stage $K$ to stage $K+1$ (and gradients backward from $K+1$ to $K$).

Because point-to-point boundary transfers involve far less total bandwidth than all-to-all or all-reduce collectives, Pipeline Parallelism is naturally well-suited to run across slower Data Center Network (DCN) connections between separate TPU slices.

```text
  Slice 0 (Stage 0: Layers 0..15)
              │
    [ DCN Point-to-Point Activation Transfer ]
              ↓
  Slice 1 (Stage 1: Layers 16..31)
```

`dcn_pipeline_parallelism` sets the degree of pipeline parallelism stages spanning across multi-slice clusters over the Data Center Network.

---

## 2. Fundamentals & Sizing

Total pipeline stages in MaxText is the product of ICI and DCN pipeline degrees:

$$\text{num\_stages} = \text{ici\_pipeline\_parallelism} \times \text{dcn\_pipeline\_parallelism}$$

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | Pipeline parallelism is not split across slices over DCN. |
| Integer $> 1$ | Allocates `N` separate TPU slices as sequential pipeline stages. |

Default in `base.yml`:
```yaml
dcn_pipeline_parallelism: 1
```

---

## 4. Interactions with Related Parameters

- **`num_pipeline_microbatches`**: Must be tuned to minimize the pipeline bubble fraction across DCN stages.
- **`pipeline_delay_activation_forwarding`**: Decouples cross-slice DCN transfer from stage computation.
- **`num_layers_per_pipeline_stage`**: Sets the number of layers hosted per pipeline slice.

---

### One-line intuition

> **`dcn_pipeline_parallelism` assigns pipeline stages across separate TPU slices over the datacenter network, utilizing point-to-point activation passing to scale deep models.**
