## 1. Why does it exist?

Before model forward computation begins, the input batch of token IDs and attention masks must be loaded by the data pipeline, sliced, and placed onto accelerator devices. 

In a complex multi-dimensional mesh (combining Data, FSDP, Pipeline, Context, Tensor, and Expert parallelism), which physical mesh axes should participate in partitioning the *input dataset batch*?

If the dataset is only sharded over the pure `data` axis, then devices allocated to `fsdp`, `context`, or `pipeline` axes would receive duplicate batches and compute redundant forward passes.

```text
Input Pipeline ──→ Global Batch: [Batch Size, Sequence Length]
                         │
                  [ data_sharding ]
                         │
   Participating Physical Axes: ['data', 'fsdp', 'context', 'stage', ...]
                         │
                         ↓
  Tensors sharded across all data-parallel & context-parallel devices
```

`data_sharding` declares the exact ordered list of physical mesh axes across which input data batches are distributed.

---

## 2. Fundamentals & DCN-Before-ICI Ordering Rule

The default in `base.yml` is a list of physical axes:
```yaml
data_sharding: [['data', 'stage', 'fsdp', 'fsdp_transpose', 'context', 'context_usp_ulysses', 'context_autoregressive', 'tensor', 'tensor_sequence', 'expert', 'autoregressive']]
```

### Critical Ordering Rule:
> **Axes used for DCN (inter-slice networking) must appear earlier in the `data_sharding` list than axes used for ICI (intra-slice interconnect).**

```text
Correct Ordering:
  [ DCN Axes (data, dcn_fsdp) ──→ ICI Axes (ici_fsdp, ici_context) ]
  (Ensures each TPU slice reads its contiguous segment of the dataset)

Incorrect Ordering:
  [ ICI Axes ──→ DCN Axes ] ──→ Fragmented DCN data loader communication
```

---

## 3. Options & Configuration

| Parameter | Type | Default |
|---|---|---|
| `data_sharding` | `list of lists` | `[['data', 'stage', 'fsdp', 'fsdp_transpose', 'context', 'context_usp_ulysses', 'context_autoregressive', 'tensor', 'tensor_sequence', 'expert', 'autoregressive']]` |

---

## 4. Interactions with Related Parameters

- **`input_data_sharding_logical_axes`**: Specifies the *logical* dimensions of the input tensor (e.g. `activation_embed_and_logits_batch`, `activation_norm_length`) that pair with `data_sharding`.
- **`ici_data_parallelism` / `dcn_data_parallelism`**: Define the sizes of the axes referenced in `data_sharding`.

---

## 5. Practical Scenarios

- **Adding Custom Parallelism**: If you introduce a custom pipeline or sequence axis, ensure it is registered in `data_sharding` so the input pipeline splits the batch or sequence across the new axis.
- **Default Stability**: In standard setups, the default list covers all active axes automatically.

---

### One-line intuition

> **`data_sharding` defines the prioritized physical mesh axes across which the input dataset pipeline partitions training batches, requiring DCN axes to precede ICI axes for optimal I/O.**
