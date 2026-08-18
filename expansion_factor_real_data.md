## 1. Why does `expansion_factor_real_data` exist?

In massive multi-host distributed training (e.g., thousands of TPU v4/v5e hosts or multi-node GPU clusters), having *every single host* independently stream, parse, and decompress data files from Google Cloud Storage (GCS) creates massive network metadata contention and storage throttling.

```text
Standard (All Hosts Read from GCS):
Host 0 ──> GCS
Host 1 ──> GCS  ===> GCS Read Contention / Metadata Rate Throttling
Host N ──> GCS

With expansion_factor_real_data > 1:
Host 0 (reads 4x batch) ───[ICI / High-speed Interconnect]───> Broadcasts to Hosts 1,2,3
Host 4 (reads 4x batch) ───[ICI / High-speed Interconnect]───> Broadcasts to Hosts 5,6,7
(Active GCS storage connections reduced by 4x!)
```

`expansion_factor_real_data` solves this by decoupling the number of data-loading hosts from the total compute hosts. A subset of hosts loads an expanded batch size from storage and shares it across the high-speed inter-chip interconnect (ICI).

---

## 2. Mechanics: Scale-out Read Reducer vs. Downscale Resumer

`expansion_factor_real_data` provides two distinct capabilities depending on its value:

### Mode 1: Multi-Host Read Consolidation (`expansion_factor_real_data > 1.0`)
When set to a factor $E > 1.0$:
- **Loading Hosts**: Only $\lfloor \text{total\_hosts} / E \rfloor$ hosts perform actual I/O operations against GCS.
- **Loaded Batch Size**: Each loading host loads an expanded slice of data equal to $\text{per\_device\_batch\_size} \times E$.
- **Non-loading Hosts**: Non-loading hosts instantiate a `PlaceHolderDataIterator` that outputs dummy placeholder data.
- **Interconnect Broadcast**: The real batch data loaded by the active hosts is scattered and broadcast across the high-speed ICI mesh to replace the dummy placeholders before gradient execution.

> [!TIP]
> You can enable `max_checkify=true` in MaxText to perform runtime assertions verifying that placeholder data is never accidentally processed during gradient updates.

```text
Host 0 (Loading Host)    ──[Reads Real Batch * 4]──┐
Host 1 (PlaceHolderHost) ──[Dummy PlaceHolder]   ──┼──> ICI Collective Scatter ──> Real Tensors on all devices
Host 2 (PlaceHolderHost) ──[Dummy PlaceHolder]   ──┤
Host 3 (PlaceHolderHost) ──[Dummy PlaceHolder]   ──┘
```

### Mode 2: Grain Iterator Checkpoint Downscaling (`0.0 < expansion_factor_real_data < 1.0`)
Grain saves deterministic dataset iterator state as JSON metadata within Orbax checkpoints. These checkpoint states record shard positions tied to the original cluster chip count.
- If you save a Grain checkpoint on a 512-chip cluster and later resume on a 256-chip cluster, standard restoration fails due to shard coordinate mismatches.
- Setting `expansion_factor_real_data: 0.5` adapts the Grain iterator's sharded checkpoint state to cleanly resume on the smaller cluster without losing position or duplicating samples.

---

## 3. Options & Default

| Parameter | Type | Default | Valid Range |
| :--- | :--- | :--- | :--- |
| `expansion_factor_real_data` | `float` | `-1.0` | `-1.0` (disabled), `> 1.0` (active host reduction), or `(0.0, 1.0)` (Grain resume adapter) |

---

## 4. Interactions with Related Parameters

- **`per_device_batch_size`**: Base batch size per accelerator device.
- **`dataset_type: grain`**: Core integration with Grain's sharded multi-host iterators.
- **`max_checkify`**: Validates that placeholder data from `PlaceHolderDataIterator` is completely replaced by real data.
- **`colocated_python_data_input`**: Cooperates with host execution topology under Pathways.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Scaling from 64 to 256 TPU hosts** | Training step times degrade due to GCS storage connection limits | Set `expansion_factor_real_data: 4.0` (reduces reading hosts to 64). |
| **Resuming Grain run on half the chip count (e.g. 512 -> 256 chips)** | Grain iterator state error: expected 512 shards | Set `expansion_factor_real_data: 0.5`. |
| **Using with gradient accumulation** | Older MaxText versions had issues with placeholder truncation in `loss_fn` | Verify MaxText version is up-to-date and run with `max_checkify=true`. |

---

### One-line intuition

> `expansion_factor_real_data` reduces cloud storage I/O bottlenecks by having fewer hosts read expanded batches and distribute them across ICI, or adapts Grain dataset checkpoints across differing cluster chip counts.
