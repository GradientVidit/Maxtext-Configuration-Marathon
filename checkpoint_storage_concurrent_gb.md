
## 1. What "concurrent GB" means

When Orbax saves or loads a checkpoint, it doesn't transfer all the data sequentially. It issues parallel I/O operations to maximize throughput. But unbounded parallelism causes memory pressure — all the in-flight data has to live somewhere in host RAM while being transferred.

`checkpoint_storage_concurrent_gb` caps the **total amount of data that can be in-flight simultaneously** during checkpoint I/O:

```text
concurrent_gb = 96
→ at any moment, at most 96GB of checkpoint data is being transferred/processed
```

From the MaxText config comment:
> "larger models requires higher concurrent GB for I/O"
> "default concurrent gb for PytreeCheckpointHandler is 96GB"

---

## 2. The throughput vs. memory trade-off

```text
higher concurrent_gb → more parallel I/O → faster checkpoint save/load
                      → more host RAM used for buffering in-flight data

lower concurrent_gb  → less RAM pressure
                      → checkpoint I/O may be serialized/slower
```

This is the same principle as TCP window size or buffer sizes in any I/O pipeline — the "window" of in-flight data determines max throughput.

---

## 3. When the default (96GB) is insufficient

For large models:

```text
70B model in bfloat16:
  params alone ≈ 140GB
  optimizer state ≈ 280GB (Adam)
  total ≈ 420GB per full checkpoint
```

If `concurrent_gb: 96`, you're processing the 420GB checkpoint in sequential 96GB windows. The parallelism is limited.

Increasing to, say, 256GB lets more arrays be loaded/saved simultaneously, reducing wall-clock checkpoint time.

**The constraint is host RAM.** Each host has a finite amount of RAM (e.g., 192GB or 512GB for large TPU hosts). Setting `concurrent_gb` higher than available host RAM causes OOM during checkpointing.

---

## 4. Practical guidance

| Model size | Suggested `concurrent_gb` |
|---|---|
| <7B | 96 (default) |
| 7B–70B | 96–256 |
| 70B+ | 256–512 (depending on host RAM) |

Don't set this blindly higher than `(host_RAM - model_working_set)`. During training, the model is already resident in memory — the concurrent_gb budget competes with it.

---

## 5. Per-host vs global

Note this is the concurrent GB limit **per PytreeCheckpointHandler**, not across the whole cluster. Each host manages its own I/O window. The total cluster-wide I/O throughput is `concurrent_gb × num_hosts`.

---

## 6. Options

Any positive integer (in GB):

```yaml
checkpoint_storage_concurrent_gb: 96    # default
checkpoint_storage_concurrent_gb: 192   # for larger models
checkpoint_storage_concurrent_gb: 256   # for 70B+ models
```

---

### One-line intuition

> **`checkpoint_storage_concurrent_gb` caps the amount of checkpoint data in-flight simultaneously — setting it higher speeds up I/O for large models but consumes more host RAM, so it must stay within available memory.**
