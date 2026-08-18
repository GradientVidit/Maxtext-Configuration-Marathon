
## 1. What this controls

`batch_split_factor` is the companion parameter to `use_batch_split_schedule`. It specifies how many micro-batches the global batch is divided into when the batch-split schedule is active.

```text
global batch (batch_size × seq_len tokens)
    ↓ split by batch_split_factor
micro-batch 0: tokens [0 : N/factor]
micro-batch 1: tokens [N/factor : 2N/factor]
...
micro-batch (factor-1): tokens [(factor-1)N/factor : N]
```

---

## 2. Why the value matters

More micro-batches → more pipeline stages → more overlap opportunity:

```text
factor=1: no splitting (degenerate case; use_batch_split_schedule has no effect)
factor=2: 2-stage pipeline (1 compute ↔ 1 dispatch pair)
factor=4: 4-stage pipeline (more overlap, but also more scheduling overhead)
```

But more splits also means smaller compute chunks per stage. If micro-batches become too small, the expert GEMM becomes memory-bandwidth-bound rather than compute-bound, and the throughput benefit shrinks.

---

## 3. Default

```yaml
batch_split_factor: 1
```

`1` = no splitting. This is the correct default when `use_batch_split_schedule=False`, which is the normal case.

---

## 4. Preconditions

- `use_batch_split_schedule: true` — without this, `batch_split_factor` has no effect
- DeepSeek sparse layers — current implementation only

---

## 5. Options

| Value | Effect |
|---|---|
| `1` (default) | No splitting — `use_batch_split_schedule` is effectively disabled |
| `2` | Split into 2 micro-batches |
| `4` | Split into 4 micro-batches |
| Higher | Diminishing returns; test empirically |

---

## 6. What to do in practice

1. Start with `batch_split_factor=2`, benchmark
2. Try `4`, compare
3. Pick the value where throughput peaks (varies with model size, EP degree, and network latency)

---

### One-line intuition

> **`batch_split_factor` controls how many micro-batches the batch is split into for the `use_batch_split_schedule` overlap strategy — higher values create more pipeline stages for hiding all-to-all latency, but micro-batches that are too small lose compute efficiency.**
