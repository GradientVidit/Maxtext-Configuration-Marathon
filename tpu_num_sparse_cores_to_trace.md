## 1. Why does `tpu_num_sparse_cores_to_trace` exist?

Modern TPU chips (e.g. TPU v5p) contain multiple independent SparseCores alongside their TensorCore MXUs.

Developers tuning ragged MoE routing or custom gather kernels need to specify how many SparseCores per chip record event timelines:

```text
TPU Chip Subsystems:
  MXU Matrix Unit (TensorCore)
  SparseCore 0 ──>[ Traced ]
  SparseCore 1 ──>[ Traced ]  <-- tpu_num_sparse_cores_to_trace = 2
  SparseCore 2..N [ Untraced ]
```

`tpu_num_sparse_cores_to_trace` defines the number of SparseCores per chip included in the trace.

---

## 2. Fundamentals & Mechanics

- Default `2` captures standard SparseCore dual-core execution.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `2` | Traces 2 SparseCores per TPU chip. |
| Single Core | `1` | Minimal SparseCore trace footprint. |

---

## 4. Interactions & Dependencies

- Active when `enable_tpu_profiling_options: true`.

---

## 5. Practical Scenarios & Failure Modes

- Used in MoE ragged gather/scatter kernel optimization to verify balanced load across SparseCores.

---

### One-line intuition

> **`tpu_num_sparse_cores_to_trace` sets the number of physical SparseCores per TPU chip monitored during profiling.**
