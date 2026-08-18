## 1. Why does `enable_tpu_profiling_options` exist?

Standard XPlane profiling records high-level TPU trace events (TensorCore execution, memory transfers).

However, low-level hardware performance tuning on TPU architectures (such as TPU v5e/v5p) requires specialized tracing of SparseCore tiles, memory controllers, and chip-level hardware subsystems:

```text
Standard TPU Profiling:
  Captures: TensorCore / MXU utilization & host dispatch

Advanced TPU Profiling (enable_tpu_profiling_options: true):
  Captures: SparseCore tiles, SparseCore units, chip-specific hardware counters
```

`enable_tpu_profiling_options` acts as the master switch to enable low-level TPU hardware trace options.

---

## 2. Fundamentals & Mechanics

- Enables downstream parameters: `tpu_num_chips_to_profile_per_task`, `tpu_num_sparse_core_tiles_to_trace`, `tpu_num_sparse_cores_to_trace`.
- Default `false` avoids inflating trace buffer sizes with low-level subsystem telemetry.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `false` | Standard high-level TPU profiling. |
| Advanced | `true` | Activates low-level SparseCore and chip tracing options. |

---

## 4. Interactions & Dependencies

```text
enable_tpu_profiling_options: true
             │
             ├──> tpu_num_chips_to_profile_per_task
             ├──> tpu_num_sparse_core_tiles_to_trace
             └──> tpu_num_sparse_cores_to_trace
```

---

## 5. Practical Scenarios & Failure Modes

- Required when debugging Dynamic Sparse Attention (DSA) or MoE SparseCore gather/scatter custom kernels.

---

### One-line intuition

> **`enable_tpu_profiling_options` unlocks low-level hardware tracing configurations for TPU SparseCore and chip subsystems.**
