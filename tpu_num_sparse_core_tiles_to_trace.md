## 1. Why does `tpu_num_sparse_core_tiles_to_trace` exist?

TPU SparseCores contain internal execution tiles dedicated to vector/ragged memory operations (used in embedding lookups, MoE token routing, and SparseCore kernels).

Tracing every tile on every core generates immense trace event volumes. Restricting tile tracing depth bounds trace file overhead:

```text
SparseCore Architecture:
  SparseCore ──> Tile 0 [Traced] | Tile 1..N [Not Traced]
  (tpu_num_sparse_core_tiles_to_trace = 1)
```

`tpu_num_sparse_core_tiles_to_trace` specifies how many SparseCore execution tiles per core are traced.

---

## 2. Fundamentals & Mechanics

- Default `1` provides representative SparseCore tile execution timelines.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `1` | Traces 1 tile per SparseCore. |
| Multi-Tile | `2`, `4` | Deep trace across multiple SparseCore tiles. |

---

## 4. Interactions & Dependencies

- Active when `enable_tpu_profiling_options: true`.

---

## 5. Practical Scenarios & Failure Modes

- Essential when profiling custom Mosaic or Pallas kernels executing on SparseCore tiles.

---

### One-line intuition

> **`tpu_num_sparse_core_tiles_to_trace` sets the number of execution tiles per SparseCore included in hardware trace dumps.**
