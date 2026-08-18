## 1. Why does `upload_all_profiler_results` exist?

In large multi-host clusters (e.g. 128 or 512 hosts), every host records its own local XPlane trace file containing hundreds of megabytes of event buffers.

Uploading traces from every single host consumes massive network bandwidth and produces redundant data for symmetric SPMD programs. However, when debugging inter-host network skew, stragglers, or asymmetric communication bottlenecks, engineers must inspect traces from all worker nodes:

```text
Trace Upload Strategy:

upload_all_profiler_results: false (Default)
Host 0 ──>[ Uploads XPlane Trace to GCS ] ──> Fast, lightweight, representative
Host 1..N ──>[ Discards Local Trace ]

upload_all_profiler_results: true
Host 0..N ──>[ ALL Upload Traces to GCS ] ──> Essential for multi-host skew diagnosis
```

`upload_all_profiler_results` controls whether profiling traces are uploaded from all cluster hosts or only host 0.

---

## 2. Fundamentals & Mechanics

- **`false` (Default):** Only `jax.process_index() == 0` uploads trace files to GCS.
- **`true`:** Every worker process uploads its local trace with host-specific filenames.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `false` | Uploads traces only from host 0. |
| Multi-Host Debug | `true` | Uploads traces from all cluster hosts for distributed diagnostics. |

---

## 4. Interactions & Dependencies

- Active when `profiler: "xplane"`.

---

## 5. Practical Scenarios & Failure Modes

- **Diagnosing Straggler Nodes:** If one host in a 64-host pod runs 20% slower due to bad optical interconnects, set `upload_all_profiler_results: true` to locate the lagging rank.

---

### One-line intuition

> **`upload_all_profiler_results` determines whether hardware profile traces are uploaded from every host in the cluster or only from process 0.**
