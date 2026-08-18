## 1. Why does `tpu_num_chips_to_profile_per_task` exist?

On multi-chip TPU VMs (e.g. v4-32 or v5p-64 hosts with multiple chips per VM task), capturing deep hardware traces on all chips simultaneously creates massive trace buffer overhead.

Limiting the trace to a subset of chips per task provides representative hardware performance data while keeping trace file sizes manageable:

```text
Host Task (4 Physical TPU Chips):
  Chip 0 ──>[ Detailed Subsystem Tracing Enabled ]
  Chip 1..3 ──>[ Standard Execution ]
  (tpu_num_chips_to_profile_per_task = 1)
```

`tpu_num_chips_to_profile_per_task` defines how many physical TPU chips per process task are traced at low-level granularity.

---

## 2. Fundamentals & Mechanics

- Default `1` chip per task is sufficient for symmetric parallel workloads.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `1` | Traces 1 TPU chip per host task. |
| Full Host Trace | `4` or `8` | Traces all chips attached to the host VM. |

---

## 4. Interactions & Dependencies

- Active only when `enable_tpu_profiling_options: true`.

---

## 5. Practical Scenarios & Failure Modes

- Use default `1` unless debugging intra-host chip-to-chip PCIe or ICI bandwidth imbalances.

---

### One-line intuition

> **`tpu_num_chips_to_profile_per_task` sets the number of physical TPU chips per host task subject to low-level hardware tracing.**
