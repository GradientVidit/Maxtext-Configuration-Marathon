
## 1. Why XLA needs cost estimates for custom kernels

XLA is MaxText's compiler. When XLA schedules operations, it uses **cost models** — estimates of how long each operation takes — to decide:
- What to run in parallel vs. sequentially
- When to prefetch data
- How to interleave communication and compute

For standard JAX operations, XLA has built-in cost models. For custom Pallas/SparseCore kernels like the ragged gather, XLA doesn't know the cost — it gets a black box. Without a cost estimate, XLA may make poor scheduling decisions: it might run the kernel at a time that blocks other work, or fail to overlap it with adjacent communication.

`ragged_gather_cost_estimate_flops` provides XLA with an estimated FLOP count for the ragged gather kernel, allowing it to make better scheduling decisions.

---

## 2. What it controls

```yaml
ragged_gather_cost_estimate_flops: -1  # auto-compute (default)
ragged_gather_cost_estimate_flops: N   # override with N FLOPs estimate
```

When `-1`: MaxText auto-computes an estimate based on model/batch parameters.  
When `> 0`: XLA uses this override as the FLOP cost for the ragged gather kernel.

---

## 3. Why you'd ever override it

Auto-computed estimates are derived from parameter counts and access patterns. But real throughput depends on hardware, memory bandwidth, and kernel efficiency — factors the formula doesn't capture.

If profiling shows XLA is scheduling the ragged gather kernel badly (e.g. blocking a high-priority all-gather, or not overlapping it with available compute), you can nudge XLA by adjusting the cost estimate:

```text
Increase estimate → XLA treats kernel as more expensive → gives it higher scheduling priority
Decrease estimate → XLA treats it as cheaper → may interleave it differently
```

---

## 4. Default

```yaml
ragged_gather_cost_estimate_flops: -1
```

Auto-compute. This is correct for the vast majority of configurations.

---

## 5. Related parameters (the four cost estimate knobs)

These four parameters are a family — same concept applied to two kernels × two cost dimensions:

| Param | Kernel | Cost dimension |
|---|---|---|
| `ragged_gather_cost_estimate_flops` | ragged gather | FLOPs |
| `ragged_gather_cost_estimate_bytes_accessed` | ragged gather | bytes accessed |
| `ragged_gather_reduce_cost_estimate_flops` | ragged gather-reduce | FLOPs |
| `ragged_gather_reduce_cost_estimate_bytes_accessed` | ragged gather-reduce | bytes accessed |

Both FLOPs and bytes-accessed matter for scheduling: a memory-bandwidth-bound kernel might have low FLOPs but high bytes-accessed, and XLA needs both to schedule correctly.

---

## 6. When to touch these

Almost never. These are last-resort tuning knobs when:
- Profiling shows XLA is clearly misscheduling the ragged gather relative to other operations
- Auto-computed estimates are measurably wrong for your hardware

---

### One-line intuition

> **`ragged_gather_cost_estimate_flops` provides XLA's scheduler with a FLOP count for the ragged gather custom kernel, enabling better scheduling decisions — leave at `-1` (auto) unless profiling reveals a scheduling pathology.**
