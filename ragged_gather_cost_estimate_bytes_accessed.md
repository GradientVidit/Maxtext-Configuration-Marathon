
## 1. Why bytes-accessed matters separately from FLOPs

XLA's scheduler models two performance bottlenecks independently:

```text
Compute-bound:   FLOPs / peak_FLOPS = minimum time
Memory-bound:    bytes_accessed / peak_bandwidth = minimum time

Actual time ≈ max(compute_bound, memory_bound)
```

The ragged gather is a gather operation — it reads from a large buffer based on sparse indices. It's **memory-bandwidth-bound**, not compute-bound. Its FLOPs estimate is small; its bytes-accessed estimate is what actually predicts execution time.

If XLA only sees the FLOP estimate and assumes the kernel is fast (small FLOPs), it may schedule high-priority work after the gather rather than before it, causing delays.

`ragged_gather_cost_estimate_bytes_accessed` provides XLA with the bytes-accessed estimate to correctly model this memory-bound characteristic.

---

## 2. What it controls

```yaml
ragged_gather_cost_estimate_bytes_accessed: -1  # auto-compute (default)
ragged_gather_cost_estimate_bytes_accessed: N   # override with N bytes estimate
```

Same semantics as `ragged_gather_cost_estimate_flops` but for memory bandwidth cost.

---

## 3. Default

```yaml
ragged_gather_cost_estimate_bytes_accessed: -1
```

Auto-computed. MaxText estimates bytes accessed based on the ragged buffer size and access patterns.

---

## 4. When to override

Very rarely — only when profiling shows XLA's scheduling of the gather is wrong due to incorrect memory cost estimates. The two-knob pair (`_flops` + `_bytes_accessed`) lets you tune how XLA classifies the kernel:

```text
Low FLOPs + High bytes  →  XLA treats it as memory-bound (correct for gather)
Low FLOPs + Low bytes   →  XLA underestimates, may schedule it late
```

---

## 5. The full cost estimate family

| Param | Kernel | Cost |
|---|---|---|
| `ragged_gather_cost_estimate_flops` | gather | FLOPs |
| `ragged_gather_cost_estimate_bytes_accessed` | gather | bytes |
| `ragged_gather_reduce_cost_estimate_flops` | gather-reduce | FLOPs |
| `ragged_gather_reduce_cost_estimate_bytes_accessed` | gather-reduce | bytes |

---

### One-line intuition

> **`ragged_gather_cost_estimate_bytes_accessed` tells XLA how much memory bandwidth the ragged gather kernel uses, letting the scheduler correctly treat it as memory-bound rather than compute-bound — auto-compute is almost always correct.**
