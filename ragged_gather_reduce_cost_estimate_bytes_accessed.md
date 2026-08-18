
## 1. What this does

`ragged_gather_reduce_cost_estimate_bytes_accessed` provides XLA's scheduler with a memory-bandwidth cost estimate for the fused ragged gather-reduce kernel — the MoE combine step that gathers expert outputs and reduces them into token representations in a single pass.

Same concept as `ragged_gather_cost_estimate_bytes_accessed`, but applied to the gather-reduce kernel.

---

## 2. Why the gather-reduce may have different memory access from plain gather

The plain gather reads: routing indices + source buffer.  
The gather-reduce reads: routing indices + source buffer + routing weights; writes: combined output.

For large k (`num_experts_per_tok`) and large embedding dimensions, the total bytes accessed can be substantially larger than the plain gather — making the bytes-accessed estimate the critical scheduling signal for XLA.

---

## 3. What it controls

```yaml
ragged_gather_reduce_cost_estimate_bytes_accessed: -1  # auto-compute (default)
ragged_gather_reduce_cost_estimate_bytes_accessed: N   # override with N bytes
```

---

## 4. Default

```yaml
ragged_gather_reduce_cost_estimate_bytes_accessed: -1
```

Auto-compute. Override only when profiling reveals a scheduling problem caused by incorrect memory cost estimation.

---

## 5. The full cost estimate family

These four parameters are logically a 2×2 table:

```text
                    | FLOPs                                    | Bytes Accessed
--------------------|------------------------------------------|---------------------------------------------
ragged gather       | ragged_gather_cost_estimate_flops        | ragged_gather_cost_estimate_bytes_accessed
ragged gather-reduce| ragged_gather_reduce_cost_estimate_flops | ragged_gather_reduce_cost_estimate_bytes_accessed
```

All four default to `-1` (auto-compute) and are override-only.

---

### One-line intuition

> **`ragged_gather_reduce_cost_estimate_bytes_accessed` gives XLA the memory bandwidth cost for the fused MoE combine (gather-reduce) kernel — the bytes-accessed sibling of the FLOP estimate; auto-compute is almost always correct.**
