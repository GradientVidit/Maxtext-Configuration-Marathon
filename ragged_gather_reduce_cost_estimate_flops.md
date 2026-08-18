
## 1. The ragged gather-reduce kernel

The ragged gather-reduce is the fused combine step in MoE:
1. Gather expert outputs from per-expert buffers back to per-token positions
2. Apply routing weights and reduce k expert outputs per token

This fused operation has its own cost model separate from the plain gather kernel. XLA needs a FLOP estimate to schedule it correctly relative to other operations in the forward/backward pass.

`ragged_gather_reduce_cost_estimate_flops` provides that estimate.

---

## 2. What it controls

```yaml
ragged_gather_reduce_cost_estimate_flops: -1  # auto-compute (default)
ragged_gather_reduce_cost_estimate_flops: N   # override with N FLOPs estimate
```

Same mechanism as `ragged_gather_cost_estimate_flops` but for the gather-reduce kernel specifically.

---

## 3. Why the gather-reduce may have different FLOPs from plain gather

The reduce step adds computation: for each token, multiply k expert outputs by routing weights and sum. This is:

```text
k × emb_dim multiply-adds per token
= k × emb_dim × 2 FLOPs per token
```

At `num_experts_per_tok=8` and `emb_dim=7168`, that's 114,688 additional FLOPs per token for the reduction. Compared to the gather's memory access cost, this may or may not be significant depending on batch size.

The fused kernel's total FLOP estimate must account for both the gather and the reduce.

---

## 4. Default

```yaml
ragged_gather_reduce_cost_estimate_flops: -1
```

Auto-compute. MaxText estimates this based on model parameters.

---

## 5. The full family

| Param | Kernel | Cost |
|---|---|---|
| `ragged_gather_cost_estimate_flops` | gather | FLOPs |
| `ragged_gather_cost_estimate_bytes_accessed` | gather | bytes |
| `ragged_gather_reduce_cost_estimate_flops` | gather-reduce | FLOPs |
| `ragged_gather_reduce_cost_estimate_bytes_accessed` | gather-reduce | bytes |

---

### One-line intuition

> **`ragged_gather_reduce_cost_estimate_flops` provides XLA with a FLOP cost for the fused gather-reduce (MoE combine step) kernel — identical concept to `ragged_gather_cost_estimate_flops` but for the reduce variant; auto-compute is almost always correct.**
