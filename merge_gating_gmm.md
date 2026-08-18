
## 1. What merge_gating_gmm does

In the standard MoE forward pass, the gating (routing) and the expert matmul are separate operations:

```text
Step 1: Router → routing logits → top-k → routing weights
Step 2: GMM    → expert matmul → expert outputs
Step 3: Combine → weight expert outputs by routing weights → sum
```

When `merge_gating_gmm=True`, MaxText fuses Step 1 (the gating computation) into the GMM kernel call itself — the routing weight application is done inside the grouped matmul rather than as a separate multiply-then-sum:

```text
merged: [Router logits + GMM + routing weighting] in one kernel call
```

---

## 2. The potential benefit

Fusing gating into the GMM kernel:
- Reduces kernel launch overhead (one kernel instead of two)
- Allows the weighting to happen at the right level of the memory hierarchy (inside the kernel, on registers/SRAM, before results are written back to HBM)
- Potentially reduces memory reads/writes by avoiding materializing intermediate per-expert outputs

---

## 3. Default

```yaml
merge_gating_gmm: false
```

Not merged by default. The fused path is an optimization that needs validation.

---

## 4. When to enable

This is a performance optimization for the GMM-heavy forward pass. Enable when:
- Profiling shows the gating computation or the weighting step is a measurable bottleneck
- Benchmarking a configuration where the extra kernel launch overhead matters
- Using a backend that supports the merged operation efficiently

Not a correctness change — merged and separate paths produce the same result.

---

## 5. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `use_tokamax_gmm` | Tokamax backend may have better support for merged gating |
| `megablox` | Megablox (default) GMM path; merge support may vary by backend |
| `sparse_matmul` | Merge only relevant in sparse path where GMM is used |

---

### One-line intuition

> **`merge_gating_gmm` fuses the routing weight application into the GMM kernel call, potentially reducing kernel overhead and intermediate memory writes — a performance optimization; benchmark before enabling.**
