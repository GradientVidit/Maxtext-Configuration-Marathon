
## 1. Why do they exist?

Dense neural networks use all their parameters for every input. This creates an uncomfortable scaling law: doubling parameters doubles compute. At 100B+ parameters, compute cost becomes prohibitive.

**Structured sparsity** breaks this coupling. N:M sparsity means: in every consecutive block of M values, at most N are non-zero (the rest are zeroed out). The zeroed weights don't contribute to the output, so their computation can be skipped.

```text
2:4 sparsity example (N=2, M=4):
  dense weights:  [0.3, -0.7, 0.1, 0.9]   (4 non-zeros, 4 MACs)
  sparse mask:    [1,    1,   0,   1  ]    ← pick top-2 by magnitude
  sparse weights: [0.3, -0.7, 0.0, 0.9]   (3 non-zeros shown, but structured)

wait, 2:4 means exactly 2 non-zeros per block of 4:
  sparse mask:    [0,    1,   0,   1  ]
  sparse weights: [0.0, -0.7, 0.0, 0.9]   (2 non-zeros, 2 MACs)
```

NVIDIA's Ampere+ sparse tensor cores can execute 2:4 sparse matmuls at up to **2x the theoretical MXU throughput** of equivalent dense matmuls, because the hardware indexes and skips the zero-valued multiplications. In practice, kernel-level speedups are typically 1.3–1.6x, and end-to-end model speedups are lower still (often 10–30%) due to non-GEMM operations and memory-bandwidth bottlenecks.

`weight_sparsity_n` and `weight_sparsity_m` define the N and M of the N:M sparsity pattern.

---

## 2. How sparsity works in training (gradual magnitude pruning)

MaxText doesn't jump to full sparsity at step 0. The process is:

```text
steps 0 → weight_sparsity_start_step:
  train densely (no sparsity)

step weight_sparsity_start_step:
  compute initial sparsity mask (keep top-N by magnitude in each M-block)
  apply mask to weights

every weight_sparsity_update_step steps after that:
  recompute mask (may change which N weights survive)
  → gradually prunes weights that become less important
```

This is **gradual magnitude pruning** — the model learns to route information around zeroed-out weights.

---

## 3. Options

| Param | Default | Behavior when null |
|---|---|---|
| `weight_sparsity_n` | `null` | Sparsity disabled |
| `weight_sparsity_m` | `null` | Sparsity disabled |

Default in base.yml:
```yaml
weight_sparsity_n: null
weight_sparsity_m: null
```

To enable:
```yaml
weight_sparsity_n: 2
weight_sparsity_m: 4
```

Both must be set simultaneously. Setting one without the other is invalid.

---

## 4. Common patterns

| N:M | Sparsity | Hardware support |
|---|---|---|
| 2:4 | 50% | NVIDIA Ampere+ sparse tensor cores |
| 1:4 | 75% | Theoretical, limited hardware support |
| 1:2 | 50% | Simpler, less common |

2:4 is by far the most practically useful because NVIDIA explicitly supports it with hardware acceleration. TPUs currently don't have equivalent sparse computation hardware — sparsity on TPU is less useful.

---

## 5. What breaks if wrong

Setting N > M is invalid (can't have more non-zeros than elements in the block).

Setting sparsity on hardware that doesn't support N:M sparse ops (e.g., TPU) means the zeros are applied but no compute savings are realized — you lose accuracy with no throughput benefit.

---

### One-line intuition

> **`weight_sparsity_n` and `weight_sparsity_m` define N:M structured weight sparsity — where at most N values per block of M survive — enabling hardware-accelerated sparse matmuls (e.g., 2:4 on NVIDIA sparse tensor cores) that achieve the same output as a dense model at half the GEMM compute.**
