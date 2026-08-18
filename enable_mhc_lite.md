## 1. Why does `enable_mhc_lite` exist?

Iterative Sinkhorn-Knopp projection requires running multiple serial row/column normalizations ($N=20$ steps) per layer. During backward passes, auto-differentiating through 20 iterative steps increases graph depth, memory consumption, and kernel launch latency.

According to the **Birkhoff-von Neumann theorem**, the set of all $k \times k$ doubly stochastic matrices (the Birkhoff polytope) is precisely the convex hull of the set of $k \times k$ permutation matrices.

Instead of running an iterative loop:

```text
Standard mHC (Sinkhorn-Knopp):
Arbitrary Logits ──> Exp ──> [Row Norm <──> Col Norm] × 20 iterations ──> Doubly Stochastic H

mHC-Lite (Birkhoff-von Neumann Convex Combination):
Learnable Weights w ∈ ℝ^{k!} ──> Softmax ──> α ∈ ℝ^{k!} (Convex Coefficients)
                                                    │
                                                    ▼
Doubly Stochastic Matrix:  H = ∑_{i=1}^{k!} α_i P_i  (Closed-form, exact in 1 step!)
```

`enable_mhc_lite` replaces the iterative Sinkhorn-Knopp algorithm with a closed-form, single-step convex combination of permutation matrices.

---

## 2. Mechanics & Factorial Complexity Trade-off

1. **Permutation Basis**: For expansion rate $k$, there are $k!$ distinct permutation matrices $P_1, P_2, \dots, P_{k!}$.
   - For $k = 2$: $2! = 2$ permutations.
   - For $k = 3$: $3! = 6$ permutations.
   - For $k = 4$: $4! = 24$ permutations.
   - For $k = 5$: $5! = 120$ permutations (cost begins to scale rapidly).
2. **Execution**: The model learns a vector of logits $w \in \mathbb{R}^{k!}$, computes $\alpha = \text{softmax}(w)$, and calculates $H = \sum \alpha_i P_i$.
3. **Guarantees**: $H$ is guaranteed to be mathematically doubly stochastic in a single forward pass without iterations, rounding errors, or backpropagation unrolling overhead.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `enable_mhc_lite` | `bool` | `false` | `true` (use permutation-based mHC-lite), `false` (use Sinkhorn-Knopp iterations) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `mhc_expansion_rate` | Must be small (typically $k=2$ or $k=4$) when using `enable_mhc_lite: true`. For $k > 4$, $k!$ scaling makes standard Sinkhorn preferable. |
| `sinkhorn_iterations` | Bypassed when `enable_mhc_lite: true`. |

---

## 5. Practical Guidance & When to Enable

| Configuration | Recommended Mode | Rationale |
| :--- | :--- | :--- |
| `mhc_expansion_rate: 4` | `enable_mhc_lite: true` | $4! = 24$ basis matrices is extremely fast, exact, and eliminates the 20-step Sinkhorn loop. |
| `mhc_expansion_rate: 2` | `enable_mhc_lite: true` | $2! = 2$ permutations; nearly zero overhead. |
| `mhc_expansion_rate: 8` | `enable_mhc_lite: false` | $8! = 40,320$ permutations is computationally impractical; use Sinkhorn-Knopp instead. |

---

### One-line intuition

> `enable_mhc_lite` computes the mHC doubly stochastic mixing matrix via a closed-form convex combination of permutation matrices, eliminating Sinkhorn iterations for small stream counts ($k \le 4$).
