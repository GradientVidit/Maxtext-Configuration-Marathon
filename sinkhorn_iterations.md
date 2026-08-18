## 1. Why does `sinkhorn_iterations` exist?

In Manifold-Constrained Hyper-Connections (mHC), multiple residual streams are mixed at each layer by a learned matrix $H \in \mathbb{R}^{k \times k}$.

If $H$ were unconstrained, repeated matrix multiplications across dozens of layers would cause exponential signal growth (if the spectral radius $> 1$) or exponential representation collapse (if the spectral radius $< 1$).

To preserve total representation energy across layers, $H$ must lie on the **Birkhoff Polytope** of doubly stochastic matrices:

$$\sum_{j=1}^k H_{ij} = 1 \quad \forall i, \qquad \sum_{i=1}^k H_{ij} = 1 \quad \forall j, \qquad H_{ij} \ge 0$$

```text
Unconstrained Pre-matrix M (Learnable Logits)
          │
          ▼  Take Exponential (Positive elements)
       A = exp(M)
          │
    ┌─────┴──────────────────────────────┐
    ▼                                    ▼
1. Row Normalization                 2. Column Normalization
   A = A / sum(A, axis=1)               A = A / sum(A, axis=0)
    │                                    │
    └──────── Repeat N Iterations ───────┘
          │
          ▼
Doubly Stochastic Matrix H (Rows & Cols sum to 1.0)
```

The **Sinkhorn-Knopp algorithm** iteratively normalizes the rows and columns of a strictly positive matrix to project it onto the doubly stochastic manifold.

`sinkhorn_iterations` sets the number of alternating row/column normalization steps $N$ executed in the forward pass.

---

## 2. Mechanics & Convergence

- **Convergence Rate**: The Sinkhorn-Knopp algorithm exhibits linear convergence. For small matrices (e.g. $k=4$), $N = 20$ iterations project the matrix within machine precision ($\approx 10^{-6}$) of exact row and column sums equal to $1.0$.
- **Backpropagation**: Gradients flow backward through all $N$ normalization steps via unrolled automatic differentiation in JAX.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `sinkhorn_iterations` | `int` | `20` | Positive integer (e.g. `10`, `20`, `30`) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `mhc_expansion_rate` | Sets the matrix size $k \times k$. Larger $k$ may require more iterations for tight convergence. |
| `enable_mhc_lite` | When `enable_mhc_lite: true`, Sinkhorn iterations are bypassed entirely in favor of closed-form permutation combinations. |

---

## 5. Practical Guidance & Failure Modes

| Iterations | Convergence & Compute Impact |
| :--- | :--- |
| `sinkhorn_iterations: 20` (Default) | Highly accurate projection; standard for DeepSeek mHC models without mHC-lite. |
| `sinkhorn_iterations: 5` or `10` | Faster forward/backward pass, but minor row/column sum drift can accumulate numerical energy error across 60+ layers. |
| When `enable_mhc_lite: true` | `sinkhorn_iterations` is ignored because mHC-lite constructs doubly stochastic matrices analytically. |

---

### One-line intuition

> `sinkhorn_iterations` controls the number of alternating row and column normalizations in the Sinkhorn-Knopp algorithm, ensuring the mHC stream mixing matrix strictly conserves representation energy.
