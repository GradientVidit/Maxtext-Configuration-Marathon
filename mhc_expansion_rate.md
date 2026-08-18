## 1. Why does `mhc_expansion_rate` exist?

Standard deep neural networks rely on single-stream residual connections ($x_{l+1} = x_l + F(x_l)$). As network depth grows to hundreds of layers, single-stream residuals can suffer from feature collapse, gradient dissipation, or representational bottlenecks.

**Hyper-Connections (HC)** generalize the residual stream by maintaining $k$ parallel residual streams:

```text
Standard Residual (k = 1):
Layer Input x_l ───────┬────────────────────────(+)───> x_{l+1}
                       │                         ▲
                       └────> Layer Sub-block ───┘

Hyper-Connections (mhc_expansion_rate = k, e.g. k = 4):
Stream 0: x^{(0)}_l ──┐
Stream 1: x^{(1)}_l ──┼───> Doubly Stochastic Mixing Matrix H (k × k) ───(+)───> x^{(0)}_{l+1} ... x^{(k-1)}_{l+1}
Stream 2: x^{(2)}_l ──┤                 ▲                                  ▲
Stream 3: x^{(3)}_l ──┘                 │                                  │
                               Learnable Projections              Layer Sub-block Output
```

`mhc_expansion_rate` specifies the expansion factor $k$—the number of parallel residual streams flowing across the transformer layers.

Setting `mhc_expansion_rate: 1` reduces the topology to a standard single-stream transformer.

---

## 2. Mechanics & Manifold Constraint

1. **State Expansion**: At the model embedding layer, the initial token embedding of shape `[Batch, Seq_Len, Hidden_Dim]` is mapped or replicated across $k$ streams to shape `[Batch, Seq_Len, k, Hidden_Dim]`.
2. **Layer Mixing**: At each layer $l$, the $k$ streams are mixed using a transition matrix $H \in \mathbb{R}^{k \times k}$:
   $$\mathbf{X}_{l+1}^{\text{res}} = H \mathbf{X}_l$$
3. **Manifold Constraint**: To prevent signal explosion or decay across depth, $H$ is constrained to be **doubly stochastic** (all rows and columns sum to 1, with elements $\ge 0$). This is enforced via Sinkhorn-Knopp iterations or mHC-lite.
4. **Sub-block Injection**: Layer sub-blocks (Attention and MLP) process a linear aggregation of the streams and project their residual output back across all $k$ streams.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `mhc_expansion_rate` | `int` | `1` | Integer $\ge 1$ (e.g. `1`, `2`, `4`) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `sinkhorn_iterations` | Determines the number of normalization iterations used to project the $k \times k$ mixing matrix when `enable_mhc_lite: false`. |
| `enable_mhc_lite` | When `true`, constructs the doubly stochastic matrix using convex combinations of permutation matrices ($k!$ permutations). Practical for $k \le 4$. |
| `base_emb_dim` | Total residual memory bandwidth scales with `k * base_emb_dim`. |

---

## 5. Practical Scenarios & Performance Trade-offs

| Value | Topology | Memory & Compute | Use Case |
| :--- | :--- | :--- | :--- |
| `mhc_expansion_rate: 1` (Default) | Standard Transformer | 1× residual stream memory bandwidth | Default for LLaMA, Gemma, DeepSeek-V3 standard architectures. |
| `mhc_expansion_rate: 4` | DeepSeek Hyper-Connection (4 streams) | 4× residual activation buffer; higher multi-stream expressive capacity | DeepSeek-style multi-stream architectures; pairs well with `enable_mhc_lite: true`. |
| `mhc_expansion_rate > 4` | Deep Hyper-Connection (e.g. 8 streams) | Significant HBM activation footprint; requires standard Sinkhorn iterations (factorial permutation explodes in mHC-lite). | Advanced architectural research on deep residual flow. |

---

### One-line intuition

> `mhc_expansion_rate` defines the number of parallel residual streams in Manifold-Constrained Hyper-Connections, enabling multi-lane signal propagation across transformer layers.
