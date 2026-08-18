## 1. Why does `use_qk_norm_in_gdn` exist?

Linear attention and delta-rule recurrence update a continuous state matrix across steps:

$$S_t = S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$

The term $(I - \beta_t k_t k_t^T)$ acts as a memory erasure / transition operator. Consider its eigenvalues:
- If $\|k_t\|_2^2 > 1$ and $\beta_t \approx 1$, $(1 - \beta_t \|k_t\|^2)$ can become negative, causing oscillations or unbounded state growth.
- If $\|k_t\|_2$ fluctuates widely across layers or sequence positions, the recurrent memory matrix either explodes exponentially or vanishes to zero.

```text
Without QK Norm:
Unbounded ||k_t|| ──> Eigenvalues of (I - β k k^T) fall outside [0, 1] ──> Numerical Instability / Exploding State

With QK Norm (use_qk_norm_in_gdn: true):
||q_t||_2 = 1, ||k_t||_2 = 1 ──> Eigenvalues strictly bounded in [0, 1] ──> Provably Stable Contractive Recurrence
```

`use_qk_norm_in_gdn` enforces L2 normalization (or per-head RMSNorm) on query and key vectors inside the Gated Delta Rule kernel, guaranteeing numerical stability across long sequences.

---

## 2. Mechanics & Mathematical Formulation

When `use_qk_norm_in_gdn: true`:

1. **Key Normalization**:
   $$\tilde{k}_t = \frac{k_t}{\|k_t\|_2 + \epsilon}$$
   Because $\|\tilde{k}_t\|_2 = 1$, the erasure projection operator $(I - \beta_t \tilde{k}_t \tilde{k}_t^T)$ has singular values strictly bounded within $[1 - \beta_t, 1]$. With gating factor $\beta_t \in [0, 1]$, the state transition is strictly contractive:

   $$\|S_t\| \le \|S_{t-1}\| + \|v_t\|$$

2. **Query Normalization**:
   $$\tilde{q}_t = \frac{q_t}{\|q_t\|_2 + \epsilon}$$
   Ensures that memory readouts $\tilde{q}_t^T S_{t-1}$ remain scaled consistently with the value vector magnitude.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `use_qk_norm_in_gdn` | `bool` | `true` | `true` (normalize Q and K), `false` (unnormalized) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `gdn_key_head_dim` | L2 normalization is computed over this head dimension ($d_k$). |
| `normalization_layer_epsilon` | Used as the numerical stability constant $\epsilon$ during vector norm computation. |
| `partial_rotary_factor` | Rotary position embedding is applied before or in coordination with head normalization. |

---

## 5. Practical Guidance & Failure Modes

| Setting | Training Dynamics | When to use |
| :--- | :--- | :--- |
| `use_qk_norm_in_gdn: true` (Default) | Rock-solid numerical stability across millions of tokens; prevents gradient spikes in deep recurrent networks. | **Recommended for all Qwen3-Next runs and DeltaNet variants.** |
| `use_qk_norm_in_gdn: false` | Recurrent state $S_t$ is prone to eigenvalue drift, FP16/BF16 overflow, and loss divergence. | Only for reproducing legacy unnormalized linear attention baselines. |

---

### One-line intuition

> `use_qk_norm_in_gdn` normalizes query and key vectors in Gated DeltaNet, ensuring the recurrent transition operator is contractive and mathematically preventing associative memory explosion.
