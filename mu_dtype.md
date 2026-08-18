## 1. Why does `mu_dtype` exist?

The AdamW first-moment buffer ($m_t$, often called "mu") requires the same tensor shape as model weights. For large models (e.g. 70B parameters), storing optimizer states in `float32` consumes massive HBM:

- Model weights in `bfloat16`: 140 GB
- First moment ($m_t$) in `float32`: 280 GB
- Second moment ($v_t$) in `float32`: 280 GB
- **Total Optimizer State:** 560 GB

```text
First-Moment Memory Footprint:
  mu_dtype: "float32"   ──> 4 bytes/parameter (Full precision, highest HBM)
  mu_dtype: "bfloat16"  ──> 2 bytes/parameter (50% HBM savings for first moment)
```

`mu_dtype` configures the storage data type for the first-moment ("mu") momentum buffer in AdamW.

---

## 2. Fundamentals & Mechanics

- **Default `""` (Empty String):** Inherits data type from `weight_dtype` (typically `float32` or `bfloat16`).
- Enables downcasting the momentum tracking buffer to `bfloat16` to conserve accelerator memory on memory-constrained hardware meshes.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `""` | Inherits from `weight_dtype` (typically `float32`). |
| Memory Optimized | `"bfloat16"` | Stores first moment in 16-bit brain float, saving 2 bytes per param. |
| Full Precision | `"float32"` | Standard 32-bit floating point storage. |

---

## 4. Interactions & Dependencies

```text
mu_dtype ──> Controls first-moment memory buffer precision in AdamW & Muon
```

- Second moment ($v_t$) precision in Optax currently inherits from weight precision and is not independently exposed.

---

## 5. Practical Scenarios & Failure Modes

- **Fitting Large Models on Fewer Chips:** Switching `mu_dtype: "bfloat16"` can free tens of gigabytes of HBM, avoiding Out-Of-Memory (OOM) errors during large batch training.

---

### One-line intuition

> **`mu_dtype` sets the numerical precision for storing the AdamW first-moment momentum buffer, trading precision for reduced HBM footprint.**
