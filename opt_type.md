## 1. Why does `opt_type` exist?

Different neural network architectures, training budgets, and convergence goals benefit from different optimization algorithms:

- **AdamW:** The industry standard for Transformers; tracks first and second moments with decoupled weight decay.
- **Adam PAX:** PAX-compatible implementation of Adam for exact cross-framework reproducibility.
- **SGD:** Simple stochastic gradient descent with momentum; lower memory footprint.
- **Muon:** Momentum Orthogonalized Optimizer; applies Newton-Schulz matrix iterations to 2D weight matrices for faster convergence.

```text
                     opt_type Selection
                             │
     ┌──────────────┬────────┴────────┬──────────────┐
     ▼              ▼                 ▼              ▼
  "adamw"       "adam_pax"          "sgd"          "muon"
 (Standard)    (PAX legacy)     (Low memory)    (Orthogonalized
                                                 matrix update)
```

`opt_type` selects the mathematical optimization algorithm used to update model parameters from accumulated gradients.

---

## 2. Fundamentals & Mechanics

- **`"adamw"` (Default):** Standard Optax implementation of decoupled weight decay Adam.
- **`"adam_pax"`:** Faithful port of the PAX framework's Adam implementation (handling epsilon roots and scaling conventions).
- **`"sgd"`:** Classic stochastic gradient descent with momentum.
- **`"muon"`:** State-of-the-art matrix optimizer using iterative polar decomposition (Newton-Schulz iterations) on hidden-state 2D weight matrices while routing 1D vectors/embeddings to AdamW.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `"adamw"` | Standard decoupled AdamW optimizer. |
| PAX Compatibility | `"adam_pax"` | PAX-style Adam implementation. |
| Low Footprint | `"sgd"` | Classic momentum SGD. |
| Advanced Fast Matrix | `"muon"` | Muon orthogonalized matrix optimizer for 2D weights. |

---

## 4. Interactions & Dependencies

```text
                         opt_type
                            │
           ┌────────────────┴────────────────┐
           ▼                                 ▼
       "adamw"                            "muon"
           │                                 │
adam_b1, adam_b2, adam_eps,       muon_beta, muon_weight_decay,
adam_weight_decay, adamw_mask     muon_consistent_rms, use_qk_clip
```

- **`opt_type: "muon"`:** Paired frequently with `use_qk_clip: true` and `qk_clip_threshold` to stabilize QK representations during high-rate matrix updates.

---

## 5. Practical Scenarios & Failure Modes

- **Standard Pretraining:** Use `"adamw"` for proven stability and hyperparameter transferability.
- **Muon Matrix Optimization:** Use `"muon"` to accelerate convergence (often achieving target loss in 30–40% fewer steps), ensuring Muon hyperparameters (`muon_beta`, `muon_consistent_rms`) are configured.

---

### One-line intuition

> **`opt_type` selects the underlying optimizer algorithm, switching between standard `"adamw"`, PAX-compatible Adam, `"sgd"`, or the orthogonalized `"muon"` optimizer.**
