## 1. Why does `indexer_sparse_training` exist?

Training a neural token indexer introduces an optimization dilemma:
- If the main model attends strictly to the indexer's top-$k$ selections from step 0, an untrained indexer will pick random tokens, producing corrupt gradients that ruin the pretrained language model.
- If the indexer is trained purely offline, it never adapts to the evolving representations of the language model.

**DSA solves this with a two-phase training strategy** controlled by `indexer_sparse_training`:

```text
Phase 1: Dense Warm-up (indexer_sparse_training = False)
┌────────────────────────────────────────────────────────────────────────────┐
│ • Indexer computes scores over ALL tokens in the sequence.                 │
│ • Auxiliary KL-divergence loss compares indexer scores to dense attention.  │
│ • Main model is FROZEN via trainable_parameters_mask.                      │
│ • Goal: Train indexer until it accurately mimics dense attention routing.  │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
Phase 2: Sparse Training (indexer_sparse_training = True)
┌────────────────────────────────────────────────────────────────────────────┐
│ • Main attention evaluates ONLY the top-k tokens selected by the indexer.  │
│ • Indexer inputs are DETACHED (gradients don't propagate into trunk).      │
│ • Main model and indexer train simultaneously in true sparse mode.         │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Options & Defaults

| Value | Training Mode | Main Attention Scope | Indexer Input Gradient | Notes |
|---|---|---|---|---|
| `false` | **Dense Warm-up** | Evaluates all tokens | End-to-end gradients | **Default**. Used during initial indexer alignment. |
| `true` | **Sparse Training** | Evaluates only Top-$K$ tokens | Detached input | Used once indexer has converged on dense attention. |

> **Note:** `indexer_sparse_training` is active only when `indexer_loss_scaling_factor > 0.0`.

Default in `base.yml`: `false`

---

## 3. Why Detaching Gradients Matters in Sparse Mode

In Phase 2 (`indexer_sparse_training: true`), the indexer's input activations are detached (`jax.lax.stop_gradient(x)`). This prevents the main language model from "gaming" the indexer by altering its hidden representations just to minimize the indexer auxiliary loss, ensuring both components optimize their primary objectives stably.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[indexer_loss_scaling_factor]] | Must be $> 0.0$ for indexer training loss to be calculated. |
| [[use_indexer]] | Parent switch. |
| [[indexer_topk]] | Defines the sparse token set evaluated during Phase 2. |

---

## 5. Practical Scenarios

- **Warming up the Indexer on a Pretrained Checkpoint:** Set `indexer_sparse_training: false`, `indexer_loss_scaling_factor: 0.1`, and freeze the base model with `trainable_parameters_mask`.
- **Full-Scale Sparse Continual Pretraining:** Once indexer recall reaches $>90\%$, switch to `indexer_sparse_training: true` and unfreeze the base model.

---

### One-line intuition

> **`indexer_sparse_training` switches the indexer from Dense Warm-up (`false`, aligning against full attention with a frozen trunk) to Sparse Co-Training (`true`, evaluating only top-$k$ tokens with detached inputs).**
