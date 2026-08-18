## 1. Why does `use_indexer` exist?

In ultra-long context sequences (e.g. 128K–1M tokens), even with Multi-Head Latent Attention (MLA) compressing KV cache storage, computing full dense attention scores across all tokens during autoregressive decoding requires $O(N)$ operations per step.

**DeepSeek Sparse Attention (DSA)** (introduced in **DeepSeek-V3.2**) integrates a learned **Indexer** module directly inside MLA layers. The Indexer acts as a lightweight, fast neural token retrieval engine that scores past tokens and selects only the top-$k$ most relevant tokens (`indexer_topk`) for the main heavy MLA attention to evaluate:

```text
Standard MLA Decode:
  Query (q) ──> Full Attention against ALL N cached tokens ──> O(N) Heavy Compute

DSA with use_indexer=True:
  Query (q) ──> Lightweight Indexer Module (64 heads × 128 dim)
                     │
                     ▼
             Scoring across N tokens (Fast, low-rank)
                     │
                     ▼
             Select Top-k Tokens (indexer_topk = 2048)
                     │
                     ▼
  Query (q) ──> Main MLA Attention computed ONLY on Top-k tokens ──> O(k) Heavy Compute
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `false` | Indexer disabled. Standard dense MLA attention is evaluated. | **Default**. |
| `true` | Enables the DSA Indexer token selection module inside MLA. | Active when `attention_type: 'mla'`. |

Default in `base.yml`: `false`

---

## 3. Two-Stage Training Recipe for the Indexer

Training an indexer from scratch requires a structured two-phase curriculum:

```text
Phase 1: Dense Warm-up (indexer_sparse_training: False)
  - Indexer computes auxiliary KL-divergence loss against full dense attention over ALL tokens.
  - The rest of the model is frozen via trainable_parameters_mask.
  - The indexer learns token relevance signals without destabilizing pretrained model representations.

Phase 2: Sparse Co-Training (indexer_sparse_training: True)
  - Indexer loss is evaluated strictly over the top-k selected tokens.
  - Indexer input is detached from the main backprop graph.
  - The main model and indexer train simultaneously in native sparse mode.
```

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[indexer_n_heads]], [[indexer_head_dim]] | Define the indexer's scoring head geometry (64 heads $	imes$ 128 dim). |
| [[indexer_topk]] | Number of tokens selected per query ($k=2048$). |
| [[indexer_loss_scaling_factor]] | Multiplier for the indexer's KL-divergence auxiliary training loss. |
| [[indexer_sparse_training]] | Toggles between Dense Warm-up (`false`) and Sparse Training (`true`). |
| [[attention_type]] | Must be `attention_type: 'mla'`. |

---

## 5. Practical Scenarios

- **Pretraining / Fine-Tuning DeepSeek-V3.2:** Set `use_indexer: true` with `indexer_topk: 2048`.
- **Inference Acceleration at 128K Context:** Indexing drops main attention compute from evaluating 131,072 tokens down to 2,048 tokens ($64	imes$ reduction per decode step).

---

### One-line intuition

> **`use_indexer=true` enables DeepSeek-V3.2's neural token indexer, dynamically selecting a sparse top-$k$ token subset ($k=2048$) before main MLA attention to achieve linear-time decoding at massive context lengths.**
