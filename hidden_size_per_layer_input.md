## 1. Why does `hidden_size_per_layer_input` exist?

On-device and edge-oriented compact language models (such as the **Gemma 4 small E2B and E4B variants**) operate under severe parameter budgets. To maximize expressiveness across layers without ballooning the width of the main residual trunk, Gemma 4 introduces **Per-Layer Embeddings (PLE)**.

In standard architectures, input token IDs are embedded once at the very beginning of the model:

$$\text{Standard:}\quad h_0 = \text{Embedding}(x_{\text{tokens}}) \quad (h_0 \in \mathbb{R}^{S 	imes d_{model}})$$

With Per-Layer Embeddings, each individual decoder layer receives an additional layer-specific embedding vector looked up from a dedicated embedding table:

$$h_l' = h_l + \text{PLE}_l(x_{\text{tokens}}) \quad (\text{where } \text{PLE}_l \in \mathbb{R}^{V_{ple} 	imes d_{ple}})$$

```text
Standard Transformer Layer:
  Token IDs ──> [Embedding] ──> Layer 0 ──> Layer 1 ──> ... ──> Layer L

Gemma 4 Small (with Per-Layer Embeddings):
  Token IDs ──┬──> [Main Embedding]  ──────────────> Layer 0 ──> Layer 1 ──> Layer L
              ├──> [PLE Table Layer 0] ───────────────▲          │          │
              ├──> [PLE Table Layer 1] ──────────────────────────▲          │
              └──> [PLE Table Layer L] ─────────────────────────────────────▲
```

`hidden_size_per_layer_input` sets the feature dimension ($d_{ple}$) of these per-layer input embeddings.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `0` | Disabled. Standard single input embedding table. | **Default**. Standard for all general models (LLaMA, standard Gemma 2, Mistral). |
| Any integer $> 0$ (e.g. `256`) | Enables Per-Layer Embedding sub-blocks in `gemma4_small.py` with feature width $d_{ple}$. | Must be paired with `vocab_size_per_layer_input > 0`. |

Default in `base.yml`: `0`

---

## 3. The `scan_layers: false` Architectural Constraint

In MaxText, standard transformer layers are evaluated inside a JAX loop primitive (`jax.lax.scan`) when `scan_layers: true`. 

However, because Per-Layer Embeddings require each layer to index into distinct per-layer parameter tensors with heterogeneous layer-dependent states, **models utilizing PLE must be executed with `scan_layers: false` (unrolled layers)**. Attempting to scan layers with PLE enabled will cause compilation or shape tracing errors.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[vocab_size_per_layer_input]] | Required companion: defines the vocabulary length $V_{ple}$ of the per-layer embedding table. |
| [[decoder_block]] | Specifically implemented within `decoder_block: 'gemma4_small'` (`models/gemma4_small.py`). |
| [[scan_layers]] | Must be set to `scan_layers: false` when PLE is enabled. |
| [[num_kv_shared_layers]] | Often paired together in Gemma 4 small configurations to rebalance the parameter budget. |

---

## 5. Practical Scenarios

- **Training Gemma 4 E2B / E4B Edge Models:** Set `hidden_size_per_layer_input: 256` (or architecture target) along with matching `vocab_size_per_layer_input`.
- **Standard LLM Pretraining (LLaMA 3 / Gemma 2):** Leave at `0`.

---

### One-line intuition

> **`hidden_size_per_layer_input` specifies the embedding dimension $d_{ple}$ for Gemma 4's Per-Layer Embedding mechanism, injecting learned per-layer token representations directly into each decoder layer.**
