## 1. Why does `vocab_size_per_layer_input` exist?

When Per-Layer Embeddings (PLE) are enabled in Gemma 4 small edge architectures (`hidden_size_per_layer_input > 0`), the model requires a vocabulary dimension for the auxiliary per-layer embedding matrices:

$$\text{PLE Tensor Shape} = [N_{\text{layers}}, V_{\text{ple}}, d_{\text{ple}}]$$

$$\text{Where:}\quad V_{\text{ple}} = \text{vocab\_size\_per\_layer\_input},\quad d_{\text{ple}} = \text{hidden\_size\_per\_layer\_input}$$

```text
Input Tokens: [batch, seq_len]
        │
        ▼
[ PLE Embedding Lookup ] (Table Shape: [Layers, vocab_size_per_layer_input, hidden_size_per_layer_input])
        │
        ▼
Per-Layer Injected Vectors [Layers, batch, seq_len, hidden_size_per_layer_input]
```

`vocab_size_per_layer_input` specifies the number of discrete token entries ($V_{ple}$) recognized by the per-layer embedding tables.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `0` | Disabled. | **Default**. |
| Any integer $> 0$ (e.g. `256_000`) | Sets the vocabulary length $V_{ple}$ for per-layer embeddings. | Typically matches or is a compressed subset of the primary tokenizer `vocab_size`. |

Default in `base.yml`: `0`

---

## 3. Separation from Main Tokenizer Vocab Size

While `vocab_size_per_layer_input` often matches the main tokenizer vocabulary size (e.g. 256,000 for Gemma), it can be configured independently:
- **Full Tokenizer Matching:** Every token in the vocabulary has a dedicated learned vector per layer.
- **Sub-Vocabulary / Byte-Level Embedding:** For ultra-compact models, PLE can operate over a smaller sub-vocabulary (e.g. 256 byte tokens) to drastically reduce parameter storage.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[hidden_size_per_layer_input]] | Prerequisite companion. Both parameters must be positive integers to instantiate `Gemma4SmallPLE`. |
| [[scan_layers]] | Must be `false` due to heterogeneous per-layer parameter access. |
| [[decoder_block]] | Active when using `decoder_block: 'gemma4_small'`. |

---

## 5. Practical Scenarios

- **Reproducing Gemma 4 E2B / E4B Checkpoints:** Set `vocab_size_per_layer_input` to match the tokenizer vocabulary size defined in the model card.
- **Standard Pretraining:** Leave at `0`.

---

### One-line intuition

> **`vocab_size_per_layer_input` sets the number of vocabulary entries $V_{ple}$ in Gemma 4's Per-Layer Embedding tables, determining the token index range for per-layer embedding lookups.**
