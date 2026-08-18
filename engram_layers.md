## 1. Why does `engram_layers` exist?

Transformers memorize multi-word expressions, entity phrases, idioms, and code snippets by dedicating significant MLP feedforward capacity to storing factual associations.

**Engram** introduces explicit non-parametric N-gram memory lookups into specific transformer layers:

```text
Layer 0: Standard Transformer Block
Layer 1: Transformer Block + Engram N-gram Memory Module (e.g. index in engram_layers)
Layer 2: Standard Transformer Block
Layer 3: Standard Transformer Block
Layer 4: Transformer Block + Engram N-gram Memory Module (e.g. index in engram_layers)
...
```

Instead of attaching costly N-gram memory tables to every single layer (which would redundantly consume parameter memory and table bandwidth), DeepSeek Engram selectively augments designated early-to-mid transformer layers.

`engram_layers` defines the list of 0-indexed transformer layer positions where the Engram module is instantiated and attached.

An empty list `[]` disables Engram entirely.

---

## 2. Mechanics & Integration

For each layer index $l \in$ `engram_layers`:
1. The layer takes the input sequence tokens, constructs rolling $n$-grams (from unigrams up to `engram_max_ngram_size`), and hashes them into hash-table embedding tables.
2. The retrieved table embeddings are processed with a 1D convolution (`engram_kernel_size`), projected via multi-head projections (`engram_num_heads`, `engram_head_dim`), and fused into the layer's hidden state.
3. For layer indices not in `engram_layers`, standard transformer computation proceeds with zero Engram overhead.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `engram_layers` | `list[int]` | `[]` | List of 0-indexed layer integers (e.g. `[1, 4]`, `[2, 6, 10]`, or `[]` to disable) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `engram_max_ngram_size` / `engram_vocab_bases` | Configures the hash tables allocated for each layer specified in `engram_layers`. |
| `base_num_decoder_layers` | Layer indices in `engram_layers` must be strictly $< \text{base\_num\_decoder\_layers}$. |
| `remat_policy` | Setting `remat_policy: 'custom'` allows fine-grained activation saving for `engram` activations. |

---

## 5. Practical Guidance & Best Practices

| Architecture Setting | Recommended Value | Rationale |
| :--- | :--- | :--- |
| **Engram Disabled (Standard Transformer)** | `engram_layers: []` | Default for standard models without N-gram memory lookup. |
| **DeepSeek Engram Benchmark** | `engram_layers: [1, 4]` | Injects N-gram lookup at layers 2 and 5 (0-indexed), where early linguistic phrase representations benefit most from explicit lexical memory. |
| **Invalid Index (e.g. `[16]` for 16-layer model)** | IndexError | Index must be within $[0, \text{base\_num\_decoder\_layers} - 1]$. |

---

### One-line intuition

> `engram_layers` specifies the transformer layer indices where Engram N-gram memory modules are attached, keeping Engram disabled when set to an empty list `[]`.
