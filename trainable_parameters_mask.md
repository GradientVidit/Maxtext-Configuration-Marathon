## 1. Why does `trainable_parameters_mask` exist?

In parameter-efficient fine-tuning (PEFT), indexer warmups (e.g. Dynamic Sparse Attention), or selective transfer learning, developers want to update only a subset of model weights while keeping the rest frozen:

```text
Model Parameters:
┌──────────────────────────────┬──────────────────────────────┐
│       Backbone Layers        │       DSA Indexer / LoRA     │
│   (Decoder / Attention / MLP)│      (Specialized Modules)   │
└──────────────────────────────┴──────────────────────────────┘
               │                               │
  Regex: '.*indexer.*'                         │
  [DO NOT MATCH]                           [MATCHES]
               │                               │
               ▼                               ▼
       Gradients Zeroed /              Gradients Computed /
       Parameters FROZEN               Parameters UPDATED
```

`trainable_parameters_mask` provides a list of regex patterns specifying which parameter paths receive gradient updates; all unmatched parameters are frozen.

---

## 2. Fundamentals & Mechanics

During optimizer initialization:
1. MaxText flattens the PyTree parameter dictionary into dotted path keys (e.g. `decoder/layers_0/indexer/kernel`).
2. If `trainable_parameters_mask` is empty (`[]`), all parameters are trainable.
3. If non-empty, MaxText matches parameter keys against the supplied regex list.
4. Parameters that match any pattern are marked trainable; non-matching parameters have their gradients zeroed out or masked from optimizer state updates.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `[]` | Empty list: all model parameters are trainable (full training). |
| Indexer Only | `['.*indexer.*']` | Freezes backbone, training only Dynamic Sparse Attention indexers. |
| LoRA Only | `['.*lora.*']` | Trains only low-rank adapter weights. |
| Heads Only | `['.*output_layer.*', '.*embedding.*']` | Trains only input embeddings and output projection heads. |

---

## 4. Interactions & Dependencies

```text
trainable_parameters_mask
            │
            ├─ Matches Regex ──> Optimizer calculates momentums & updates weights
            └─ Unmatched     ──> Optimizer ignores / gradients zeroed
```

- **DSA Dense Warmup:** Used in Dynamic Sparse Attention training to warm up the indexer network before enabling sparse token routing.

---

## 5. Practical Scenarios & Failure Modes

- **Typos in Regex Patterns:** If `trainable_parameters_mask: ['indexer']` without leading/trailing wildcards fails to match `decoder/layers_0/indexer/kernel`, zero parameters will train, resulting in flat loss curves. Always use `['.*indexer.*']`.

---

### One-line intuition

> **`trainable_parameters_mask` defines regex patterns to selectively train specific model sub-modules while freezing the remainder of the network.**
