## 1. Why does `pure_nnx` exist?

While `enable_nnx` and `pure_nnx_decoder` govern individual components, `pure_nnx` enforces that the **entire end-to-end model graph**—including token embeddings, multimodal encoders, projectors, decoder layers, loss functions, and parameter sharding—is executed completely within the Flax NNX subsystem.

```text
End-to-End Pure NNX Graph (pure_nnx: true):
[ NNX Token Embeddings ] ──► [ NNX Decoder Stack ] ──► [ NNX Output Head ]
             ▲                          ▲                         ▲
             └──────────────── Pure NNX State Graph ──────────────┘
```

---

## 2. Options and Defaults

| Value | Behavior | Recommendation |
|---|---|---|
| `true` (Default) | Entire model graph is pure NNX | Standard for all active development and runs |
| `false` | Allows legacy Linen components or hybrid wrappers | Legacy checkpoint inspection only |

---

## 3. Interactions

- **`enable_nnx`** and **`pure_nnx_decoder`**: Work together as the NNX configuration trio (all defaulting to `true`).

---

### One-line intuition
> **`pure_nnx` enforces that the entire end-to-end model graph and training state operate purely within modern Flax NNX.**
