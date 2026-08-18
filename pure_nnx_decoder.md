## 1. Why does `pure_nnx_decoder` exist?

During the migration from Flax Linen to Flax NNX, complex models often underwent phased adoption where the main transformer decoder stack was migrated to NNX while legacy outer wrappers, tokenizers, or custom loss heads remained in Linen/JAX functional wrappers.

`pure_nnx_decoder` controls whether the core transformer decoder stack runs as a native pure NNX module rather than a hybrid Linen wrapper.

```text
Hybrid Mode (pure_nnx_decoder: false):
Linen Module ──► [ NNX Bridge Wrapper ] ──► NNX Decoder Layers ──► [ Bridge ]

Pure NNX Mode (pure_nnx_decoder: true):
Native NNX Module ──► Native NNX Decoder Layers ──► Output (Zero wrapper overhead!)
```

---

## 2. Options and Defaults

| Value | Behavior | Context |
|---|---|---|
| `true` (Default) | Pure native NNX decoder stack | Standard in current MaxText codebase |
| `false` | Hybrid Linen-wrapped decoder | Legacy debugging |

---

## 3. Interactions

- **`enable_nnx`**: Must be `true`.
- **`pure_nnx`**: Global counterpart.

---

### One-line intuition
> **`pure_nnx_decoder` specifies whether the core transformer decoder stack runs natively in pure Flax NNX without Linen compatibility bridges.**
