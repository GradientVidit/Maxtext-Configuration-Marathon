## 1. Why does `enable_nnx` exist?

Flax originally utilized the **Linen** module API (`flax.linen`), which relied on functional variable management, implicit PRNG sequencing, and separate `init` / `apply` state dictionaries.

Google and the Flax team introduced **Flax NNX**, a modern, Pythonic object-oriented API that treats weights, modules, and optimizer states as standard mutable Python attributes (similar to PyTorch) while preserving JAX's pure functional compilation semantics (`jax.jit`, `jax.grad`, `shard_map`).

MaxText has transitioned its core architecture to Flax NNX to improve developer ergonomics, simplify model surgery, and streamline dynamic multimodal routing.

```text
Flax Linen (Legacy):
variables = model.init(rng, x)
out, state = model.apply(variables, x, mutable=['cache'])

Flax NNX (Modern MaxText):
model = TransformerDecoder(cfg, rngs=nnx.Rngs(0))
out = model(x) # Clean, direct, Pythonic!
```

`enable_nnx` is the master switch enabling NNX-based module architectures across MaxText.

---

## 2. Options and Defaults

| Value | API Framework | Status |
|---|---|---|
| `true` (Default) | Flax NNX (Modern object-oriented Flax) | Standard default across all modern MaxText configs |
| `false` | Flax Linen (Legacy functional API) | Legacy compatibility mode for historical Linen checkpoints |

---

## 3. Key Interactions

- **`pure_nnx`**: Ensures the entire model stack (embeddings, layers, heads, checkpointers) runs purely under NNX.
- **`pure_nnx_decoder`**: Dictates whether the decoder specifically uses NNX.

---

### One-line intuition
> **`enable_nnx` toggles the use of modern Flax NNX object-oriented model implementations over legacy Flax Linen.**
