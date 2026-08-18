
## 1. Why does this exist?

`jax.lax.scan` is how MaxText (when `scan_layers=true`) avoids unrolling the decoder stack — it requires every iteration of the scanned loop to be **structurally identical**: same operations, same shapes, same parameters layout.

That assumption breaks for architectures with **repeating but non-uniform layer sequences**, like Llama4 Maverick:

```text
Layer 0: dense + RoPE
Layer 1: MoE + RoPE
Layer 2: dense + RoPE
Layer 3: MoE + NoPE
Layer 4: dense + RoPE   ← cycle restarts
Layer 5: MoE + RoPE
...
```

Each individual layer is not interchangeable with the next. But the *cycle* of 4 layers repeats identically. `jax.lax.scan` can still work — you just have to scan over **groups of 4 layers** (the full cycle) instead of individual layers.

That's exactly what `inhomogeneous_layer_cycle_interval` controls: it tells MaxText the length of that repeating but internally heterogeneous block.

---

## 2. Mechanics

```text
num_decoder_layers = 48
inhomogeneous_layer_cycle_interval = 4

Scan operates over: 48 / 4 = 12 iterations
Each iteration processes: layers [4k, 4k+1, 4k+2, 4k+3]

JAX sees: 12 identical "super-layer" blocks → scan is valid
MaxText sees: 48 layers of varying type → but grouped correctly
```

With `inhomogeneous_layer_cycle_interval = 1` (default), each layer is scanned individually — valid only when all layers are truly structurally identical.

---

## 3. Options

| Value | Meaning |
|---|---|
| `1` | Default — all decoder layers are identical; scan over individual layers |
| `N > 1` | Layers repeat with period N; scan over N-layer blocks |

Must satisfy: `num_decoder_layers % inhomogeneous_layer_cycle_interval == 0`.

---

## 4. What breaks if wrong

Setting this to `1` when layers are actually inhomogeneous causes `jax.lax.scan` to fail or silently mis-trace — XLA will see structurally different operations across scan iterations and either error on shape mismatch or produce incorrect computation by coercing mismatched pytree structures.

Setting it to `N` when layers are actually homogeneous is harmless (just slightly less efficient scan unrolling), but confusing.

---

## 5. Interaction with pipeline parallelism

Pipeline parallelism has its own layer grouping (`num_layers_per_pipeline_stage`). These interact:

```text
inhomogeneous_layer_cycle_interval = 4
num_layers_per_pipeline_stage = 4

→ Each pipeline stage contains exactly one cycle: safe.

inhomogeneous_layer_cycle_interval = 4
num_layers_per_pipeline_stage = 2

→ Each stage contains half a cycle: heterogeneous within stage.
   This may violate scan assumptions inside the stage loop.
```

Best practice: set `num_layers_per_pipeline_stage` to be a multiple of `inhomogeneous_layer_cycle_interval`.

---

## 6. Practical scenarios

**Llama4 Maverick (dense+rope, moe+rope, dense+rope, moe+nope):**
```yaml
inhomogeneous_layer_cycle_interval: 4
```

**Standard dense model (Llama2, Gemma, etc.) — all layers identical:**
```yaml
inhomogeneous_layer_cycle_interval: 1  # default; nothing to set
```

**Custom MoE with 8-layer interleaving pattern:**
```yaml
inhomogeneous_layer_cycle_interval: 8
```

---

### One-line intuition

> **`inhomogeneous_layer_cycle_interval` tells MaxText how many consecutive layers form one repeating block, so `jax.lax.scan` can iterate over full cycles instead of individual structurally-dissimilar layers.**
