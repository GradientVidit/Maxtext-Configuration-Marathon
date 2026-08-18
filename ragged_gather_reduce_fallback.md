
## 1. What the ragged gather-reduce does

After expert computation, tokens need to be combined: each token's k expert outputs are weighted (by routing probabilities) and summed back into the token's representation. This "combine" operation has two components:

1. **Gather**: pull the expert outputs from per-expert buffers back to per-token positions
2. **Reduce**: apply the routing weights and sum k expert outputs per token

The "ragged gather-reduce" fuses these into a single kernel — gathering and reducing in one pass, which avoids materializing all expert outputs before combining.

MaxText's SparseCore ragged-gather-reduce kernel performs this fused operation on SparseCore hardware.

`ragged_gather_reduce_fallback` is the escape hatch: when `true`, forces the plain JAX reference instead of the SparseCore kernel.

---

## 2. What it does

```yaml
ragged_gather_reduce_fallback: false  # (default) use SparseCore ragged-gather-reduce kernel
ragged_gather_reduce_fallback: true   # force JAX reference implementation
```

Same pattern as `ragged_gather_fallback`, but for the reduce/combine variant.

---

## 3. `ragged_gather_fallback` vs. `ragged_gather_reduce_fallback`

These are two separate kernel operations in the MoE combine pass:

```text
expert outputs
    ↓
ragged_gather        ← ragged_gather_fallback controls this
    ↓
per-token weighted sum (reduce) ← fused in ragged_gather_reduce_fallback
    ↓
combined token representation
```

Setting `ragged_gather_reduce_fallback=True` forces fallback only on the fused gather-reduce, while the gather itself may still use SparseCore (unless `ragged_gather_fallback=True` too).

---

## 4. Default

```yaml
ragged_gather_reduce_fallback: false
```

Use SparseCore kernel.

---

## 5. When to use it

**Debugging combine step:** if the combine step produces wrong values, force fallback here to isolate whether it's the SparseCore kernel or the gather step that's buggy.

**Hardware compatibility:** TPU versions without SparseCore.

**Performance baseline:** measure the fused kernel vs. reference.

---

## 6. Options

| Value | Behavior |
|---|---|
| `false` (default) | SparseCore ragged-gather-reduce kernel |
| `true` | JAX reference implementation |

---

### One-line intuition

> **`ragged_gather_reduce_fallback` forces the JAX reference for the fused gather-reduce in the MoE combine step instead of the SparseCore kernel — like `ragged_gather_fallback` but for the reduce/combine operation; keep `false` in production.**
