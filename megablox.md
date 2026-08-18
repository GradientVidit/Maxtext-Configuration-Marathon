
## 1. The problem MegaBlocks solves

After routing, tokens are distributed unevenly across experts:

```text
expert_0: 47 tokens   ← heavy
expert_1:  3 tokens   ← light
expert_2: 31 tokens
expert_3: 19 tokens
```

The naive approach pads every expert's batch to the maximum observed load:

```text
padded batch per expert: [47, 47, 47, 47]
→ wasted compute on padding = (47-3)/47 ≈ 94% waste for expert_1
```

MegaBlocks avoids this by using **block-sparse matrix multiplication** — the variable-length expert batches are packed into a ragged structure and processed with a single grouped matmul (GMM) that handles variable-length groups natively, with no padding.

---

## 2. What `megablox` actually controls

```yaml
megablox: true   # use MegaBlocks block-sparse GMM kernel
megablox: false  # use standard padded dense matmul per expert
```

When `true`, MaxText routes through the MegaBlocks-style GMM path — block-sparse matmul that handles ragged expert batch sizes without padding.

When `false`, falls back to a dense padded approach.

---

## 3. Data flow comparison

**Without MegaBlocks (`megablox: false`):**
```text
tokens → sort by expert assignment
       → pad each expert's batch to max_capacity
       → N separate dense matmuls (one per expert)
       → discard padding, reassemble
```

**With MegaBlocks (`megablox: true`):**
```text
tokens → sort by expert assignment
       → pack into block-sparse ragged layout
       → single grouped matmul (GMM) over all experts
       → reassemble (no padding to discard)
```

---

## 4. Interaction with `sparse_matmul`

These two flags combine to select the MoE compute path:

| `megablox` | `sparse_matmul` | Effective path |
|---|---|---|
| `true` | `true` (default) | MegaBlocks sparse GMM — primary production path |
| `true` | `false` | Dense padded matmul (megablox setting overridden by sparse_matmul=false) |
| `false` | `true` | Alternative sparse path (non-megablox) |
| `false` | `false` | Pure dense padded matmul |

In practice, `megablox=true, sparse_matmul=true` is the standard configuration.

---

## 5. Interaction with GMM tiling params

When `megablox=true`, the `wi_tile_fwd_*` and `wo_tile_fwd_*` params control GMM tiling for the forward pass. The MegaBlocks/JAX ragged-dot backend supports only the 6 forward-pass tiling configs; backward-pass tiling requires switching to the Tokamax backend (`use_tokamax_gmm=true`).

---

## 6. When to change it

**Leave at `true`** in virtually all cases — it's the production-tested default and is meaningfully faster than padded dense matmul for any imbalanced routing.

**Set to `false` if:**
- Debugging correctness: dense path is simpler and easier to inspect
- Using a specialized alternative kernel path (e.g. `use_tokamax_gmm=true` may alter which path is actually taken)
- Profiling to quantify the speedup MegaBlocks provides on your specific hardware/model shape

---

## 7. Default

```yaml
megablox: true
```

Confirmed in `base.yml`.

---

### One-line intuition

> **`megablox` switches MoE expert computation from padded dense matmuls (which waste compute on lightly-loaded experts) to a block-sparse grouped matmul that handles ragged expert batch sizes natively with no padding overhead.**
