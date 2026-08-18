
## 1. Why does `global_parameter_scale` exist?

When scaling a transformer, you don't just change one dimension — you change all of them together, in proportions that preserve the model's aspect ratio:

```text
emb_dim:   2048 → 4096 → 8192
num_heads: 16   → 32   → 64
mlp_dim:   7168 → 14336 → 28672
layers:    16   → 32   → 64
```

Without `global_parameter_scale`, doing this requires editing 5+ separate parameters. `global_parameter_scale` is a single scalar multiplier that scales all of them at once, for quick "make it 2× bigger" experiments.

---

## 2. How it works

MaxText computes effective dimensions as:

```text
emb_dim   = base_emb_dim   × global_parameter_scale
num_heads = base_num_query_heads × global_parameter_scale
kv_heads  = base_num_kv_heads   × global_parameter_scale
mlp_dim   = base_mlp_dim        × global_parameter_scale
layers    = base_num_decoder_layers × global_parameter_scale
```

So with defaults (`base_emb_dim=2048`, `scale=1`):

```text
scale=1  → 2048 emb, 16 heads, 7168 mlp, 16 layers  (default ~117M params)
scale=2  → 4096 emb, 32 heads, 14336 mlp, 32 layers (~1B params)
scale=4  → 8192 emb, 64 heads, 28672 mlp, 64 layers (~7B params)
```

---

## 3. The power-of-2 constraint

```yaml
global_parameter_scale: 4   # OK
global_parameter_scale: 3   # INVALID — must be power of 2
```

The constraint exists because the scaled dimensions must divide evenly across sharding axes. If `base_emb_dim=2048` and you scale by 3, you get 6144 — which may not divide cleanly across e.g. 8-way tensor parallelism. Powers of 2 always compose cleanly with 2^k sharding.

---

## 4. The explicit-dims alternative

`global_parameter_scale` is the coarse knob. When you need a specific target (e.g., exactly 7B parameters):

```yaml
# Coarse: scale everything together
global_parameter_scale: 4

# Fine: set each dimension explicitly
base_emb_dim: 4096
base_num_query_heads: 32
base_num_kv_heads: 8          # GQA: 8 KV heads, 32 Q heads
base_mlp_dim: 11008
base_num_decoder_layers: 32
head_dim: 128
```

Real pretraining configs for specific models always use the explicit dims.  `global_parameter_scale` is mostly useful for quick scaling experiments or sweep scripts.

---

## 5. Default

```yaml
global_parameter_scale: 1
```

Scale of 1 = use the `base_*` dims exactly as specified. Since the base dims define a ~117M parameter model, this is MaxText's default "sanity check" size.

---

## 6. Options

| Value | Typical use |
|---|---|
| `1` | Default — use base dims directly |
| `2` | 2× all dims — quick "medium" experiment |
| `4` | 4× all dims — quick "large" experiment |
| `8`, `16`, ... | Larger; must be power of 2 |

---

## 7. What breaks

Setting a non-power-of-2 value raises an error in MaxText's config validation. Setting it too large without also increasing `head_dim` can produce an unusual model shape (very deep and wide, but with tiny per-head dimension).

---

### One-line intuition

> **`global_parameter_scale` is a single power-of-2 multiplier that uniformly scales all model dimensions together — a fast "make the model N× bigger" knob for experiments, replaced by explicit `base_*` dims in production.**
