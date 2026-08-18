## 1. Why does `local_rope_max_timescale` exist?

Modern architectures (such as Gemma 2, Mistral, and hybrid sliding-window models) interleave two distinct types of attention layers:

1. **Global Attention Layers**: Every token attends to all previous tokens across the entire context window.
2. **Local (Sliding Window) Attention Layers**: Tokens only attend to neighbours within a local receptive field of size $W$ (e.g. $W = 4096$).

```text
Layer 0 (Local Window W=4k):   Max relative distance = 4096  ──> Needs high-resolution local theta
Layer 1 (Global Attention):    Max relative distance = 128k  ──> Needs large global theta to avoid phase wrapping
Layer 2 (Local Window W=4k):   Max relative distance = 4096  ──> Needs high-resolution local theta
```

Using a massive `rope_max_timescale` (e.g. $500{,}000$) inside a local 4k sliding window dilutes local positional discriminability. 

`local_rope_max_timescale` allows decoupling the timescale of local sliding-window attention layers from global attention layers.

---

## 2. What it actually controls

```yaml
local_rope_max_timescale: -1
```

- When `-1` (default): Local attention layers fall back to using `rope_max_timescale`. Both local and global layers share the same rotary base theta.
- When `> 0` (e.g. `10_000`): Local attention layers use `local_rope_max_timescale` as their rotary base theta, while global layers use `rope_max_timescale` (or `global_rope_max_timescale`).

```text
Attention Layer Type Check:
Is Layer Local (Sliding Window)?
      │
      ├── YES ──> Is local_rope_max_timescale > 0?
      │                │
      │                ├── YES ──> Use local_rope_max_timescale (e.g. 10_000)
      │                └── NO  ──> Use rope_max_timescale (Fallback)
      │
      └── NO (Global) ──> Use rope_max_timescale / global_rope_max_timescale
```

---

## 3. Options and Defaults

| Value | Behavior | Use Case |
|---|---|---|
| `-1` (default) | Local layers inherit `rope_max_timescale` | Standard homogeneous models (Llama 2/3, vanilla Transformer) |
| `> 0` (e.g. `10000`) | Explicitly overrides base theta for sliding-window layers | Gemma 2 / hybrid sliding-window architectures |

---

## 4. Interactions

- **`sliding_window_size`**: `local_rope_max_timescale` is only meaningful when sliding-window attention is active (`sliding_window_size > 0`).
- **`rope_max_timescale`**: Acts as the global timescale and default fallback.

---

## 5. Practical Scenarios

- **Homogeneous Models (Llama 3)**: Keep `-1`.
- **Hybrid Local/Global Models (Gemma 2)**: Set `rope_max_timescale: 1000000` and `local_rope_max_timescale: 10000` so local attention retains sharp short-range positional resolution.

---

### One-line intuition

> **`local_rope_max_timescale` overrides the RoPE base theta specifically for local sliding-window attention layers to preserve high positional resolution across short receptive fields.**
