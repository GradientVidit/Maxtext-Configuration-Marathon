## 1. Why does `global_rope_max_timescale` exist?

In next-generation hybrid architectures (such as Gemma 4 and multi-scale attention networks), global attention layers may require specialized rotary timescales distinct from both the default model-wide `rope_max_timescale` and local sliding-window layers.

`global_rope_max_timescale` exists as an explicit timescale override specifically dedicated to global attention layers, enabling fine-grained, per-layer-type frequency tuning.

---

## 2. What it actually controls

```yaml
global_rope_max_timescale: -1
```

- When `-1` (default): Global attention layers use the general `rope_max_timescale`.
- When `> 0`: Global attention layers override `rope_max_timescale` with `global_rope_max_timescale`.

```text
Timescale Resolution Hierarchy for Global Attention:
Is global_rope_max_timescale > 0?
        │
        ├── YES ──> Use global_rope_max_timescale
        └── NO  ──> Fall back to rope_max_timescale
```

---

## 3. Options and Defaults

| Value | Behavior | Target Architecture |
|---|---|---|
| `-1` (default) | Falls back to `rope_max_timescale` for global attention | Standard Llama, Mistral, Gemma 1/2 |
| `> 0` | Dedicated base theta for global attention layers | Gemma 4 / experimental multi-timescale models |

---

## 4. Interactions with Related Timescales

```text
┌─────────────────────────────────────────────────────────────┐
│                    RoPE Timescale Routing                   │
│                                                             │
│ Layer Type: Local Attention      Layer Type: Global Attention│
│      │                                    │                 │
│      ▼                                    ▼                 │
│ local_rope_max_timescale > 0?    global_rope_max_timescale > 0?
│   ├── YES: local_rope_max_...      ├── YES: global_rope_max...
│   └── NO:  rope_max_timescale      └── NO:  rope_max_timescale
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Practical Scenarios

- **Standard Models**: Keep `-1`.
- **Gemma 4 / Multi-Timescale Pretraining**: Set `global_rope_max_timescale` to match the exact published configuration for global attention layers.

---

### One-line intuition

> **`global_rope_max_timescale` provides an explicit base theta override for global attention layers in hybrid multi-timescale Transformer architectures.**
