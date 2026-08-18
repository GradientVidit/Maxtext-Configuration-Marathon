
## 1. The two loops in pipelined MaxText

```text
outer loop: pipeline iterations (one per microbatch, passes through all stages)
    inner loop: pipeline repeats (how many times stages cycle for a single microbatch)
```

`scan_pipeline_repeats` controls whether `jax.lax.scan` is applied to the **repeats** dimension — how many times the full stage cycle repeats per microbatch pass.

---

## 2. What "repeats" are

Each pipeline "repeat" is one full traversal through all stages handling a set of layers. If `num_pipeline_repeats=4`, a single microbatch passes through all stages 4 times to cover all decoder layers.

```text
Microbatch K:
  Repeat 0: stages handle layers  0–15
  Repeat 1: stages handle layers 16–31
  Repeat 2: stages handle layers 32–47
  Repeat 3: stages handle layers 48–63
```

Scanning over these repeats vs. unrolling them is the `scan_pipeline_repeats` choice.

---

## 3. Default and why it's false

```yaml
scan_pipeline_repeats: false  # default
```

The outer scan (`scan_pipeline_iterations`) already provides scan over microbatches. The repeats dimension is smaller (typically 1–4) and may not benefit as much from scanning — while unrolled repeats give XLA more freedom to optimize across repeats independently.

More importantly: both scan dimensions at once + remat policy causes over-rematerialization. The current recommendation is to scan and remat only one loop (outer iterations), not both.

---

## 4. When to set true

- Very large `num_pipeline_repeats` (e.g. 8+) where unrolled HLO becomes too large
- Compilation time is a bottleneck and repeat-level unrolling is the culprit
- Memory is tight and scan-level remat policy on repeats would help

Note: `set_remat_policy_on_pipeline_iterations` controls remat on the outer scan. There is no separate `set_remat_policy_on_pipeline_repeats` — if you scan repeats, you'd use the layers_per_stage remat setting or accept that the policy isn't applied there.

---

## 5. Options

| Value | Behavior |
|---|---|
| `false` | Default — unroll the repeats dimension |
| `true` | `jax.lax.scan` over pipeline repeats |

Default: `false`.

---

## 6. The full scan/remat matrix

| Setting | Default | Effect |
|---|---|---|
| `scan_pipeline_iterations` | `true` | Scan outer microbatch loop |
| `scan_pipeline_repeats` | `false` | Scan repeat loop (off by default) |
| `scan_layers_per_stage` | `false` | Scan inner layers loop (off by default) |
| `set_remat_policy_on_pipeline_iterations` | `true` | Remat on outer scan |
| `set_remat_policy_on_layers_per_stage` | `false` | Remat on inner scan (off) |

Only one scan+remat combination is active by default: outer iterations.

---

### One-line intuition

> **`scan_pipeline_repeats` applies `jax.lax.scan` to the repeats dimension of pipeline parallelism — off by default because the outer microbatch scan already handles compilation efficiency, and scanning both loops simultaneously causes unintended over-rematerialization.**
