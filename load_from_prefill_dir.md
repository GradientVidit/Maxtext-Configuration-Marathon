## 1. Why does `load_from_prefill_dir` exist?

In large language model serving and benchmark workflows, the prefill phase (processing large prompts of $8k$–$128k$ tokens) consumes significant compute and time:

```text
Standard Decode Run:
[ Read Prompt ] ──(Compute Heavy Prefill)──> [ Generate KV Cache ] ──(Decode Loop)──> [ Tokens ]

Cached Decode Run (load_from_prefill_dir: true):
[ Load Pre-computed KV Cache from Disk ] ───────────────────────────(Decode Loop)──> [ Tokens ]
```

When repeatedly benchmarking token generation latency, sampling algorithms, or decoding throughput across many experimental configurations on identical long prompts, recalculating the prefill KV cache every run is wasteful.

`load_from_prefill_dir` allows `decode.py` to bypass prefill compute entirely and load a previously saved KV cache from storage.

---

## 2. What it actually controls

```yaml
load_from_prefill_dir: false
```

- When `false` (default): `decode.py` tokenizes `prompt`, executes the prefill forward pass, populates the KV cache, and optionally saves it if `prefill_cache_dir` is configured.
- When `true`: `decode.py` skips the prefill forward pass and directly restores the KV cache state from `prefill_cache_dir`.

---

## 3. Options and Defaults

| Value | Prefill Phase Behavior | Use Case |
|---|---|---|
| `false` (default) | Executes full prefill forward pass | Normal generation, dynamic prompts |
| `true` | Reads precomputed KV cache from `prefill_cache_dir` | Decode benchmarking, fixed-prompt evaluation |

---

## 4. Interactions and Dependencies

- **`prefill_cache_dir`**: Must be set to a valid directory path containing saved prefill state when `load_from_prefill_dir: true`.
- **Model / Mesh Consistency**: The loaded prefill cache must match the current model architecture, head dimension, and sharding mesh.

---

## 5. Practical Scenarios

- **Benchmarking Generation Throughput**: First run with `load_from_prefill_dir: false` and `prefill_cache_dir: "gs://bucket/cache"` to save the KV cache. Then run multiple decoding experiments with `load_from_prefill_dir: true` to measure pure autoregressive step latency.

---

### One-line intuition

> **`load_from_prefill_dir` enables `decode.py` to bypass prompt prefill computation by loading pre-calculated KV cache tensors directly from disk.**
