## 1. Why does `prefill_cache_dir` exist?

To support saving and reloading prefill KV cache states across decoding runs, MaxText requires a dedicated file storage path (local filesystem or GCS bucket).

`prefill_cache_dir` defines the directory path where prefilled KV cache arrays are written to or read from.

---

## 2. What it actually controls

```yaml
prefill_cache_dir: ""
```

- If `load_from_prefill_dir: false` and `prefill_cache_dir != ""`: `decode.py` executes prefill and serializes the resulting KV cache tensors into `prefill_cache_dir`.
- If `load_from_prefill_dir: true` and `prefill_cache_dir != ""`: `decode.py` loads the saved KV cache tensors from `prefill_cache_dir`.

```text
Write Flow:
Prefill Pass ──> Compute KV Cache ──> Write to prefill_cache_dir

Read Flow:
prefill_cache_dir ──> Load KV Cache ──> Start Autoregressive Decode Loop
```

---

## 3. Options and Defaults

| Value | Behavior |
|---|---|
| `""` (default, empty) | Prefill caching disabled |
| `"/tmp/prefill_cache"` | Local directory path for single-host caching |
| `"gs://my-bucket/prefill_cache"` | Cloud Storage path for multi-host TPU cluster access |

---

## 4. Interactions

- **`load_from_prefill_dir`**: Determines whether `prefill_cache_dir` is used as a read source (`true`) or write destination (`false`).

---

## 5. Practical Scenarios

- **Saving Prefill State for Fast Restarts**:
```yaml
prefill_cache_dir: "gs://my-bucket/benchmarks/llama3_prefill_128k"
load_from_prefill_dir: false  # First run: generates and saves
```

---

### One-line intuition

> **`prefill_cache_dir` specifies the local or GCS storage directory used to persist or load precomputed KV cache tensors.**
