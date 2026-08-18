## 1. Why does it exist?

When Ahead-of-Time (AOT) compilation is executed via Google-internal topology descriptors (`internal_compile: true`), the compiler cannot infer the target slice size from open-source topology strings (like `'v5e-256'`). The explicit number of accelerator devices to compile the computation graph for must be provided explicitly.

```text
internal_compile: true
        ↓
Need explicit device count for XLA graph construction
        ↓
internal_compile_num_devices: 256  ──→ Compile graph for exactly 256 devices
```

`internal_compile_num_devices` provides the required device count for internal Google AOT compilation runs.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `-1` (default) | Unset. Valid when `internal_compile: false`. |
| Positive integer (e.g. `64`, `256`, `2048`) | The exact number of accelerator devices targeted during internal compilation. |

Default in `base.yml`:
```yaml
internal_compile_num_devices: -1
```

---

## 3. Validation Rules

MaxText enforces:
- If `internal_compile: true` $\implies$ `internal_compile_num_devices` must be $> 0$.
- If `internal_compile: false` $\implies$ `internal_compile_num_devices` remains `-1` (ignored).

---

### One-line intuition

> **`internal_compile_num_devices` supplies the explicit target device count when compiling models using Google-internal topology generators.**
