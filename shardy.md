## 1. Why does it exist?

In distributed JAX, automatic sharding propagation has historically been handled by **GSPMD** (General SPMD Partitioning for XLA). While GSPMD pioneered automated SPMD partitioning, it suffered from complex C++ heuristics, slow compilation times at massive scale, and opaque error messages when sharding propagation failed.

Google developed **Shardy** as the next-generation, MLIR-based sharding propagation engine for JAX and OpenXLA. Shardy provides:
1. Faster, more predictable sharding propagation during XLA compilation.
2. Direct integration with MLIR dialect representations (`sdy`).
3. Clear diagnostics and deterministic sharding resolution across complex multi-dimensional meshes.

```text
Legacy Compiler Path (shardy: false):
  JAX Graph ──→ GSPMD (Legacy C++ Sharding Engine, slated for deprecation) ──→ XLA HLO

Modern Compiler Path (shardy: true):
  JAX Graph ──→ Shardy (Next-Gen MLIR Sharding Engine) ──→ OpenXLA
```

`shardy` controls whether MaxText uses the modern Shardy sharding-propagation backend instead of legacy GSPMD.

---

## 2. Fundamentals & JAX Roadmap

- **Default in JAX**: Starting in JAX 0.7.0+, Shardy is the default sharding propagation backend.
- **Deprecation Timeline**: Legacy GSPMD is slated for complete deprecation across the XLA ecosystem around 2026.

---

## 3. Options & Configuration

| Value | Backend Engine | Status |
|---|---|---|
| `true` (default) | **Shardy (MLIR)** | Recommended standard backend; faster compilation and modern sharding semantics. |
| `false` | **GSPMD** | Legacy backend; only for backwards-compatibility or regression debugging. |

Default in `base.yml`:
```yaml
shardy: true # Whether to use shardy XLA backend (default in Jax starting 0.7.0), or GSPMD (to be fully deprecated ~2026)
```

---

## 4. Interactions with Related Parameters

- **`shard_mode: "auto"`**: Shardy powers the automated propagation pipeline when `shard_mode` is set to `"auto"`.
- **`jax_cache_dir`**: Toggling between Shardy and GSPMD changes the compilation hash and triggers a fresh graph compile.

---

### One-line intuition

> **`shardy` selects the modern MLIR-based Shardy sharding propagation compiler backend (default `true`) over legacy GSPMD for faster, more robust distributed graph compilation.**
