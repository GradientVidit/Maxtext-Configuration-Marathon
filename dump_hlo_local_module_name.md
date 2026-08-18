## 1. Why does `dump_hlo_local_module_name` exist?

A full MaxText execution builds dozens of small helper modules (initialization, data formatting, RNG splitting, scalar reductions).

Writing every helper module to local disk creates thousands of files. Filtering by module substring at the compiler level ensures only the core training step is dumped:

```text
Compiler Module Filter:
  Module "init_state"        ── [Filtered Out]
  Module "jit_train_step"    ── [MATCHES dump_hlo_local_module_name] ──> Saved to disk
  Module "rng_split"         ── [Filtered Out]
```

`dump_hlo_local_module_name` provides a substring filter for saving modules locally.

---

## 2. Fundamentals & Mechanics

- Default `"jit_train_step"` isolates the main training graph.
- Empty string `""` removes the filter and dumps all compiled modules.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `"jit_train_step"` | Dumps only the primary training step module. |
| Dump All | `""` | Dumps every compiled XLA module without filtering. |

---

## 4. Interactions & Dependencies

- Passed to `--xla_dump_hlo_module_re` in `dump_hlo_xla_flags`.

---

## 5. Practical Scenarios & Failure Modes

- When debugging eval step compilation, change to `"jit_eval_step"`.

---

### One-line intuition

> **`dump_hlo_local_module_name` filters which compiled XLA modules are saved to local disk, defaulting to `"jit_train_step"`.**
