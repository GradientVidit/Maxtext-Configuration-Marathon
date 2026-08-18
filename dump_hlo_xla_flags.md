## 1. Why does `dump_hlo_xla_flags` exist?

XLA's dumping behavior is controlled by low-level compiler flags passed via the `XLA_FLAGS` environment variable.

Rather than forcing users to manually construct complex shell strings, MaxText provides `dump_hlo_xla_flags` with a sensible default while allowing advanced compiler engineers to supply custom flags:

```text
Flag Construction:
dump_hlo_xla_flags: "" (Default)
        │
        ▼
Auto-generates:
"--xla_dump_to={local_dir} --xla_dump_hlo_module_re={local_module_name} --xla_dump_large_constants"
```

`dump_hlo_xla_flags` allows passing custom raw XLA compiler dumping flags.

---

## 2. Fundamentals & Mechanics

- **Default `""` (Empty):** MaxText automatically formats standard dump flags with large constants, output paths, and regex filters.
- When set non-empty, overrides the automatic flag string.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `""` | Uses standard auto-formatted `--xla_dump_*` flags. |
| Custom Flags | String | Direct override for custom XLA compiler flags. |

---

## 4. Interactions & Dependencies

- Formatted using `dump_hlo_local_dir` and `dump_hlo_local_module_name`.

---

## 5. Practical Scenarios & Failure Modes

- Adding `--xla_dump_hlo_as_html` via custom flags generates interactive HTML graph visualizations.

---

### One-line intuition

> **`dump_hlo_xla_flags` specifies raw XLA compiler dump flags, automatically populating standard directory and module filters when empty.**
