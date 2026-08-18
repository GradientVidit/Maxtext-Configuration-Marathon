## 1. Why does it exist?

When performing Ahead-of-Time (AOT) compilation in open-source MaxText, target hardware topologies are looked up via standard public string names (e.g. `compile_topology: 'v5e-256'`), which map to predefined physical chip dimension bounds in JAX.

Inside Google's internal development infrastructure, engineers compile directly against internal hardware topology descriptors and simulator engines (`get_topology_desc`) rather than open-source string mappings.

```text
Standard Open-Source Compilation:
  compile_topology: "v5e-256" ──→ Lookup OSS topology dictionary ──→ XLA Mesh

Google-Internal Compilation (internal_compile: true):
  internal_compile: true ──→ Bypass OSS mappings ──→ Calls internal `get_topology_desc`
```

`internal_compile` is an internal Google flag that bypasses open-source topology string resolution and invokes Google-internal topology generators directly.

---

## 2. Fundamentals & Mechanics

When `internal_compile: true`:
1. Open-source topology translation tables are bypassed.
2. MaxText directly requests a hardware topology descriptor using internal Google TPU cluster descriptions.
3. Requires specifying the explicit device target count via `internal_compile_num_devices`.

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `false` (default) | Standard open-source AOT compilation path using `compile_topology`. |
| `true` | Google-internal compilation path. |

Default in `base.yml`:
```yaml
internal_compile: false
```

---

## 4. Interactions with Related Parameters

```text
internal_compile: true
  └── internal_compile_num_devices: <int> (Must be specified, cannot remain -1)
```

- **`internal_compile_num_devices`**: Mandatory when `internal_compile: true`.
- **`compile_topology`**: Ignored or overridden when internal compilation is active.

---

## 5. Practical Usage Note

- **Open-Source Users**: Should **always** leave `internal_compile: false`. Enabling this in public cloud environments without internal Google development packages will raise runtime attribute errors.

---

### One-line intuition

> **`internal_compile` is a Google-internal flag for compiling models directly against proprietary TPU topology generators, bypassing public open-source topology tables.**
