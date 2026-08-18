## 1. Why does `compile_topology` exist?

Normally, JAX/XLA compiles computation graphs Just-In-Time (JIT) using the physical TPU devices attached to the current host. 

However, you often want to compile a model for a large TPU cluster (e.g. a 256-chip or 2048-chip pod) from a cheap single-CPU development machine without provisioning or paying for the expensive TPU cluster during the compilation phase.

JAX provides simulated hardware topologies that allow XLA to compile graphs as if it were targeting physical TPU chips:

```text
Compilation on Single CPU Machine:
compile_topology: 'v5e-256' ──> JAX Simulated Device Mesh (256 v5e chips) ──> XLA Graph Optimization
```

`compile_topology` specifies the target hardware accelerator topology name for offline Ahead-of-Time (AOT) compilation.

---

## 2. What it actually controls

```yaml
compile_topology: ''
```

- When empty `''` (default): MaxText compiles against physically attached hardware devices.
- When set (e.g. `'v5e-256'`, `'v4-128'`, `'v5p-512'`): MaxText initializes a simulated hardware mesh corresponding to the specified TPU generation and chip count.

```text
compile_topology Options:
'v4-128'   ──> TPU v4 with 128 chips (64 hosts)
'v5e-256'  ──> TPU v5e with 256 chips
'v5p-512'  ──> TPU v5p with 512 chips
'v6e-256'  ──> TPU v6e (Trillium) with 256 chips
```

---

## 3. Options and Common Values

| Value | Target Accelerator Pod | Total Chips |
|---|---|---|
| `''` (default) | Physical hardware present on host | Physical device count |
| `'v4-16'`, `'v4-128'`, `'v4-512'` | Cloud TPU v4 | 16, 128, 512 |
| `'v5e-64'`, `'v5e-256'` | Cloud TPU v5e (Lite) | 64, 256 |
| `'v5p-128'`, `'v5p-512'`, `'v5p-2048'` | Cloud TPU v5p (Performance) | 128, 512, 2048 |
| `'v6e-64'`, `'v6e-256'` | Cloud TPU v6e (Trillium) | 64, 256 |

---

## 4. Interactions and Requirements

- **`compile_topology_num_slices`**: For multi-slice topologies, `compile_topology_num_slices` must be specified alongside `compile_topology`.
- **`compiled_trainstep_file`**: Output file where the compiled artifact is saved.

---

## 5. Practical Scenarios

- **Validating Sharding & HBM Fits Before Allocation**: Run AOT compilation with `compile_topology: 'v5p-512'` to verify that your sharding rules and batch sizes do not run out of memory (OOM) before requesting TPU quota.

---

### One-line intuition

> **`compile_topology` defines the target TPU hardware architecture and chip count for Ahead-of-Time (AOT) offline compilation without physical hardware.**
