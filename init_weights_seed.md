## 1. Why does `init_weights_seed` exist?

Neural network weights (dense kernels, attention projections, embeddings) are initialized using pseudo-random distributions (e.g. truncated normal, LeCun normal).

Without an explicit seed parameter, weight initialization would be non-deterministic, making it impossible to isolate whether loss differences between two runs stem from code changes or lucky initializations:

```text
Weight Initialization:
init_weights_seed: 0 ──> JAX PRNG Key(0) ──> Initial Weights W_0 (Exact bitwise match across runs)
```

`init_weights_seed` defines the root random seed used to initialize all trainable model parameters.

---

## 2. Fundamentals & Mechanics

- Split into sub-keys for every layer in the model PyTree.
- When loading a pretrained checkpoint via `load_parameters_path`, loaded weights overwrite this initialization.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `0` | Default weight initialization PRNG seed. |
| Custom Integer | `N` | Explicit seed for randomized parameter generation. |

---

## 4. Interactions & Dependencies

```text
init_weights_seed
        │
        ├─ load_parameters_path = ""  ──> Generates initial random weights
        └─ load_parameters_path != "" ──> Overwritten by loaded checkpoint weights
```

---

## 5. Practical Scenarios & Failure Modes

- **A/B Testing:** When benchmarking optimizer variations, keeping `init_weights_seed` identical ensures both configurations start from the exact same initial parameter point.

---

### One-line intuition

> **`init_weights_seed` provides the master PRNG seed for initial random parameter generation across all model layers.**
