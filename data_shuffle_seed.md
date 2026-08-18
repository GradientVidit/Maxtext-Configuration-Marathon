## 1. Why does `data_shuffle_seed` exist?

Pseudo-random number generators (PRNGs) govern how dataset records and shard indices are permuted across workers in distributed training.

Using an explicit, configurable seed ensures that:
1. Different training runs can use different data orderings to study data curriculum effects.
2. An identical data sequence can be reproduced across repeated experiments for debugging.

```text
data_shuffle_seed = 0  ──> PRNG Stream 0 ──> Batch Sequence [A, X, M, B...]
data_shuffle_seed = 42 ──> PRNG Stream 42 ──> Batch Sequence [K, L, P, Q...]
```

`data_shuffle_seed` specifies the integer PRNG seed for dataset shuffling.

---

## 2. Fundamentals & Mechanics

- Passed directly to the input loader (Grain, tf.data, or Hugging Face datasets).
- Independent of weight initialization seeds.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `0` | Default seed for data permutation. |
| Custom Integer | `N` | Any integer to initialize a distinct data permutation sequence. |

---

## 4. Interactions & Dependencies

- Interacts with `enable_data_shuffling: true`.

---

## 5. Practical Scenarios & Failure Modes

- **Multi-Run Ensembles:** When training multiple seeds of a smaller model to measure variance, changing `data_shuffle_seed` and `init_weights_seed` ensures true statistical independence.

---

### One-line intuition

> **`data_shuffle_seed` sets the random seed governing input data shuffling order across training steps.**
