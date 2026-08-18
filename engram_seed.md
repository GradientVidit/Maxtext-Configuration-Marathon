## 1. Why does `engram_seed` exist?

Engram maps variable-length token n-grams into table indices using polynomial rolling hash functions:

$$\text{Hash}(t_1, \dots, t_n) = \sum_{i=1}^n t_i \cdot p_n^i \pmod{2^{64}}$$

```text
Tokens: [t_1, t_2, t_3]
             │
             ▼
Polynomial Hash with Multipliers derived from engram_seed
             │
             ▼
Deterministic Hash Address ──> Hash Table Index
```

The choice of polynomial multipliers and hashing offsets determines which n-gram token combinations collide into identical table slots.

`engram_seed` initializes the pseudo-random generation of these hashing multipliers and table offsets.

---

## 2. Mechanics & Multi-Host Determinism

- **Cluster-Wide Determinism**: In distributed TPU clusters across thousands of hosts, all hosts must use the identical `engram_seed` so that identical token sequences resolve to the exact same hash slot across all data/tensor parallel replicas.
- **Independence**: Changing `engram_seed` changes the hash distribution without changing model architecture or parameter shapes.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `engram_seed` | `int` | `0` | Any non-negative integer |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `engram_vocab_bases` | Hash outputs are modulated by `engram_vocab_bases` to obtain valid table indices. |
| `engram_max_ngram_size` | Unique multiplier sequences are generated for each order up to `engram_max_ngram_size`. |

---

## 5. Practical Guidance

| Scenario | Recommendation |
| :--- | :--- |
| **Standard Training / Resumption** | Keep fixed (default `0`) across checkpoints to preserve weight-to-ngram slot alignment. |
| **Ablation Studies on Hash Collisions** | Vary `engram_seed` across trial runs to measure sensitivity to hash collision patterns. |

---

### One-line intuition

> `engram_seed` sets the random seed for Engram's polynomial hash multipliers, ensuring deterministic, collision-consistent n-gram table addressing across distributed cluster nodes.
