## 1. Why does `engram_vocab_bases` exist?

Engram stores explicit phrase representations in hash-indexed embedding tables.

Because the total space of all possible multi-token n-grams is astronomical ($V^n$, where $V \approx 100\text{k}$ vocabulary size means $10^{10}$ possible bigrams and $10^{15}$ trigrams), storing exact uncompressed tables is impossible.

Engram uses **modulo hash addressing** to compress this combinatorial space into fixed-size embedding tables:

$$\text{Index}_n = \text{Hash}(t_1, \dots, t_n) \pmod{B_n}$$

```text
3-gram Token Span: ["Machine", "Learning", "System"]
                         │
                         ▼  Polynomial Hash Function
                   Raw 64-bit Integer Hash
                         │
                         ▼  Modulo engram_vocab_bases[2] (e.g. 200,003)
                   Slot Index in Table 3 (0 ... 200,002)
                         │
                         ▼
                   Retrieve Embedding Vector
```

`engram_vocab_bases` specifies the list of table modulo capacities $[B_1, B_2, \dots, B_N]$ for each n-gram order from $1$ to `engram_max_ngram_size`.

---

## 2. Mechanics & Collision Minimization

- **Prime Moduli**: Hash collision rates are minimized when the table capacity $B_n$ is chosen as a large prime number (e.g. `[100003, 200003, 400009]`).
- **Capacity Scaling**: Higher n-gram orders typically use larger base sizes because higher-order phrase spaces are exponentially larger.
- **Empty List**: When `engram_vocab_bases: []` (default), MaxText computes automatic default table sizes proportional to `vocab_size`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `engram_vocab_bases` | `list[int]` | `[]` | List of integers specifying table sizes per n-gram order (e.g. `[131072, 262144, 524288]`), or `[]` for auto-sizing |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `engram_max_ngram_size` | Length of `engram_vocab_bases` should equal `engram_max_ngram_size`. |
| `engram_seed` | Seed used in the rolling hash function before applying the modulo operation. |
| `vocab_size` | Used as the fallback reference when `engram_vocab_bases` is empty. |

---

## 5. Practical Guidance & Best Practices

| Setting | Behavior | Use Case |
| :--- | :--- | :--- |
| `engram_vocab_bases: []` (Default) | Automatically sizes hash tables based on tokenizer vocabulary | Convenient default for fast prototyping. |
| Explicit Prime List (e.g. `[100003, 200003, 400009]`) | Deterministic, collision-resistant hash table sizing across TPUs | **Recommended for production pretraining** with Engram. |

---

### One-line intuition

> `engram_vocab_bases` defines the hash table modulo capacities for each n-gram order in the Engram memory module, controlling memory footprint and hash collision rates.
