## 1. Why does `engram_max_ngram_size` exist?

Human language is rich in hierarchical n-gram patterns: individual words (unigrams), compound phrases (bigrams: "machine learning"), and idiomatic expressions (trigrams: "as well as").

To give a neural network non-parametric access to multi-token phrase memories, Engram extracts rolling token spans of varying orders:

```text
Input Tokens: [ "Deep", "Mind", "Alpha", "Zero" ]
               ───┬───  ───┬───  ───┬───  ───┬───
                  │        │        │        │
Order 1 (Unigrams): [Deep], [Mind], [Alpha], [Zero]
Order 2 (Bigrams):  [Deep Mind], [Mind Alpha], [Alpha Zero]
Order 3 (Trigrams): [Deep Mind Alpha], [Mind Alpha Zero] (if engram_max_ngram_size = 3)
```

`engram_max_ngram_size` sets the maximum n-gram window order ($N$) extracted and hashed into the Engram memory tables.

A value of $N=3$ means Engram simultaneously looks up unigram, bigram, and trigram hash entries.

---

## 2. Mechanics & Hash Lookup Tables

For an input sequence of tokens $[t_1, t_2, \dots, t_S]$:

```text
Sequence of Token IDs
          │
    ┌─────┴──────────────────────────────┐
    ▼                                    ▼
1-gram Hashes: Hash(t_i)             2-gram Hashes: Hash(t_{i-1}, t_i) ... N-gram Hashes
    │                                    │
    ▼                                    ▼
Table 1 Lookup: E_1                  Table 2 Lookup: E_2 ... Table N Lookup: E_N
    │                                    │
    └─────────────────┬──────────────────┘
                      ▼
     Concatenate / Merge Multi-Order Embeddings
                      │
                      ▼
            Engram Conv & Head Fusion
```

- Each order $n \in [1, N]$ has its own dedicated hash embedding table size defined by `engram_vocab_bases`.
- Multi-order embeddings are retrieved concurrently, capturing both fine-grained lexical tokens and multi-word semantic units.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `engram_max_ngram_size` | `int` | `3` | Positive integer (typically `2`, `3`, or `4`) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `engram_vocab_bases` | Length of `engram_vocab_bases` should match `engram_max_ngram_size` (providing a table size for each n-gram order). |
| `engram_layers` | Active only when `engram_layers` is non-empty. |
| `engram_seed` | Supplies hash multipliers used across the $N$ n-gram levels. |

---

## 5. Practical Guidance & Trade-offs

| Value | Coverage | Memory & Table Size | Use Case |
| :--- | :--- | :--- | :--- |
| `engram_max_ngram_size: 3` (Default) | Unigram + Bigram + Trigram | Balanced parameter allocation and broad phrase coverage | Standard Engram configuration. |
| `engram_max_ngram_size: 2` | Unigram + Bigram | Lower hash table memory footprint | Lightweight memory-constrained models. |
| `engram_max_ngram_size: 4` | Up to 4-grams (long idioms) | Higher table memory; requires larger hash modulo base to avoid hash collisions | Large-scale knowledge-intensive models. |

---

### One-line intuition

> `engram_max_ngram_size` sets the maximum n-gram length (e.g. trigrams for $N=3$) hashed and looked up in the Engram non-parametric memory subsystem.
