## 1. Why does it exist?

In modern large language models, vocabulary sizes have grown dramatically (e.g. Llama 3 has 128k tokens; Gemma has 256k tokens).

At the final output layer of the model, computing the Cross-Entropy loss requires projecting the hidden activations $[B \times S, D]$ against the embedding matrix $[D, V]$ to generate the unnormalized logit tensor $[B \times S, V]$.

For a batch of $B \times S = 32,768$ tokens with Gemma's $V = 256,000$ vocabulary in FP32/BF16:

$$\text{Logit Tensor Memory} = 32,768 \times 256,000 \times 4 \text{ bytes} \approx 33.5 \text{ GB}$$

Materializing this gigantic logit tensor all at once in accelerator HBM causes immediate Out-Of-Memory (OOM) errors during the loss computation.

```text
Without Vocab Tiling (num_vocab_tiling: 1):
  All 32k tokens projected to 256k vocab simultaneously
  ──→ 33.5 GB peak HBM spike ──→ OOM Crash!

With Vocab Tiling (num_vocab_tiling: 4):
  Chunk 1: 8k tokens projected to vocab ──→ Compute Partial Loss ──→ Free Memory
  Chunk 2: 8k tokens projected to vocab ──→ Compute Partial Loss ──→ Free Memory
  Chunk 3: 8k tokens projected to vocab ──→ Compute Partial Loss ──→ Free Memory
  Chunk 4: 8k tokens projected to vocab ──→ Compute Partial Loss ──→ Free Memory
  ──→ Peak HBM spike slashed to ~8.4 GB!
```

`num_vocab_tiling` is a critical memory-saving optimization that computes the cross-entropy loss in $N$ sequential chunks along the batch-sequence axis, dramatically slashing peak HBM usage.

---

## 2. Fundamentals & Mathematics

Instead of computing $\text{CrossEntropy}(\text{logits}, \text{targets})$ over the entire global batch at once, MaxText tiles the batch-sequence dimension:

$$\text{Chunk Size} = \frac{\text{Batch} \times \text{Sequence}}{\text{num\_vocab\_tiling}}$$

Each chunk computes its unnormalized logit slice, evaluates the log-sum-exp and cross-entropy loss contributions, accumulates the loss scalar, and discards the logit slice before computing the next chunk. The mathematical output and gradient are bit-exact.

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | Vocab tiling disabled; entire logit tensor materialized at once. |
| Integer $> 1$ (e.g. `2`, `4`, `8`, `16`) | Chunks cross-entropy loss computation into $N$ sequential slices along batch-sequence. |

Default in `base.yml`:
```yaml
num_vocab_tiling: 1
```

---

## 4. Practical Guidelines

- **Standard Models (Vocab $\le 32\text{k}$)**: `num_vocab_tiling: 1` is usually fine.
- **Large-Vocab Models (Gemma 256k, Llama 3 128k, Qwen 150k)**: Setting `num_vocab_tiling: 4` or `8` is **strongly recommended** to prevent loss-computation OOMs without affecting training throughput.

---

### One-line intuition

> **`num_vocab_tiling` breaks the output cross-entropy loss calculation into $N$ sequential chunks along the batch-sequence axis, eliminating massive HBM memory spikes in large-vocabulary models (like Gemma and Llama 3).**
