## 1. Why does `attention_sink` exist?

During long autoregressive generation or infinite-horizon streaming, maintaining an unbounded Key-Value (KV) cache becomes impossible due to HBM limits. The naive solution—a FIFO sliding window cache—fails catastrophically: as soon as the initial sequence tokens (token 0 / BOS) slide out of the cache window, **perplexity explodes to infinity**.

This failure occurs because softmax in multi-head attention requires attention weights to sum to 1.0. Even when a query has no strong semantic match with recent tokens, the model offloads excess softmax probability mass onto the **initial tokens** (the "attention sinks").

```text
Without Attention Sink (Standard Sliding Window Eviction):
Step 1000: [t900, t901, ..., t1000]  <-- Initial tokens evicted!
           Softmax mass cannot be dumped onto t0/BOS --> Perplexity explodes!

With Attention Sink (StreamingLLM Pattern):
Step 1000: [t0, t1, t2, t3] + [t904, t905, ..., t1000]
           └── Sink Tokens ──┘   └── Sliding Window ───┘
           Softmax anchor preserved --> Perplexity remains stable indefinitely.
```

The **attention sink** mechanism (Xiao et al., 2023 / StreamingLLM) permanently pins the initial tokens in the KV cache so that attention normalization remains stable regardless of sequence length.

---

## 2. Options & Defaults

| Value | Mechanism | Impact on Streaming / Long Context |
|---|---|---|
| `false` | Standard sliding window / full causal eviction. | Initial tokens get evicted once sequence length exceeds cache buffer; unstable for infinite streaming. **Default**. |
| `true` | Dedicated attention sink tokens are retained in the KV cache throughout generation. | Preserves softmax mass offloading anchor; stabilizes generation across millions of streaming tokens without recomputation. |

Default in `base.yml`: `false`

---

## 3. Mechanics: Softmax Anchoring

$$lpha_{i, j} = rac{\exp(q_i k_j^T / \sqrt{d})}{\sum_{m \in \text{Cache}} \exp(q_i k_m^T / \sqrt{d})}$$

When attention sinks are enabled:
1. The first $S$ tokens (typically 1 to 4 initial tokens, including `<BOS>`) are permanently locked into the KV cache tensor at positions $[0 \dots S-1]$.
2. The remaining cache positions $[S \dots C-1]$ operate as a cyclic sliding window over the most recent tokens.
3. Every new query computes attention over:

$$\text{Attendable Tokens} = \{\text{Sink Tokens } t_{0 \dots S-1}\} \cup \{\text{Recent Tokens } t_{i-W \dots i}\}$$

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[sliding_window_size]] | `attention_sink` is designed specifically to pair with local sliding window attention to eliminate perplexity blowup. |
| [[attention_type]] | Most effective with `attention_type: 'local_sliding'`. |
| [[attn_logits_soft_cap]] | Soft-capping attention logits provides an alternative mechanism for bounding softmax peakiness, but does not replace positional sinks during eviction. |

---

## 5. Practical Scenarios

- **Standard Pretraining with Fixed Sequence Length ($N \le 8\text{K}$):** Leave `attention_sink: false`.
- **Infinite Streaming Inference (Chatbots, Long-running Agents):** Set `attention_sink: true` alongside `sliding_window_size` to stream millions of tokens with bounded constant HBM footprint.
- **Pretraining Streaming-Native Models:** Pretraining models with an explicit attention sink token ensures the network naturally learns to dump residual attention mass into dedicated placeholder positions.

---

### One-line intuition

> **`attention_sink=true` permanently pins initial anchor tokens in the KV cache, providing a dedicated sink for residual softmax probability mass and preventing perplexity collapse during sliding-window generation.**
