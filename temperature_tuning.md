## 1. Why does `temperature_tuning` exist?

In long-context transformers (sequences $> 32\text{k}$ up to $1\text{M}+$ tokens), attention logit distributions tend to flatten or develop severe entropy degradation as sequence length scales. Standard softmax dividing by a static $\sqrt{d_k}$ fails to compensate for the increasing sum of exponential terms across thousands of keys, causing attention dispersion, hallucinations, and retrieval failures in needle-in-a-haystack tasks.

Recent research (such as *Dynamic Temperature Tuning for Long-Context LLMs*, [arXiv:2501.19399](https://arxiv.org/abs/2501.19399)) demonstrates that dynamically modulating attention temperature as a function of sequence position restores sharp, stable attention distributions.

```text
Standard Static Attention:
Attention_Weights = Softmax( Q K^T / sqrt(d_k) )
Problem: At 128k context, massive denominator flattens attention sharpness.

Dynamic Temperature Tuning:
tau(s) = f(position_s, sequence_length)
Attention_Weights = Softmax( Q K^T / (sqrt(d_k) * tau(s)) )
Result: Attention entropy remains stable across 1k -> 128k tokens.
```

`temperature_tuning` activates dynamic, length-adaptive attention temperature scaling in MaxText.

---

## 2. Mathematical Mechanics

When `temperature_tuning: true`, the attention scaling factor is computed per query token dynamically:

```text
Token Position s (0 <= s < seq_len)
         │
         ▼
Compute Length Scale Factor:
  tau = 1.0 + alpha * log(s + 1) / log(max_target_length)
         │
         ▼
Scale Attention Logits:
  Logits(s, :) = (Q_s K^T) / (sqrt(d_k) * tau)
```

This dynamically adjusts the softness of the softmax distribution so that later tokens in ultra-long sequences retain precise recall over earlier context.

---

## 3. Options and Defaults

| Value | Behavior | Recommended Setting |
|---|---|---|
| `false` (Default) | Standard static scaling ($1 / \sqrt{d_k}$) | Standard pre-training and sequences $\le 32\text{k}$ |
| `true` | Dynamically scales attention temperature per sequence length | Long-context training and fine-tuning ($> 32\text{k}-1\text{M}$ tokens) |

---

## 4. Key Interactions

- **`max_target_length`**: Dictates the sequence length horizon over which temperature curves are calibrated.
- **`attn_logits_soft_cap`**: Complementary technique (Gemma-style) that clips extreme logits; can be combined with temperature tuning.
- **`attention`**: Compatible with FlashAttention and XLA fused attention kernels supporting custom logit scales.

---

## 5. When to Change It

- **Enable**: When pushing context windows beyond 32k tokens on Llama, Mistral, or custom architectures, especially if noticing attention degradation on long needle retrieval.
- **Disable**: For standard baseline runs or exact model reproduction of legacy weights.

---

### One-line intuition
> **`temperature_tuning` dynamically adjusts attention temperature based on token sequence length, preventing attention flattening and maintaining output stability in ultra-long context windows.**
