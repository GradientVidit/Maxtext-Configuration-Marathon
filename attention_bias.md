## 1. Why does `attention_bias` exist?

Modern autoregressive transformer architectures (LLaMA, Gemma, PaLM, DeepSeek) generally omit bias terms in linear layers to improve training stability, reduce parameter overhead, and simplify kernel fusions:

$$\text{No Bias:}\quad Q = X W_q,\quad K = X W_k,\quad V = X W_v$$

However, earlier architectures (such as GPT-2, GPT-3, GPT-J, and Megatron-LM) explicitly include learnable bias vectors in Q, K, and V projections:

$$\text{With Bias:}\quad Q = X W_q + b_q,\quad K = X W_k + b_k,\quad V = X W_v + b_v$$

`attention_bias` allows MaxText to toggle these projection biases on or off.

```text
Without Bias (attention_bias=false):
  X [B, S, D] ──> Matmul(W_q) ──> Q [B, S, H, D_head]

With Bias (attention_bias=true):
  X [B, S, D] ──> Matmul(W_q) + b_q ──> Q [B, S, H, D_head]
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `false` | Linear projections without additive bias terms ($Q = XW$). | **Default**. Standard for modern LLMs (LLaMA 1/2/3, Gemma 1/2, Mistral). |
| `true` | Linear projections include learnable additive bias vectors ($Q = XW + b$). | Required for exact weight conversion and reproduction of GPT-2/GPT-3 style models. |

Default in `base.yml`: `false`

---

## 3. Parameter and Memory Impact

When `attention_bias=true`, each attention layer instantiates three additional 1D bias parameters:
- $b_q \in \mathbb{R}^{N_{q\_heads} 	imes d_{head}}$
- $b_k \in \mathbb{R}^{N_{kv\_heads} 	imes d_{head}}$
- $b_v \in \mathbb{R}^{N_{kv\_heads} 	imes d_{head}}$

For a 70B model with $d_{model}=8192$ and 80 layers, the bias parameters account for $<0.01\%$ of total model weight, but their presence alters tensor shapes and checkpoint naming schemes.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[mlp_bias]] | Companion parameter for MLP feed-forward projections (`mlp_bias: true` adds bias to gate/up/down projections). |
| [[fused_qkv]] | When `fused_qkv: true` and `attention_bias: true`, biases are concatenated into a single $[b_q, b_k, b_v]$ vector. |
| [[source_checkpoint_layout]] | Must match the original model architecture when converting HuggingFace / Megatron checkpoints. |

---

## 5. Practical Scenarios

- **Fine-tuning or pretraining modern models (LLaMA 3, Gemma 2, DeepSeek):** Keep `attention_bias: false`.
- **Loading legacy GPT-2 / GPT-NeoX / StarCoder checkpoints:** Set `attention_bias: true` to prevent shape mismatch errors during weight loading.
- **Ablation Studies:** Test whether affine projection flexibility improves convergence in small sub-billion parameter models.

---

### One-line intuition

> **`attention_bias` toggles learnable additive bias vectors $b_q, b_k, b_v$ on attention projections, necessary for bit-exact reproduction of GPT-style legacy models.**
