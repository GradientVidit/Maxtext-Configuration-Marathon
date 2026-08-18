## 1. Why it exists: generating multiple candidate completions per prompt

Many LLM serving workloads require generating $N$ independent completions for a single input prompt:

```text
Single Sampling (multi_sampling: false, N=1):
[Prompt] ──> [Prefill: Generate KV Cache once] ──> [Decode: 1 Token Stream] ──> Completion 1

Multi-Sampling (multi_sampling: true, N > 1):
                                                  ┌──> [Decode Stream 1] ──> Completion 1
                                                  │
[Prompt] ──> [Prefill: Generate KV Cache once] ───┼──> [Decode Stream 2] ──> Completion 2
                                                  │
                                                  └──> [Decode Stream N] ──> Completion N
```

Examples include:
- **Best-of-$N$ Re-ranking / Majority Voting**: Generating 8 code solutions or math derivations and selecting the best one via a verifier.
- **Speculative decoding & tree search**: Branching candidate tokens.
- **Agentic reasoning (Monte Carlo Tree Search)**: Sampling diverse candidate paths.

Without server-level multi-sampling support, a client requesting 8 completions must send 8 duplicate requests, forcing the inference engine to run redundant **prefill computations** 8 times over the exact same prompt tokens, wasting enormous compute and quadrupling Time-To-First-Token.

`multi_sampling` enables the inference server to perform prompt prefill **once**, duplicate or broadcast the resulting KV cache across $N$ generation slots, and run simultaneous stochastic sampling across all branches.

---

## 2. Mechanics: KV cache replication & branched decoding

When `multi_sampling: true`:

```text
 1. Prompt Ingestion (Prefill):
    - Compute prompt activations for shape [1, L_prompt]
    - Generate KV Cache: [1, L_prompt, Heads, D_kv]
                       │
                       ▼
 2. Cache Expansion:
    - Broadcast/replicate KV cache along batch axis to shape [N_samples, L_prompt, Heads, D_kv]
                       │
                       ▼
 3. Stochastic Autoregressive Generation:
    - Sample N distinct tokens using independent random seeds:
      Token_1 ~ P(x | prompt, T > 0, seed_1)
      Token_2 ~ P(x | prompt, T > 0, seed_2)
      ...
      Token_N ~ P(x | prompt, T > 0, seed_N)
                       │
                       ▼
 4. Emit N parallel token streams to client
```

By branching after prefill, the expensive $O(L^2)$ matrix multiplications of prompt attention occur only once, drastically reducing per-sample latency and amortizing prefill cost.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
multi_sampling: false
```

| Value | Mode | Prefill Execution | Memory & Compute Profile | Use Case |
|---|---|---|---|---|
| `false` (default) | Single sequence per request ($N=1$) | 1 prefill per request | Minimal KV cache memory; 1 decode slot per query. | Standard conversational chat, single-response APIs. |
| `true` | Multi-candidate generation ($N > 1$) | 1 shared prefill per $N$ branches | Replicates KV cache across $N$ slots; $N$ parallel decode steps. | Best-of-$N$ code generation, self-consistency reasoning, RLHF/DPO rollout generation. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                      multi_sampling                       │
└─────────────┬───────────────────────────────┬─────────────┘
              │ (when true)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Requires stochastic sampling parameters:                  │
│ - decode_sampling_strategy: 'temperature' or 'nucleus'    │
│ - decode_sampling_temperature: > 0.0                      │
│ - decode_sampling_top_k / decode_sampling_nucleus_p       │
└───────────────────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────┐
│ Memory Impact:                                            │
│ KV Cache footprint on `generate_slice` scales by N.       │
└───────────────────────────────────────────────────────────┘
```

- **`decode_sampling_temperature`**: If `multi_sampling: true` is used with greedy decoding (`temperature: 0.0`), all $N$ branches will produce identical output tokens. Temperature or top-p/top-k sampling must be active to generate diverse samples.
- **KV Cache Memory Footprint**: In the decode stage, storing $N$ active generation trajectories requires $N \times$ the decode cache slots in TPU HBM.
- **`return_log_prob`**: When returning multiple candidate sequences, clients frequently request per-token log probabilities to score and rank the candidates.

---

## 5. Practical Scenarios & Failure Modes

### Best-of-8 Code Generation
For automated unit test generation where an agent needs 8 candidate implementations:
```yaml
inference_server: "MaxtextInterleavedServer"
multi_sampling: true
decode_sampling_strategy: "nucleus"
decode_sampling_temperature: 0.8
decode_sampling_nucleus_p: 0.95
```
MaxText processes the code context prompt once and decodes 8 diverse candidate completions concurrently.

### What breaks if misconfigured:
- **HBM OOM from high $N$**: If a high sample count is requested alongside large batch concurrency, the expanded KV cache buffer will exceed available accelerator memory.
- **Zero temperature generation**: Enabling `multi_sampling` while leaving greedy decoding active produces $N$ duplicate responses, wasting compute without generating diversity.

---

### One-line intuition

> **`multi_sampling` enables generating multiple independent token completions from a single prompt prefill, eliminating redundant prompt computation in Best-of-$N$ and self-consistency reasoning workflows.**
