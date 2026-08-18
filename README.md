# Maxtext Configuration Marathon

A structured, first-principles reference for MaxText configuration parameters — one deep-dive file per parameter.

## What this is

[MaxText](https://github.com/AI-Hypercomputer/maxtext) is Google's open-source JAX-based framework for training large language models on TPUs (and GPUs). It has a large flat config surface (`base.yml`) where every parameter has a non-obvious default and meaningful interactions with others.

This repo is a **personal knowledge base** that works through each parameter category by category — not just documenting what a param does, but *why* it exists, what its first-principles motivation is, what goes wrong if you set it wrong, and how it interacts with other params.

## Structure

Each category has a summary table file (the starting point) and one `.md` file per parameter (the deep dive):

| File | Contents |
|---|---|
| `1. General Run Setup + Checkpointing.md` | Run identity, checkpoint loading priority, all checkpointing params |
| `2. Logging-Misc + Precision & Quantization.md` | Metrics, dtype, quantization flags |
| `3. Core Model Architecture.md` | Embedding dims, heads, layers, attention, MLP |
| `4. Mixture of Experts (MoE).md` | MoE-specific routing and capacity params |
| `5. Pipeline Parallelism + Rematerialization.md` | Pipeline stages, microbatches, remat policies |
| `6. Attention Mechanisms.md` | Attention types, MLA, GQA, sliding window, KV cache |
| `7. Hardware + Sharding or Mesh.md` | Hardware selection, mesh axes, ICI/DCN parallelism, Splash Attention tuning |
| `8. Tokenizer + Dataset & Data Pipeline.md` | Tokenizer config, Grain/HF data loading, packing, shuffling |
| `9. Training Loop, LR Schedule, Optimizer & Profiling.md` | Steps, learning rate, Adam/Muon, gradient clipping, profiler |
| `10. RoPE, Eval-Decode, AOT Compilation & Monitoring.md` | RoPE variants, eval/decode params, ahead-of-time compile, GCP monitoring |
| `11. Inference Serving + KV Cache Layout.md` | Inference server, prefill/decode cache, prefix caching |
| `12. Multimodal (Vision + Audio).md` | ViT encoder, audio encoder, multimodal data params |
| `13. Qwen3-Next, mHC, Engram, Distillation, Elastic Training.md` | GDN linear attention, hyper connections, Engram n-gram memory, knowledge distillation, elastic training |

Individual parameter deep-dives live as flat `.md` files (e.g. `run_name.md`, `checkpoint_period.md`). Each covers:

- **Why it exists** — the problem it solves from first principles
- **What it controls** — mechanics, not just a description
- **Options** — all valid values with their effects
- **Interactions** — how it combines with related params
- **Practical guidance** — when to change it and what to watch out for
- **One-line intuition** — the sharpest possible summary

## Viewing

> **This vault is Obsidian-targeted.** Open the folder as an Obsidian vault for the best experience.

With Obsidian:
- `[[wiki-links]]` in the summary tables navigate directly to the deep-dive file for that parameter
- Graph view shows how parameters relate to each other
- The reading experience (rendered tables, code blocks, diagrams) matches the intended format

Without Obsidian, every file is plain Markdown and readable in any editor or renderer — but links won't be clickable.

## Scope

**Complete.** All 13 sections are fully documented — ~640 individual parameter deep-dives covering the entire `base.yml` surface, from core run setup through inference serving, multimodal, distillation, and elastic training.
