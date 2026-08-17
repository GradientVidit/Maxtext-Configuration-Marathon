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

Currently covers **Section 1: General Run Setup + Checkpointing** — all 29 parameters fully documented.

Remaining sections (Logging, Precision/Quantization, Architecture, MoE) are in progress.
