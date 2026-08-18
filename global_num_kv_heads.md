
## 1. Why does `global_num_kv_heads` exist?

Gemma4 uses hybrid local/global attention where the global attention layers can have a **different number of KV heads** than the local layers. This is the counterpart to `global_head_dim`:

```text
Local attention layers:  base_num_kv_heads + head_dim
Global attention layers: global_num_kv_heads + global_head_dim
```

The separation exists because global layers attend over the full context (long range), and may benefit from a different GQA configuration than local layers (which attend only within a sliding window and thus have smaller KV cache needs per layer).

---

## 2. Default

```yaml
global_num_kv_heads: 0
```

`0` = unused. Same as `global_head_dim` — only active for hybrid local/global architectures like Gemma4.

---

## 3. How it mirrors `base_num_kv_heads`

The same GQA/MQA logic applies to global layers:

```text
global_num_kv_heads == base_num_query_heads   → MHA for global layers
global_num_kv_heads < base_num_query_heads    → GQA for global layers
global_num_kv_heads == 1                       → MQA for global layers
```

Each is configured independently from the local layer's KV head count.

---

## 4. Memory impact

Global attention's KV cache is already larger due to full-context attention (every token generates a KV entry). `global_num_kv_heads` further multiplies this:

```text
KV bytes per token (global layer) = 2 × global_num_kv_heads × global_head_dim × dtype_bytes

If global_num_kv_heads=4, global_head_dim=256, bf16:
  = 2 × 4 × 256 × 2 = 4096 bytes per global-layer KV entry
```

At 128k token context, this is 512 MB per global layer — significant.

---

## 5. Interaction summary

| Parameter | Applies to |
|---|---|
| `base_num_kv_heads` | Local (sliding-window) attention layers |
| `global_num_kv_heads` | Global (full-context) attention layers |
| `head_dim` | Local layer head dimension |
| `global_head_dim` | Global layer head dimension |

All four are only meaningful when `decoder_block` uses a hybrid attention architecture.

---

## 6. Safe to ignore

If you're not using Gemma4 or an equivalent hybrid model, leave this at `0`. It literally has no code path that executes when `decoder_block` is anything other than a hybrid architecture.

---

### One-line intuition

> **`global_num_kv_heads` sets the KV head count for Gemma4's full-context global attention layers — the GQA equivalent of `base_num_kv_heads`, but only for those specific layers.**
