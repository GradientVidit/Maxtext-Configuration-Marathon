## 1. Why does `use_sliced_mla_proj` exist?

In Multi-Head Latent Attention (MLA), the latent KV vector $c_{kv} \in \mathbb{R}^{d_c}$ must be projected into both non-positional Keys ($K_{nope} \in \mathbb{R}^{N_h \cdot d_{nope}}$) and Values ($V \in \mathbb{R}^{N_h \cdot d_v}$).

There are two mathematically equivalent computational paths to execute this up-projection:

```text
Path A (use_sliced_mla_proj=False, Default):
  1. Full Joint Matmul:  c_kv · W_UKV  ──> Produces [K_nope, V] concatenated in one tensor
  2. Split Tensor:       K_nope, V = jnp.split(output, [d_nope_total], axis=-1)

Path B (use_sliced_mla_proj=True):
  1. Slice Kernel Weights: W_UK, W_UV = jnp.split(W_UKV, [d_nope_total], axis=-1)
  2. Two Separate Matmuls: K_nope = c_kv · W_UK,  V = c_kv · W_UV
```

`use_sliced_mla_proj` is an **XLA compiler optimization knob**: depending on TPU/GPU memory bandwidth, slicing weights prior to contraction can reduce peak intermediate tensor allocations and improve XLA GEMM fusion.

---

## 2. Options & Defaults

| Value | Execution Strategy | Compiler / Kernel Impact |
|---|---|---|
| `false` | Single joint GEMM followed by tensor splitting. | **Default**. Standard contraction. |
| `true` | Slices projection kernel weights before contraction into two distinct GEMMs. | Reduces intermediate HBM buffer pressure in specific XLA layouts; only supported when `quantization=""`. |

Default in `base.yml`: `false`

---

## 3. Important Quantization Constraint

`use_sliced_mla_proj=true` is **incompatible with weight/activation quantization** (e.g. AQT / Qwix FP8 or INT8). Quantization kernels require static contiguous weight matrices. Slicing unquantized weights on the fly disrupts scale-factor calibration and quantized GEMM dispatching.

If `quantization` is set, `use_sliced_mla_proj` **must be `false`**.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[kv_lora_rank]] | Width of the input latent $c_{kv}$ being projected. |
| [[qk_nope_head_dim]], [[v_head_dim]] | Determine the slice boundary ($N_h \cdot d_{nope}$ vs $N_h \cdot d_v$). |
| [[quantization]] | Must be unquantized (`quantization: ""`) if `use_sliced_mla_proj: true`. |
| [[attention_type]] | Active when `attention_type: 'mla'`. |

---

## 5. Practical Scenarios

- **Standard Pretraining & Quantized Training:** Leave `use_sliced_mla_proj: false` (default).
- **TPU Memory Optimization during Prefill:** Test `use_sliced_mla_proj: true` when running extremely long prefill sequences without quantization to evaluate if XLA achieves better SRAM tiling.

---

### One-line intuition

> **`use_sliced_mla_proj=true` splits the MLA up-projection weight matrix before contraction rather than splitting activations after, optimizing XLA kernel scheduling for unquantized models.**
