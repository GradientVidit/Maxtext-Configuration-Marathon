## 1. Why does `qk_nope_head_dim` exist?

In Multi-Head Latent Attention (MLA), Query and Key vectors are split into two decoupled components:
1. **NoPE (No Positional Encoding):** A position-invariant component that encodes purely semantic / content relationships.
2. **RoPE (Rotary Positional Encoding):** A decoupled positional component that carries token distance and ordering information.

$$\text{Full Query Head } Q = [Q_{\text{nope}}, Q_{\text{rope}}], \quad \text{Full Key Head } K = [K_{\text{nope}}, K_{\text{rope}}]$$

```text
MLA Attention Head Decomposition:

Query: ┌───────────────────────────┬──────────────────┐
       │   Q_nope (qk_nope_head_dim)│ Q_rope (qk_rope) │
       └───────────────────────────┴──────────────────┘
Key:   ┌───────────────────────────┬──────────────────┐
       │   K_nope (qk_nope_head_dim)│ K_rope (qk_rope) │
       └───────────────────────────┴──────────────────┘
                     │                         │
                     ▼                         ▼
         Content Dot-Product           Positional Dot-Product
       (Derived from c_kv latent)      (Derived via RoPE)
```

`qk_nope_head_dim` defines the dimensionality ($d_{nope}$) of the content-carrying, position-invariant portion of each Query and Key head.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `128` | Head dimension of $d_{nope}=128$. | **Default**. Exact DeepSeek-V2 / DeepSeek-V3 specification. |
| Any integer $> 0$ | Custom NoPE head dimension. | Total query/key head dimension will be $d_{nope} + d_{rope}$. |

Default in `base.yml`: `128`

---

## 3. Why NoPE is Necessary for KV Compression

Positional encodings (like RoPE) depend explicitly on token sequence index $t$. If positional encodings were injected into the compressed latent $c_{kv}$, the latent representation would change at every token position, destroying matrix associativity and preventing KV cache compression.

By isolating all semantic information into $K_{nope}$, MLA allows $K_{nope}$ to be reconstructed dynamically from the position-invariant latent $c_{kv} = X W_{DKV}$ using a fixed up-projection matrix $W_{UK}$.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[qk_rope_head_dim]] | Companion: defines the rotary positional portion ($d_{rope}$). Total head dimension is $\text{qk\_nope\_head\_dim} + \text{qk\_rope\_head\_dim}$. |
| [[kv_lora_rank]] | $K_{nope}$ is projected from the $c_{kv}$ latent vector of size `kv_lora_rank`. |
| [[v_head_dim]] | Value head dimension ($d_v$), configured independently of $d_{nope}$. |
| [[attention_type]] | Active when `attention_type: 'mla'`. |

---

## 5. Practical Scenarios

- **Pretraining DeepSeek Architecture:** Set `qk_nope_head_dim: 128` and `qk_rope_head_dim: 64`.
- **Total Attention Scale Factor:** The attention scaling denominator in MLA is $\sqrt{d_{nope} + d_{rope}} = \sqrt{128 + 64} = \sqrt{192} pprox 13.856$.

---

### One-line intuition

> **`qk_nope_head_dim` sets the dimension ($d_{nope}=128$) of MLA's position-invariant Query and Key components, allowing content addressing to be projected directly from the compressed KV latent.**
