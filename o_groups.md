## 1. Why does `o_groups` exist?

In standard attention output projections, every attention head interacts with every feature dimension of the output hidden state. 

In **Compressed Attention**, the output projection can be partitioned into $G$ independent **Group Linear Layers** (similar to grouped convolutions):

$$G = \text{o\_groups}$$

Instead of a single monolithic matmul, the $N_h$ attention heads and output channels $d_{model}$ are split into $G$ separate groups:

$$\text{Output}_g = \text{Concat}(\text{Heads}_g) \cdot W_{o, g} \quad \text{for } g \in [1, G]$$

$$\text{Where:}\quad W_{o, g} \in \mathbb{R}^{(N_h / G \cdot d_h) 	imes (d_{model} / G)}$$

```text
Standard Output Projection (o_groups = 0 or 1):
  All Heads [1..N_h] ─────────────> Single Giant W_o ─────────────> Output [d_model]

Grouped Output Projection (o_groups = 4):
  Heads [1..N_h/4]   ──> W_o,0 ──> Output Slice 0 [d_model/4] ──┐
  Heads [N_h/4..N_h/2] ──> W_o,1 ──> Output Slice 1 [d_model/4] ──┼──> Concat ──> Out [d_model]
  Heads [N_h/2..3N_h/4]──> W_o,2 ──> Output Slice 2 [d_model/4] ──┤
  Heads [3N_h/4..N_h]  ──> W_o,3 ──> Output Slice 3 [d_model/4] ──┘
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `0` | Standard non-grouped output projection ($G=1$). | **Default**. |
| Any integer $> 0$ (e.g. `2`, `4`, `8`) | Divides the output projection into $G$ independent groups. | $G$ must evenly divide both `base_num_query_heads` and `base_emb_dim`. |

Default in `base.yml`: `0`

---

## 3. Parameter Scaling under Grouping

$$\text{Total Parameters with Grouping} = G 	imes \left(rac{N_h \cdot d_h}{G} 	imes rac{d_{model}}{G}
ight) = rac{(N_h \cdot d_h) 	imes d_{model}}{G}$$

Setting `o_groups: 4` reduces output projection parameter count and FLOPs by **$4	imes$** ($75\%$ reduction).

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[o_lora_rank]] | Can be combined with LoRA decomposition to apply low-rank compression within each individual group. |
| [[compress_ratios]] | Per-layer compression ratios operating alongside grouped output projections. |
| [[attention_type]] | Active when `attention_type: 'compressed'`. |

---

## 5. Practical Scenarios

- **Edge / Mobile Architecture Design:** Use `o_groups: 4` or `8` to aggressively reduce projection FLOPs without sacrificing head count.
- **Standard Pretraining:** Leave at `0`.

---

### One-line intuition

> **`o_groups` splits the attention output projection into $G$ independent group linear matrices, dividing parameter count and FLOPs by $G$.**
