
## 1. Why does `mtp_num_layers` exist?

Standard language model training uses a single next-token prediction objective:

```text
Input: [t_0, t_1, ..., t_{n-1}]
Target: [t_1, t_2, ..., t_n]     (predict next token at each position)
```

This gives one loss signal per position. Multi-Token Prediction (MTP), introduced in DeepSeek-V3, adds **auxiliary prediction heads** that each predict a token *further in the future*:

```text
Main head (depth 0): predicts t_{i+1}    (standard)
MTP head 1:          predicts t_{i+2}    (one step ahead)
MTP head 2:          predicts t_{i+3}    (two steps ahead)
```

Each auxiliary head provides an additional training signal. The intuition: forcing the model to predict multiple future tokens makes it learn more forward-looking, compositional representations — improving training efficiency and downstream quality.

`mtp_num_layers` controls how many auxiliary prediction heads are added.

---

## 2. Default

```yaml
mtp_num_layers: 0
```

`0` = feature disabled. MTP is an optional add-on. Set `mtp_num_layers > 0` only when you want to reproduce a DeepSeek-V3-style training setup or explicitly experiment with multi-token prediction.

---

## 3. Architecture with MTP enabled

```text
Main transformer body
    ↓
Hidden states at each position
    ↓
├── Main prediction head → loss_0 (standard next-token loss)
├── MTP module 1
│     (shared transformer block + prediction head)
│     → loss_1 (predict 2 tokens ahead)
└── MTP module N
      (shared transformer block + prediction head)
      → loss_N (predict N+1 tokens ahead)
```

Each MTP module is a lightweight transformer block (typically using shared parameters) with its own prediction head. The total loss is:

```text
final_loss = main_loss + mtp_loss_scaling_factor × mean(mtp_losses)
```

---

## 4. Training-only feature

MTP heads are **only active during training**. At inference, the model's forward pass uses only the main body and main prediction head. The auxiliary heads are unused.

The training cost is proportional to `mtp_num_layers` since each requires its own forward and backward pass through the auxiliary module.

---

## 5. When to use

| Scenario | Setting |
|---|---|
| Standard pretraining | `0` (disabled) |
| Replicating DeepSeek-V3 | `1` (one auxiliary head, as in the paper) |
| Research: varying MTP depth | `2`, `3`, ... |
| Speculative decoding prep (approximate) | `1`–`3` |

DeepSeek-V3 used `mtp_num_layers=1` in pretraining.

---

## 6. Interaction with `mtp_loss_scaling_factor`

```yaml
mtp_num_layers: 1
mtp_loss_scaling_factor: 0.1
```

The scaling factor weights the auxiliary loss contribution. Too large → auxiliary loss dominates, potentially harming main task quality. Too small → MTP provides no useful signal. 0.1 is the published DeepSeek-V3 value.

---

## 7. Interaction with `mtp_eval_target_module`

During evaluation, `mtp_eval_target_module` selects which MTP head to evaluate (e.g., track token acceptance rate). Setting `mtp_num_layers=0` and `mtp_eval_target_module > 0` is invalid — you can't evaluate heads that don't exist.

---

### One-line intuition

> **`mtp_num_layers` adds N auxiliary transformer heads that each predict tokens further in the future, providing extra training signal — `0` disables the feature; `1` replicates DeepSeek-V3's setup.**
