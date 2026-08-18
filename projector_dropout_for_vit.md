## 1. Why does `projector_dropout_for_vit` exist?

During multimodal alignment training, the projector MLP bridges frozen visual features to the language model. When fine-tuning on limited multimodal instruction datasets, the projector can overfit to specific visual artifact patterns or training prompts.

`projector_dropout_for_vit` introduces dropout regularization inside the projector MLP to prevent overfitting.

```text
ViT Features ──► [ Projector Linear 1 ] ──► [ Activation ] ──► [ Dropout (rate = projector_dropout_for_vit) ] ──► [ Projector Linear 2 ]
```

---

## 2. Options and Defaults

| Value | Behavior |
|---|---|
| `0.0` (Default) | Dropout disabled (standard for large-scale pre-training) |
| `0.05` / `0.1` | Moderate dropout applied during low-resource fine-tuning |

---

## 3. Interactions

- **`enable_dropout`**: Global dropout toggle. If `enable_dropout: false`, projector dropout is also deactivated.

---

### One-line intuition
> **`projector_dropout_for_vit` specifies the dropout probability inside the multimodal vision projector to regularize cross-modal alignment training.**
