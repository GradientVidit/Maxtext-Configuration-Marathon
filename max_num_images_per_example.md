## 1. Why does `max_num_images_per_example` exist?

In multi-image training datasets (e.g., interleaved document pages, multi-turn visual conversations, or before-and-after photo pairs), individual examples contain varying numbers of images. 

In static-shape compiled XLA graphs (like those on TPU), batch tensors must have deterministic static shapes. If unconstrained, MaxText must pad every example to the theoretical maximum number of images that could fit in the sequence, consuming immense amounts of padding memory and wasting ViT computation.

```text
Without max_num_images_per_example (-1):
Batch padded to theoretical max (e.g. 10 images per sample):
Example 1 (1 image):  [ Image 1 ] [ PAD ] [ PAD ] [ PAD ] [ PAD ] [ PAD ] [ PAD ] [ PAD ] [ PAD ] [ PAD ]
                      └───────── 90% Wasted ViT Compute & Memory! ──────────────────────────────┘

With max_num_images_per_example = 2:
Batch padded only to known dataset ceiling (2 images):
Example 1 (1 image):  [ Image 1 ] [ PAD ] (Minimal overhead!)
```

`max_num_images_per_example` explicitly caps the image list padding dimension per example during training.

---

## 2. Options and Defaults

| Value | Behavior | Use Case |
|---|---|---|
| `-1` (Default) | No explicit limit; pads to the maximum allowed by sequence length | General exploratory training |
| Integer $\ge 1$ (e.g. `2`, `4`, `8`) | Strictly caps image slots per example to $N$ | Known dataset limits (e.g. doc-pair training, single-image fine-tuning) |

---

## 3. Practical Tuning

- If your dataset is strictly single-image VQA, set `max_num_images_per_example: 1`. This immediately frees up ViT input buffer memory on TPUs.

---

### One-line intuition
> **`max_num_images_per_example` sets the maximum image padding slots per training example, eliminating redundant ViT compute on padded image slots.**
