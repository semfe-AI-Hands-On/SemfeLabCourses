# Week 7 — Computer Vision & Generative Image Models

## What this lab covers

This lab introduces the core ideas behind modern computer vision and generative image models. Rather than reviewing them in isolation, the notebooks are sequenced to show how image understanding and image generation share the same underlying building blocks — diffusion processes, attention, and conditioning signals.

You will work with four notebooks. Each one is independent; you can run them in any order. By the end you will have run a text-to-image pipeline, an image-conditioned generation pipeline, an inpainting workflow using segmentation masks, and a high-resolution XL model.

### The four notebooks

1. **Text-to-image** — Generate images from a text prompt using Stable Diffusion 1.5. Understand the role of the text encoder, U-Net, and scheduler.
2. **Image + text to image (img2img)** — Start from an existing image and steer generation with both a prompt and a strength parameter.
3. **Mask-guided inpainting (Mask R-CNN)** — Use a segmentation model to detect objects, build a binary mask, and replace or fill the masked region with diffusion.
4. **Diffusion XL** — Run SDXL, a two-stage (base + refiner) high-resolution model. Requires more VRAM; cloud execution recommended if you don't have a capable GPU.

---

## Architecture overview

All four notebooks share the same diffusion backbone:

```
Text prompt / Image
       │
       ▼
┌──────────────┐
│ Text Encoder  │  ── CLIP: converts words to embeddings the U-Net can attend to
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    U-Net      │  ── iteratively denoises a latent representation (typically 20-50 steps)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     VAE       │  ── decodes the final latent into a full-resolution image
└──────────────┘
```

For img2img, the input image is first encoded by the VAE into a latent, partially noised, then denoised. For mask-based inpainting, only the masked region is noised; the rest is kept from the original.

---

## Models used

| Notebook | Model | HuggingFace ID |
|----------|-------|----------------|
| text_to_image | Stable Diffusion 1.5 | `runwayml/stable-diffusion-v1-5` |
| image_text_to_image | Stable Diffusion 1.5 (img2img) | `runwayml/stable-diffusion-v1-5` |
| mask-rcnn | Mask R-CNN + SD 1.5 | torchvision + `runwayml/stable-diffusion-v1-5` |
| diffusion-xl | SDXL Base + Refiner | `stabilityai/stable-diffusion-xl-base-1.0` |

---

## Prerequisites

### Hardware

| Notebook | Minimum | Recommended |
|----------|---------|-------------|
| text_to_image | CPU (slow) | 6 GB GPU |
| image_text_to_image | CPU (slow) | 6 GB GPU |
| mask-rcnn | CPU (slow) | 6 GB GPU |
| diffusion-xl | 10 GB GPU | 16 GB GPU / cloud |

If you don't have a capable GPU, run the first three on CPU (expect 2–5 min per image) and run diffusion-xl on Google Colab or a cloud provider with a T4/A10 instance.

### Software

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| CUDA (optional) | 11.8 or 12.x |

### Python packages

Install dependencies with:

```bash
pip install -r requirements.txt
```

Key packages:
- `diffusers` — pipelines for Stable Diffusion and SDXL
- `transformers` — CLIP text encoder
- `accelerate` — device management
- `torchvision` — Mask R-CNN
- `pillow` — image I/O
- `torch` — PyTorch

---

## Setup

### 1. Set up Python with pyenv

We recommend **Python 3.11**:

```bash
pyenv install 3.11.13
pyenv local 3.11.13
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv myvenv
source myvenv/bin/activate   # Windows: myvenv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run any notebook

```bash
jupyter notebook
```

Open the notebook of your choice and run cells from top to bottom.

Model weights are downloaded automatically from HuggingFace on first run (~3–6 GB depending on the notebook). Subsequent runs load from the local cache.

---

## Lab structure

```
Week7_Computer_Vision/
├── README.md                        ← you are here
├── requirements.txt                 ← Python dependencies
├── text_to_image.ipynb              ← Lab 7a: text → image with SD 1.5
├── image_text_to_image.ipynb        ← Lab 7b: image + text → image (img2img)
├── mask-rcnn.ipynb                  ← Lab 7c: segmentation mask + inpainting
└── diffusion-xl.ipynb               ← Lab 7d: high-resolution SDXL generation
```

## Notes

- Each notebook is **independent** and can be run on its own.
- Model weights are cached by HuggingFace in `~/.cache/huggingface/` after the first download.
- The `diffusion-xl` notebook may not run on most laptops. Cloud execution (Colab, Kaggle, Modal) is recommended.
