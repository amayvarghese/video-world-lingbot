---
title: video-world-lingbot
emoji: 🌍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
app_file: app.py
suggested_hardware: a10g-large
pinned: false
license: apache-2.0
---

# video-world-lingbot — LingBot World (image → world)

This Space is a Gradio front-end for **LingBot-World**, an open image-to-video world model built on **Wan 2.2** with **camera-pose conditioning**. Upload a starting image, describe the world in text, pick a camera trajectory, and generate a short world video (480p-class resolution, ~16 FPS).

**Source code:** [github.com/amayvarghese/video-world-lingbot](https://github.com/amayvarghese/video-world-lingbot)

## Hugging Face Space setup (name: `video-world-lingbot`)

The public URL will be: `https://huggingface.co/spaces/<your_hf_username>/video-world-lingbot`

This repo uses the **`docker` SDK** (see `Dockerfile`): installs compatible **`diffusers` / `transformers`** pins, then **`lingbot-world` with `--no-deps`** so **`flash_attn` is not required** at build time (the app uses PyTorch SDPA instead).

### Option A — Create from the website (recommended)

1. Open [Create new Space](https://huggingface.co/new-space).
2. **Space name:** `video-world-lingbot` (must match this slug if you want the same URL).
3. **SDK:** **Docker** (or import from GitHub — the README front matter sets `sdk: docker`).
4. **Hardware:** pick a **large GPU** (e.g. A10G / L4); this model is multi‑GB and needs VRAM.
5. **Repo:** under **“Import from GitHub”** pick  
   [`amayvarghese/video-world-lingbot`](https://github.com/amayvarghese/video-world-lingbot), branch **`main`**.
6. After each push, the Space rebuilds from the `Dockerfile` (first build can take a long time).

### Option B — Push from your machine (CLI)

With a [HF access token](https://huggingface.co/settings/tokens) (role: **write**):

```bash
cd /path/to/video-world-lingbot   # repo root: app.py lives here
export HF_TOKEN=hf_your_token_here
export HF_USERNAME=your_hf_username   # optional; defaults to amayvarghese in the script
pip install huggingface_hub
python scripts/deploy_hf_space.py
```

If your Hugging Face username is **not** `amayvarghese`, set `HF_USERNAME` so the Space is created under your account.

## Model and links

- **Weights:** [robbyant/lingbot-world-base-cam](https://huggingface.co/robbyant/lingbot-world-base-cam) (Apache 2.0)
- **Code and paper:** [robbyant/lingbot-world](https://github.com/robbyant/lingbot-world)
- **Technical report:** [arXiv:2601.20540](https://arxiv.org/abs/2601.20540)

The published checkpoints use **Diffusers-compatible** `ModelMixin` transformer shards (`high_noise_model/`, `low_noise_model/`). This demo runs inference through the upstream **`WanI2V`** stack from the LingBot-World repository (same as the authors’ `generate.py`), which loads those shards via Diffusers’ model loader.

## Usage

1. Upload a clear **starting image** (the model will animate the scene from this frame).
2. Enter a **world prompt** describing motion, lighting, and content.
3. Choose a **camera trajectory** (forward fly, pan, orbit, or zoom).
4. Adjust **frames**, **guidance scale**, and **inference steps** if needed.
5. Click **Generate**.

**VRAM:** The base model is large (~14B-class). A GPU with sufficient memory is required for practical runtimes. CPU-only runs are not supported for full inference.

## Local install

The LingBot-World **Python package** (`wan`) is installed from GitHub **without** its published dependency metadata, because upstream pins `flash_attn` (often unavailable on macOS / CPU) and versions that conflict with **Gradio 4.44**’s compatible `huggingface_hub` range. This demo patches attention to use **PyTorch SDPA** when FlashAttention is missing.

```bash
cd lingbot-gradio
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install "git+https://github.com/robbyant/lingbot-world.git" --no-deps
```

If `pip install -r requirements.txt` fails while building `flash_attn`, install base dependencies first, then only the LingBot **code** (as above). Hugging Face **GPU** Spaces often succeed with the single `requirements.txt` install because Linux CUDA wheels exist for `flash_attn` on supported runtimes.

Then start the UI:

```bash
python app.py
```

The first launch downloads weights from Hugging Face (multi‑GB).

## Credits

- **LingBot-World** — [Robbyant / robbyant/lingbot-world](https://github.com/robbyant/lingbot-world)
- **Wan 2.2** — [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
