"""
Gradio Space for LingBot-World (camera-conditioned Wan 2.2 I2V).

Weights: https://huggingface.co/robbyant/lingbot-world-base-cam
Upstream inference: https://github.com/robbyant/lingbot-world
"""

from __future__ import annotations

import os
import tempfile
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Upstream `wan` calls `torch.cuda.current_device()` at import time (see
# `wan/modules/t5.py`). CPU-only PyTorch wheels would otherwise crash before
# the UI loads; stub only `current_device` so imports succeed on macOS/CI.
# ---------------------------------------------------------------------------
_real_cuda_is_available = torch.cuda.is_available
if not _real_cuda_is_available():
    torch.cuda.current_device = lambda: 0  # type: ignore[method-assign]

# ---------------------------------------------------------------------------
# Hugging Face Spaces ZeroGPU: optional `spaces` import for local dev.
# ---------------------------------------------------------------------------
try:
    import spaces
except ImportError:  # pragma: no cover - local without `spaces`
    spaces = None  # type: ignore

from diffusers import DiffusionPipeline  # noqa: F401
from huggingface_hub import snapshot_download

# DiT checkpoints in the Hub repo are Diffusers `ModelMixin` / safetensors trees.
assert DiffusionPipeline is not None

MODEL_REPO = "robbyant/lingbot-world-base-cam"
# 480p-class area cap (matches LingBot `480*832` preset and intrinsics_org in cam_utils).
MAX_AREA_480P = 480 * 832

# Upstream default negative prompt (shown truncated in the footer; full string is in the checkpoint config).
_NEG_PROMPT_FOOTER = (
    "画面突变，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部…"
)

_APP_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _APP_DIR / "examples"

_WAN_I2V: Optional[object] = None
_CKPT_DIR: Optional[str] = None
_LOAD_ERR: Optional[str] = None


def _patch_wan_flash_attention() -> None:
    """Route `flash_attention` to PyTorch SDPA when FlashAttention is not installed."""
    import wan.modules.attention as wan_attn

    def _sdpa_flash_attention(*args, version=None, **kwargs):
        return wan_attn.attention(*args, fa_version=version, **kwargs)

    wan_attn.flash_attention = _sdpa_flash_attention


def _round_frames_to_4n_plus_1(n: int) -> int:
    """LingBot / Wan I2V expects F = 4n + 1."""
    n = int(n)
    n = max(5, n)
    k = (n - 1) // 4
    return 4 * k + 1


def _opencv_look_at_c2w(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """Camera-to-world (OpenCV): +X right, +Y down, +Z forward."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)
    z = target - eye
    nz = np.linalg.norm(z)
    if nz < 1e-8:
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z = z / nz
    x = np.cross(world_up, z)
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        world_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x = np.cross(world_up, z)
        nx = np.linalg.norm(x)
    x = x / nx
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T.astype(np.float32)


def build_camera_poses(trajectory: str, num_frames: int) -> np.ndarray:
    """
    Build OpenCV camera-to-world poses [F, 4, 4] for LingBot cam conditioning.
    Trajectories are smooth synthetic paths (demo-quality).
    """
    n = _round_frames_to_4n_plus_1(num_frames)
    poses: list[np.ndarray] = []

    if trajectory == "Forward Fly":
        T = np.eye(4, dtype=np.float32)
        poses.append(T.copy())
        step = 0.035
        for _ in range(1, n):
            R = T[:3, :3]
            t = T[:3, 3]
            forward = R @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
            t = t + step * forward
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = t
            poses.append(T.copy())

    elif trajectory in ("Pan Left", "Pan Right"):
        sign = 1.0 if trajectory == "Pan Left" else -1.0
        max_deg = 14.0
        for i in range(n):
            a = np.deg2rad(sign * max_deg * (i / max(n - 1, 1)))
            c, s = float(np.cos(a)), float(np.sin(a))
            R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            poses.append(T)

    elif trajectory == "Orbit":
        radius = 0.22
        center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        sweep = 0.55 * np.pi
        for i in range(n):
            theta = sweep * (i / max(n - 1, 1))
            eye = np.array([radius * np.sin(theta), 0.04, radius * np.cos(theta)], dtype=np.float64)
            poses.append(_opencv_look_at_c2w(eye, center, world_up))

    elif trajectory == "Zoom In":
        for i in range(n):
            z = 0.06 * (i / max(n - 1, 1))
            T = np.eye(4, dtype=np.float32)
            T[2, 3] = z
            poses.append(T)

    else:
        for _ in range(n):
            poses.append(np.eye(4, dtype=np.float32))

    return np.stack(poses, axis=0)


def default_intrinsics(num_frames: int) -> np.ndarray:
    """[fx, fy, cx, cy] for 832x480 reference grid (matches LingBot cam_utils defaults)."""
    fx = fy = 700.0
    cx = 416.0
    cy = 240.0
    row = np.array([[fx, fy, cx, cy]], dtype=np.float32)
    return np.repeat(row, int(num_frames), axis=0)


def write_action_directory(poses: np.ndarray) -> str:
    d = tempfile.mkdtemp(prefix="lingbot_action_")
    np.save(os.path.join(d, "poses.npy"), poses.astype(np.float32))
    np.save(os.path.join(d, "intrinsics.npy"), default_intrinsics(len(poses)))
    return d


def tensor_to_mp4(video: torch.Tensor, out_path: Path, fps: int = 16) -> str:
    """video: (C, F, H, W) in [-1, 1]."""
    import imageio

    arr = video.detach().float().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 3, 0))  # F, H, W, C
    arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        str(out_path),
        list(arr),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    return str(out_path)


def load_model_weights() -> None:
    """Download (if needed) and construct WanI2V once at process startup."""
    global _WAN_I2V, _CKPT_DIR, _LOAD_ERR
    if _WAN_I2V is not None:
        return
    _LOAD_ERR = None
    try:
        if not _real_cuda_is_available():
            raise RuntimeError("CUDA is not available; LingBot-World inference requires a GPU.")

        # `wan` imports call CUDA at module import time; defer until we know a GPU exists.
        _patch_wan_flash_attention()
        from wan.configs import WAN_CONFIGS  # noqa: E402
        from wan.image2video import WanI2V  # noqa: E402

        _CKPT_DIR = snapshot_download(MODEL_REPO)
        cfg = WAN_CONFIGS["i2v-A14B"]
        _WAN_I2V = WanI2V(
            config=cfg,
            checkpoint_dir=_CKPT_DIR,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            init_on_cpu=True,
            convert_model_dtype=False,
        )
    except Exception as e:  # pragma: no cover - environment specific
        _WAN_I2V = None
        _LOAD_ERR = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def _decorate_zero_gpu(fn):
    if spaces is None or not hasattr(spaces, "GPU"):
        return fn
    gpu = spaces.GPU
    # `spaces.GPU` is used as @spaces.GPU or @spaces.GPU(duration=...).
    try:
        wrapped = gpu(duration=2400)(fn)
    except TypeError:
        try:
            wrapped = gpu(fn)
        except TypeError:
            wrapped = gpu()(fn)
    return wrapped


@_decorate_zero_gpu
def run_inference(
    image_path: Optional[str],
    prompt: str,
    trajectory: str,
    num_frames: int,
    guidance_scale: float,
    inference_steps: int,
    progress: Optional[gr.Progress] = None,
):
    if progress is None:
        progress = gr.Progress(track_tqdm=True)

    action_dir: Optional[str] = None
    try:
        if _WAN_I2V is None:
            raise RuntimeError(_LOAD_ERR or "Model is not loaded. Check server logs.")

        if not image_path or not os.path.isfile(image_path):
            raise ValueError("Please upload a valid starting image.")
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Please enter a world prompt.")

        n_frames = _round_frames_to_4n_plus_1(int(num_frames))
        poses = build_camera_poses(trajectory, n_frames)
        action_dir = write_action_directory(poses)

        progress(0.05, desc="Preparing image and camera conditioning…")
        img = Image.open(image_path).convert("RGB")

        cfg = _WAN_I2V.config
        max_area = MAX_AREA_480P
        # 480p-class: lower shift matches upstream README guidance.
        shift = 3.0

        out_mp4 = Path(tempfile.gettempdir()) / f"lingbot_world_{os.getpid()}.mp4"

        progress(0.15, desc="Generating world video (this can take several minutes)…")

        autocast_cm = (
            torch.autocast("cuda", dtype=torch.bfloat16) if _real_cuda_is_available() else nullcontext()
        )

        with autocast_cm:
            video = _WAN_I2V.generate(
                prompt,
                img,
                action_path=action_dir,
                allow_act2cam=False,
                action_string=None,
                vis_ui=False,
                max_area=max_area,
                frame_num=n_frames,
                shift=shift,
                sample_solver="unipc",
                sampling_steps=int(inference_steps),
                guide_scale=float(guidance_scale),
                seed=-1,
                offload_model=True,
            )

        progress(0.92, desc="Encoding MP4…")
        fps = int(getattr(cfg, "sample_fps", 16))
        path = tensor_to_mp4(video, out_mp4, fps=fps)
        progress(1.0, desc="Done.")
        return path

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        gr.Warning(f"Generation failed: {msg}")
        return None
    finally:
        try:
            import shutil

            if action_dir is not None and os.path.isdir(action_dir):
                shutil.rmtree(action_dir, ignore_errors=True)
        except Exception:
            pass


def build_demo():
    device_badge = (
        '<span style="display:inline-block;padding:0.25rem 0.6rem;border-radius:999px;'
        'background:#14532d;color:#bbf7d0;font-weight:600;font-size:0.85rem;">'
        "Running on GPU (CUDA)</span>"
        if _real_cuda_is_available()
        else '<span style="display:inline-block;padding:0.25rem 0.6rem;border-radius:999px;'
        'background:#7f1d1d;color:#fecaca;font-weight:600;font-size:0.85rem;">'
        "CPU only — full inference needs a CUDA GPU</span>"
    )

    intro = f"""
<div style="max-width:920px;margin:0 auto 1rem auto;padding:1rem 1.25rem;border-radius:12px;
border:1px solid rgba(148,163,184,0.35);background:rgba(15,23,42,0.85);color:#e2e8f0;line-height:1.55;">
<div style="margin-bottom:0.5rem;">{device_badge}</div>
<p style="margin:0.4rem 0;"><strong>LingBot-World</strong> is an open image-to-video <em>world model</em> built on
<strong>Wan 2.2</strong>. Give it a starting frame, a text description, and a synthetic camera path; it returns a short
world clip (~16 FPS, 480p-class by default).</p>
<p style="margin:0.4rem 0;font-size:0.92rem;opacity:0.9;">Transformer weights are Apache-2.0 checkpoints on Hugging Face
(<code>{MODEL_REPO}</code>), loaded with the upstream <code>WanI2V</code> path from the reference repository (DiT uses
Diffusers <code>ModelMixin</code> / safetensors loaders).</p>
</div>
"""

    custom_css = """
    .gradio-container { max-width: 1100px !important; margin: auto !important; }
    footer.footer { text-align: center; margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.85; }
    @media (max-width: 768px) {
      .gradio-row { flex-direction: column !important; }
    }
    """

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill_dark="#0b1220",
        block_background_fill_dark="#111827",
        block_border_width="1px",
        border_color_primary_dark="#334155",
    )

    ex_files = []
    for name in ("example_city.jpg", "example_astronaut.jpg", "example_landscape.jpg"):
        p = _EXAMPLES_DIR / name
        if p.is_file():
            ex_files.append(str(p))

    with gr.Blocks(
        theme=theme,
        css=custom_css,
        title="LingBot World - Image to World Generator",
    ) as demo:
        gr.Markdown(
            "# LingBot World — Image to World Generator",
            elem_id="title",
        )
        gr.HTML(intro)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=280):
                inp_image = gr.Image(
                    type="filepath",
                    label="Upload your starting image",
                    height=320,
                )
                prompt = gr.Textbox(
                    label="World Prompt",
                    placeholder="A futuristic city at sunset…",
                    lines=3,
                )
                traj = gr.Radio(
                    choices=["Forward Fly", "Pan Left", "Pan Right", "Orbit", "Zoom In"],
                    label="Camera Trajectory",
                    value="Forward Fly",
                )
                nf = gr.Slider(
                    minimum=49,
                    maximum=97,
                    step=16,
                    value=49,
                    label="Number of Frames",
                )
                gs = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    step=0.5,
                    value=6.0,
                    label="Guidance Scale",
                )
                steps = gr.Slider(
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=20,
                    label="Inference Steps",
                )
                gen_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1, min_width=280):
                out_vid = gr.Video(label="Generated World", autoplay=True)

        example_rows = []
        if len(ex_files) >= 1:
            example_rows.append(
                [
                    ex_files[0],
                    "A cinematic drone flight over a neon coastal megacity at blue hour.",
                    "Forward Fly",
                    49,
                    6.0,
                    20,
                ]
            )
        if len(ex_files) >= 2:
            example_rows.append(
                [
                    ex_files[1],
                    "Slow orbital move revealing alien terrain and twin suns on the horizon.",
                    "Orbit",
                    49,
                    6.0,
                    20,
                ]
            )
        if len(ex_files) >= 3:
            example_rows.append(
                [
                    ex_files[2],
                    "Gentle pan across misty mountains as morning light spills into the valley.",
                    "Pan Left",
                    49,
                    6.0,
                    20,
                ]
            )
        if example_rows:
            gr.Examples(
                examples=example_rows,
                inputs=[inp_image, prompt, traj, nf, gs, steps],
                cache_examples=False,
            )

        load_status = gr.Markdown("" if not _LOAD_ERR else f"**Model load error**\n\n```\n{_LOAD_ERR}\n```")

        def _on_gen(ip, pr, tr, n, g, st, prog=gr.Progress()):
            return run_inference(ip, pr, tr, n, g, st, prog)

        gen_btn.click(
            fn=_on_gen,
            inputs=[inp_image, prompt, traj, nf, gs, steps],
            outputs=out_vid,
        )

        gr.Markdown(
            f"""
<footer class="footer">
<p><a href="https://github.com/robbyant/lingbot-world" target="_blank" rel="noopener noreferrer">LingBot-World on GitHub</a>
&nbsp;·&nbsp;
<a href="https://huggingface.co/{MODEL_REPO}" target="_blank" rel="noopener noreferrer">Model card on Hugging Face</a></p>
<p style="font-size:0.8rem;">Negative prompt (fixed, upstream default): <code>{_NEG_PROMPT_FOOTER}</code></p>
</footer>
"""
        )

    return demo, load_status


if __name__ == "__main__":
    print("Loading LingBot-World (first run downloads checkpoints)…")
    load_model_weights()
    if _LOAD_ERR:
        print("WARNING: model load failed:\n", _LOAD_ERR)

    demo, _status = build_demo()

    demo.queue(default_concurrency_limit=1)
    _tmp = str(Path(tempfile.gettempdir()).resolve())
    _on_hf_space = bool(os.environ.get("SPACE_ID") or os.environ.get("SPACES_ZERO_GPU"))
    _server = (
        os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
        if _on_hf_space
        else os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    )
    demo.launch(
        server_name=_server,
        server_port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860"))),
        show_error=True,
        show_api=False,
        allowed_paths=[str(_APP_DIR.resolve()), _tmp],
        # When True, Gradio verifies `localhost` is reachable (fails in some sandboxes). Default off.
        _frontend=os.environ.get("GRADIO_VALIDATE_LOCALHOST", "").lower() in ("1", "true", "yes"),
    )
