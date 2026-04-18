#!/usr/bin/env python3
"""
Create (if needed) the Hugging Face Space and upload this folder.

Usage:
  export HF_TOKEN=hf_...                    # https://huggingface.co/settings/tokens
  export HF_USERNAME=your_hf_username     # optional; defaults to amayvarghese
  python scripts/deploy_hf_space.py

Space slug is fixed to: video-world-lingbot
→ https://huggingface.co/spaces/<HF_USERNAME>/video-world-lingbot
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_SLUG = "video-world-lingbot"


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) to a write-capable Hugging Face token.", file=sys.stderr)
        return 1

    user = os.environ.get("HF_USERNAME", "amayvarghese").strip().strip("/")
    repo_id = f"{user}/{SPACE_SLUG}"

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    print(f"Ensuring Space exists: {repo_id} …")
    # Must be "docker" so the Hub runs our Dockerfile. "gradio" uses the managed
    # Python 3.13 image and ignores Dockerfile — that path hit diffusers/lingbot-world conflicts.
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        private=False,
        space_sdk="docker",
        exist_ok=True,
    )

    print(f"Uploading {REPO_ROOT} → {repo_id} …")
    api.upload_folder(
        folder_path=str(REPO_ROOT),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[".git/**", ".venv/**", "__pycache__/**", "*.pyc", ".DS_Store"],
    )

    print(f"Done. Open: https://huggingface.co/spaces/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
