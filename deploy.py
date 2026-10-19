#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy the drone roof inspection demo: a vision-language model and the inspection app.

Two Chalk ScalingGroups:
  <prefix>-vlm  vLLM serving Qwen3-VL-8B-Instruct on one L40S (authenticated)
  <prefix>      the inspection app (CPU): samples keyframes with ffmpeg, asks the model for boxed
                findings per frame, then for an adjuster summary, and streams it all to the page

The sample videos are public-domain U.S. Army Corps of Engineers drone footage from DVIDS,
downloaded and trimmed at build time.

Credentials come from your existing `chalk login` (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>] [--prefix roof-claims] [--gpu nvidia-l40s]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chalkcompute import ScalingGroup

HERE = Path(__file__).resolve().parent

VLLM_IMAGE = "vllm/vllm-openai:v0.19.1"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

# DVIDS video IDs 860100 (Operation Blue Roof, Hurricane Ian, Fort Myers, 2022) and 815901
# (Blue Roof drone b-roll, Hurricane Ida, LaPlace, 2021). Both are marked public domain.
SAMPLES = {
    # name: (source URL, ffmpeg trim arguments, ffmpeg video filter)
    "ian": (
        "https://d34w7g4gy10iej.cloudfront.net/video/2210/DOD_109260431/DOD_109260431.mp4",
        # Skip the title card; keep the close pass over one damaged roof and the tarp going on.
        "-ss 12 -to 47",
        "scale=1280:-2",
    ),
    "ida": (
        "https://d34w7g4gy10iej.cloudfront.net/video/2109/DOD_108601796/DOD_108601796-1024x576-1769k.mp4",
        "-ss 0 -to 60",
        "null",
    ),
}


def deploy(sg: ScalingGroup, ready_timeout: float) -> str:
    t0 = time.monotonic()
    sg.deploy(build_timeout=2400.0, ready_timeout=ready_timeout)
    url = sg.web_url
    if url is None:
        raise RuntimeError(f"{sg!r} is ready but has no web_url")
    print(f"{sg!r}: ready in {time.monotonic() - t0:.0f}s at {url}")
    return url


def sample_commands() -> list[str]:
    cmds = ["mkdir -p /samples /tmp/raw"]
    for name, (url, trim, vf) in SAMPLES.items():
        cmds.append(f"python -c \"import urllib.request; urllib.request.urlretrieve('{url}', '/tmp/raw/{name}.mp4')\"")
        # H.264 + faststart so the browser can start playing before the file finishes loading.
        cmds.append(
            f"ffmpeg -v error {trim} -i /tmp/raw/{name}.mp4 -vf {vf} "
            + f"-c:v libx264 -preset veryfast -crf 26 -an -movflags +faststart /samples/{name}.mp4"
        )
    cmds.append("rm -rf /tmp/raw")
    return cmds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="roof-claims")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--gpu", default="nvidia-l40s")
    parser.add_argument("--ready-timeout", type=float, default=2400.0)
    # Redeploy only the app and reuse the running model group, so a code change does not restart
    # (and re-download) a 17 GB model.
    parser.add_argument("--app-only", action="store_true")
    args = parser.parse_args()

    if args.env_id:
        os.environ["CHALK_ENVIRONMENT"] = args.env_id
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    if args.app_only:
        vlm_url = ScalingGroup.from_name(f"{args.prefix}-vlm").web_url
        if vlm_url is None:
            raise RuntimeError(f"{args.prefix}-vlm has no web_url; deploy without --app-only first")
    else:
        vlm = ScalingGroup(
            image=Image.base(VLLM_IMAGE),
            name=f"{args.prefix}-vlm",
            gpu=args.gpu,
            # Leaves room for node daemonsets on an 8-vCPU, 64 GB g6e.2xlarge; asking for all 8 vCPUs
            # makes the pod unschedulable on any L40S size Karpenter offers.
            cpu="4",
            memory="24Gi",
            env={"HF_XET_HIGH_PERFORMANCE": "1"},
            entrypoint=[
                "vllm",
                "serve",
                MODEL,
                "--revision",
                MODEL_REVISION,
                "--served-model-name",
                MODEL,
                "--port",
                "8080",
                "--max-model-len",
                "16384",
                "--limit-mm-per-prompt",
                '{"image": 5, "video": 0}',
                "--gpu-memory-utilization",
                "0.90",
            ],  # fmt: skip
            port=8080,
            authenticated=True,
            # vLLM answers /health once the weights are loaded and the engine is up.
            readiness_probe=ReadinessProbe.http("/health"),
            min_replicas=1,
            max_replicas=1,
        )
        vlm_url = deploy(vlm, args.ready_timeout)

    image = (
        Image.debian_slim(python_version="3.12")
        .apt_install(["ffmpeg"])
        .pip_install(["fastapi==0.116.1", "uvicorn[standard]==0.35.0", "httpx==0.28.1", "python-multipart==0.0.20"])
        .run_commands(*sample_commands())
        .add_local_file(str(HERE / "app.py"), "/app/app.py", strategy="copy")
        .add_local_file(str(HERE / "index.html"), "/app/index.html", strategy="copy")
        .workdir("/app")
    )
    app = ScalingGroup(
        image=image,
        name=args.prefix,
        cpu="2",
        memory="4Gi",
        env={"VLM_URL": vlm_url, "VLM_MODEL": MODEL},
        entrypoint=["python", "/app/app.py"],
        port=8080,
        chalk_identity=True,
        readiness_probe=ReadinessProbe.http("/api/status"),
        min_replicas=1,
        max_replicas=1,
    )
    print("web_url:", deploy(app, args.ready_timeout))


if __name__ == "__main__":
    main()
