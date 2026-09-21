#!/usr/bin/env python3
"""Generate a MiniMax-H3 text-to-audio-video clip through local vLLM-Omni."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="H3 prompt text")
    prompt.add_argument("--prompt-file", type=Path, help="UTF-8 prompt file")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("output_h3.mp4"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 4 <= args.duration <= 15:
        raise SystemExit("--duration must be between 4 and 15 seconds")
    if args.width % 32 or args.height % 32:
        raise SystemExit("--width and --height must be multiples of 32")

    prompt = (
        args.prompt_file.read_text(encoding="utf-8").strip()
        if args.prompt_file
        else args.prompt.strip()
    )
    if not prompt:
        raise SystemExit("prompt is empty")
    if len(prompt) > 7000:
        raise SystemExit("prompt exceeds H3's 7000-character limit")

    server = args.server.rstrip("/")
    try:
        health = requests.get(f"{server}/health", timeout=10)
        health.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"H3 server is not ready at {server}: {exc}") from exc

    data = {
        "prompt": prompt,
        "width": str(args.width),
        "height": str(args.height),
        "aspect_ratio": f"{args.width // _gcd(args.width, args.height)}:"
        f"{args.height // _gcd(args.width, args.height)}",
        "fps": "24",
        "num_inference_steps": str(args.steps),
        "flow_shift": str(args.flow_shift),
        "seed": str(args.seed),
        "extra_params": json.dumps(
            {
                "task": "t2va",
                "duration": args.duration,
                "audio_flow_shift": args.audio_flow_shift,
            }
        ),
    }

    print(f"Generating {args.duration}s at {args.width}x{args.height}...", flush=True)
    try:
        response = requests.post(f"{server}/v1/videos/sync", data=data)
        response.raise_for_status()
    except requests.RequestException as exc:
        body = getattr(exc.response, "text", "")[:2000] if exc.response else ""
        raise SystemExit(f"generation failed: {exc}\n{body}") from exc

    content_type = response.headers.get("content-type", "")
    if "video/" not in content_type and not response.content.startswith(b"\x00\x00"):
        raise SystemExit(
            f"server returned {content_type or 'unknown content type'} instead of MP4:\n"
            f"{response.text[:2000]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(response.content)
    print(f"Saved: {args.output.resolve()}")
    return 0


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


if __name__ == "__main__":
    sys.exit(main())
