#!/usr/bin/env python3
"""Submit MiniMax-H3 T2VA/FL2VA jobs to a local SGLang server."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc.reason}") from exc


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=3600) as response, output.open("wb") as file:
            while chunk := response.read(8 * 1024 * 1024):
                file.write(chunk)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Download failed ({exc.code}): {body}") from exc


def file_uri(value: str) -> str:
    if value.startswith(("http://", "https://", "file://", "data:")):
        return value
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    return path.as_uri()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    raise ValueError("Provide --prompt or --prompt-file.")


def status_from(response: dict[str, Any]) -> str:
    status = response.get("status")
    if status is None and isinstance(response.get("data"), dict):
        status = response["data"].get("status")
    return str(status or "unknown").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiniMax-H3 through local SGLang.")
    parser.add_argument("--server", default="http://127.0.0.1:30010")
    parser.add_argument("--task", choices=("t2va", "fl2va"), default="t2va")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file")
    parser.add_argument("--image", help="First-frame image path or URL; required for fl2va.")
    parser.add_argument("--output", default="outputs/minimax_h3.mp4")
    parser.add_argument("--short-edge", type=int, default=768)
    parser.add_argument("--aspect-ratio", default="16:9", help="For images, 'auto' is recommended.")
    parser.add_argument("--duration", type=int, default=5, choices=range(4, 16), metavar="[4-15]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=7200.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.task == "fl2va" and not args.image:
        raise ValueError("--image is required when --task fl2va is used.")
    if args.task == "t2va" and args.image:
        raise ValueError("--image is only valid with --task fl2va.")

    conditions: list[dict[str, Any]] = []
    if args.image:
        conditions.append(
            {
                "type": "image",
                "uri": file_uri(args.image),
                "role": "keyframe",
                "frame_index": 0,
            }
        )

    payload = {
        "task": args.task,
        "prompt": read_prompt(args),
        "conditions": conditions,
        "target": {
            "short_edge": args.short_edge,
            "aspect_ratio": args.aspect_ratio,
            "duration_seconds": args.duration,
        },
        "seed": args.seed,
    }

    server = args.server.rstrip("/")
    created = request_json(f"{server}/v1/videos", method="POST", payload=payload)
    video_id = created.get("id")
    if not video_id:
        raise RuntimeError(f"Server response did not contain a video id: {created}")
    print(f"Submitted job: {video_id}", flush=True)

    deadline = time.monotonic() + args.timeout
    last_status = ""
    while True:
        job = request_json(f"{server}/v1/videos/{video_id}")
        status = status_from(job)
        if status != last_status:
            print(f"Status: {status}", flush=True)
            last_status = status
        if status in {"completed", "succeeded", "success"}:
            break
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"Generation ended with status '{status}': {json.dumps(job, ensure_ascii=False)}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Generation did not finish within {args.timeout:.0f} seconds.")
        time.sleep(args.poll_interval)

    output = Path(args.output).expanduser().resolve()
    download(f"{server}/v1/videos/{video_id}/content", output)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
