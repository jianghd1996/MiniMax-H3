#!/usr/bin/env python3
"""Offline MiniMax-H3 T2VA inference with Modular Diffusers on two A100 GPUs."""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# Do this before importing Hugging Face libraries. Local inference must never
# fall back to the Hub when a checkpoint file is missing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MiniMax-H3 text-to-video+audio directly with Diffusers."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/mnt/DataPart/jianghongda/checkpoint/MiniMax-H3"),
    )
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", help="Generate once with this prompt")
    prompt.add_argument("--prompt-file", type=Path, help="UTF-8 prompt file")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, help="Output path for one-shot mode")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-smaller-gpus",
        action="store_true",
        help="Attempt CPU offload below 70 GiB VRAM; may still run out of memory.",
    )
    return parser.parse_args()


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"ERROR: {message}")


def validate_environment(args: argparse.Namespace) -> None:
    required = [
        args.model_path / "model_index.json",
        args.model_path / "transformer",
        args.model_path / "text_encoder",
        args.model_path / "vae",
        args.model_path / "audio_vae",
        args.model_path / "scheduler",
        args.model_path / "audio_scheduler",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        fail("incomplete Diffusers checkpoint; missing:\n  " + "\n  ".join(missing))

    if not torch.cuda.is_available():
        fail("CUDA is not available in this Python environment")
    if torch.cuda.device_count() < 2:
        fail(
            "two visible GPUs are required; run with "
            "CUDA_VISIBLE_DEVICES=<gpu0>,<gpu1>"
        )
    if args.width % 32 or args.height % 32:
        fail("--width and --height must be multiples of 32")
    if not 5.0 <= args.duration <= 15.0:
        fail("--duration must be between 5 and 15 seconds")
    if args.steps < 2:
        fail("--steps must be at least 2")

    memory_gib = [
        torch.cuda.get_device_properties(index).total_memory / 1024**3
        for index in range(2)
    ]
    for index, gib in enumerate(memory_gib):
        name = torch.cuda.get_device_name(index)
        print(f"GPU {index}: {name}, {gib:.1f} GiB")

    if min(memory_gib) < 70 and not args.allow_smaller_gpus:
        fail(
            "BF16 split inference expects two roughly 80 GiB GPUs. "
            "For A100 40GB, an INT8/streamed-offload path is required; "
            "use --allow-smaller-gpus only to attempt the slower CPU-offload path."
        )


def import_h3_diffusers():
    try:
        from diffusers import ComponentsManager, ModularPipeline
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
        from diffusers.utils.export_utils import encode_video
    except ImportError as exc:
        fail(
            "the installed Diffusers does not contain MiniMax-H3 Modular "
            "Diffusers support. Install a current Diffusers source checkout "
            "in an isolated environment; do not install vLLM. "
            f"Original import error: {exc}"
        )
    return ComponentsManager, ModularPipeline, MiniMaxH3Blocks, encode_video


def load_two_gpu_pipeline(model_path: Path):
    ComponentsManager, ModularPipeline, _, encode_video = import_h3_diffusers()

    print("Building the T2VA workflow...", flush=True)
    base = ModularPipeline.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    workflow = base.blocks.get_workflow("t2va")
    del base

    # The official two-GPU split places the 62 GB Qwen3-VL conditioner on
    # cuda:1 and the 61.7 GB H3 DiT plus VAEs on cuda:0.
    text_block = workflow.sub_blocks.pop("text_encoder")

    text_manager = ComponentsManager()
    text_manager.enable_auto_cpu_offload(
        device="cuda:1",
        memory_reserve_margin="8GB",
    )
    conditioner = text_block.init_pipeline(
        str(model_path),
        components_manager=text_manager,
    )
    print("Loading Qwen3-VL conditioner on cuda:1...", flush=True)
    conditioner.load_components(dtype=torch.bfloat16)

    generation_manager = ComponentsManager()
    generation_manager.enable_auto_cpu_offload(
        device="cuda:0",
        memory_reserve_margin="8GB",
    )
    generator_pipe = workflow.init_pipeline(
        str(model_path),
        components_manager=generation_manager,
    )
    print("Loading H3 transformer and VAEs on cuda:0...", flush=True)
    generator_pipe.load_components(dtype=torch.bfloat16)

    return conditioner, generator_pipe, encode_video


def duration_to_frames(duration: float, fps: int = 24) -> int:
    # H3's temporal VAE accepts frame counts of 17*n+5.
    requested = math.ceil(duration * fps)
    n = max(0, math.ceil((requested - 5) / 17))
    frames = 17 * n + 5
    if frames / fps > 15.1:
        fail(f"duration snaps to {frames / fps:.3f}s, above H3's limit")
    return frames


@torch.inference_mode()
def generate(
    conditioner,
    generator_pipe,
    encode_video,
    prompt: str,
    output_path: Path,
    args: argparse.Namespace,
    seed: int,
) -> None:
    prompt = prompt.strip()
    if not prompt:
        fail("prompt is empty")
    if len(prompt) > 7000:
        fail("prompt exceeds H3's 7000-character limit")

    num_frames = duration_to_frames(args.duration)
    actual_duration = num_frames / 24
    print(
        f"Generating {num_frames} frames ({actual_duration:.3f}s), "
        f"{args.width}x{args.height}, seed={seed}...",
        flush=True,
    )

    state = conditioner(prompt=prompt)
    results = generator_pipe(
        state=state,
        width=args.width,
        height=args.height,
        num_frames=num_frames,
        num_inference_steps=args.steps,
        generator=torch.Generator().manual_seed(seed),
        output=["videos", "audio", "sampling_rate"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        results["videos"][0],
        fps=24,
        output_path=str(output_path),
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    print(f"Saved: {output_path.resolve()}", flush=True)


def main() -> int:
    args = parse_args()
    validate_environment(args)
    conditioner, generator_pipe, encode_video = load_two_gpu_pipeline(args.model_path)

    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    else:
        prompt = args.prompt

    if prompt is not None:
        output = args.output or args.output_dir / "h3_output.mp4"
        generate(
            conditioner,
            generator_pipe,
            encode_video,
            prompt,
            output,
            args,
            args.seed,
        )
        return 0

    print("\nH3 is ready. Enter one prompt per generation.")
    print("Submit an empty line (or Ctrl-D) to stop.\n")
    index = 0
    while True:
        try:
            prompt = input("Prompt> ")
        except EOFError:
            print()
            break
        if not prompt.strip():
            break
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = args.output_dir / f"h3_{stamp}_{index + 1}.mp4"
        generate(
            conditioner,
            generator_pipe,
            encode_video,
            prompt,
            output,
            args,
            args.seed + index,
        )
        index += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
