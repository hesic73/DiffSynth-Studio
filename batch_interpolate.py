#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

console = Console()

_MODEL_CONFIGS = [
    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
                origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
                origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
]

_DEFAULT_PROMPT = (
    "A small boat bravely rides the wind and waves forward. The azure sea is turbulent, "
    "white waves crash against the hull, but the boat fearlessly sails steadily towards the distance. "
    "Sunlight shines on the water surface, sparkling with golden light, adding a touch of warmth to this magnificent scene."
)


def parse_ranges(ranges_str: str | None) -> List[Tuple[int | None, int | None]]:
    """Parse range specification string into list of (start, end) tuples."""
    if not ranges_str:
        return [(None, None)]

    ranges = []
    for range_part in ranges_str.split(','):
        range_part = range_part.strip()
        if ':' not in range_part:
            raise ValueError(f"Invalid range format: {range_part}. Expected 'start:end'")

        start_str, end_str = range_part.split(':', 1)
        start = int(start_str.strip()) if start_str.strip() else None
        end_val = end_str.strip()

        if end_val == '-1' or end_val == '':
            end = None
        else:
            end = int(end_val)

        ranges.append((start, end))

    return ranges


def in_ranges(clip_id: int, ranges: List[Tuple[int | None, int | None]]) -> bool:
    """Check if a clip_id falls within any of the specified ranges."""
    if ranges == [(None, None)]:
        return True

    for start, end in ranges:
        in_range = True
        if start is not None and clip_id < start:
            in_range = False
        if end is not None and clip_id >= end:
            in_range = False
        if in_range:
            return True

    return False


def find_pending_clips(
    clip_base: Path,
    output_base: Path,
    ranges: List[Tuple[int | None, int | None]] | None = None,
    skip_existing: bool = True,
    descending: bool = False,
) -> List[int]:
    """Find clip IDs that need processing based on range and existence filters."""
    if ranges is None:
        ranges = [(None, None)]

    all_clips = [d for d in clip_base.iterdir() if d.is_dir()]

    # Build list of (clip_id, clip_path) tuples for proper numeric sorting
    clip_tuples = []
    for clip_path in all_clips:
        try:
            clip_id = int(clip_path.name)
            clip_tuples.append((clip_id, clip_path))
        except ValueError:
            continue

    # Sort by clip_id (numeric)
    clip_tuples.sort(key=lambda x: x[0], reverse=descending)

    # Filter by range and existence
    pending = []
    for clip_id, clip_path in clip_tuples:
        if not in_ranges(clip_id, ranges):
            continue

        if skip_existing:
            output_path = output_base / f"{clip_id}.mp4"
            if output_path.exists():
                continue

        pending.append(clip_id)

    return pending


def _read_specific_frames(path: Path, indices: Sequence[int]) -> Dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: Dict[int, np.ndarray] = {}
    for idx in sorted(set(indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Unable to read frame {idx} from video {path}")
        frames[idx] = frame
    cap.release()
    return frames


def load_clip_bundle(folder: str, frame_indices: Sequence[int] = (0, 9, 19)) -> Dict[str, np.ndarray]:
    folder_path = Path(folder).expanduser().resolve()
    pose_npz = folder_path / "vipe_results" / "pose.npz"

    if pose_npz.exists():
        with np.load(pose_npz) as data:
            extrinsics = data["data"].astype(np.float32)
            inds = data["inds"].astype(np.int64)
        rotations = extrinsics[:, :3, :3]
        translations = extrinsics[:, :3, 3]
        camera_positions = -np.einsum("nij,ni->nj", np.transpose(rotations, (0, 2, 1)), translations)
    else:
        extrinsics = np.empty((0, 4, 4), dtype=np.float32)
        inds = np.empty((0,), dtype=np.int64)
        camera_positions = np.empty((0, 3), dtype=np.float32)

    original_frames = _read_specific_frames(folder_path / "original.mp4", frame_indices)

    return {
        "extrinsics": extrinsics,
        "indices": inds,
        "camera_positions": camera_positions,
        "original_frames": original_frames,
    }


def _bgr_to_pil(image_array) -> Image.Image:
    return Image.fromarray(image_array[..., ::-1])


def load_start_end_images(clip_folder: Path, start_frame: int, end_frame: int):
    bundle = load_clip_bundle(str(clip_folder), frame_indices=(start_frame, end_frame))
    frames = bundle["original_frames"]
    return _bgr_to_pil(frames[start_frame]), _bgr_to_pil(frames[end_frame])


def create_status_table(
    total_clips: int,
    completed: int,
    failed: int,
    current_clip: str,
    elapsed_time: float,
) -> Table:
    """Create a status table with overall statistics."""
    table = Table(title="Processing Status", show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    remaining = total_clips - completed - failed
    table.add_row("Total Clips", f"{total_clips}")
    table.add_row("✓ Completed", f"[green]{completed}[/green]")
    table.add_row("✗ Failed", f"[red]{failed}[/red]")
    table.add_row("⏳ Remaining", f"[yellow]{remaining}[/yellow]")

    if current_clip:
        table.add_row("Current", f"[bold cyan]{current_clip}[/bold cyan]")

    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    table.add_row("Elapsed", f"[cyan]{elapsed_str}[/cyan]")

    if completed > 0:
        avg_time = elapsed_time / completed
        eta_seconds = avg_time * remaining
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        table.add_row("ETA", f"[cyan]{eta_str}[/cyan]")

    return table


def parse_args():
    parser = argparse.ArgumentParser(description="Batch video frame interpolation")
    parser.add_argument("--clip-base", type=Path, required=True,
                        help="Base directory containing numbered clip subdirectories")
    parser.add_argument("--output-base", type=Path, required=True,
                        help="Base output directory")
    parser.add_argument("--ranges", type=str, default=None,
                        help="Comma-separated ranges (e.g., '100:200,250:-1'). Use -1 for open-ended.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output videos. By default, existing videos are skipped.")
    parser.add_argument("--descending", action="store_true",
                        help="Process clips in descending order by ID. By default, processes in ascending order.")
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT,
                        help="Prompt for generation")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Start frame index (default: 0)")
    parser.add_argument("--end-frame", type=int, default=29,
                        help="End frame index (default: 29)")
    parser.add_argument("--num-intermediate-frames", type=int, default=27,
                        help="Number of intermediate frames (default: 27)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Output video FPS (default: 15)")
    parser.add_argument("--quality", type=int, default=5,
                        help="Output video quality (default: 5)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16",
                        help="Torch dtype (default: bfloat16)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate paths
    clip_base = args.clip_base.expanduser().resolve()
    output_base = args.output_base.expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    # Parse ranges
    parsed_ranges = parse_ranges(args.ranges) if args.ranges else None

    # Print configuration
    console.print("\n[bold cyan]═" * 30)
    console.print("[bold cyan]Wan Video Frame Interpolation")
    console.print("[bold cyan]═" * 30)
    console.print(f"[cyan]Clip base:[/cyan] {clip_base}")
    console.print(f"[cyan]Output:[/cyan] {output_base}")

    # Show range info
    if parsed_ranges and parsed_ranges != [(None, None)]:
        ranges_str = ", ".join([f"[{s if s is not None else '0'}:{e if e is not None else '∞'})" for s, e in parsed_ranges])
        console.print(f"[cyan]Range filter:[/cyan] {ranges_str}")
    if args.overwrite:
        console.print(f"[yellow]Overwrite mode:[/yellow] enabled (will reprocess existing)")
    else:
        console.print(f"[cyan]Skip existing:[/cyan] enabled (default)")

    order_str = "descending" if args.descending else "ascending"
    console.print(f"[cyan]Processing order:[/cyan] {order_str}")

    console.print(f"[cyan]Frame range:[/cyan] {args.start_frame} to {args.end_frame}")
    console.print(f"[cyan]Intermediate frames:[/cyan] {args.num_intermediate_frames}")
    console.print(f"[cyan]Device:[/cyan] {args.device}")
    console.print("[bold cyan]═" * 30 + "\n")

    # Find pending clips
    console.print("[bold cyan]Scanning for clips to process...[/bold cyan]")
    skip_existing = not args.overwrite
    pending_clips = find_pending_clips(
        clip_base, output_base, ranges=parsed_ranges, skip_existing=skip_existing, descending=args.descending
    )

    if not pending_clips:
        console.print(f"[green]No clips to process. All clips already exist![/green]")
        return

    console.print(f"[bold]Found {len(pending_clips)} clips to process[/bold]\n")

    # Load model
    console.print("[yellow]Loading model...[/yellow]")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.torch_dtype.lower()]

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=_MODEL_CONFIGS,
    )
    pipe.enable_vram_management()
    console.print("[green]✓ Model loaded successfully[/green]\n")

    # Initialize statistics
    total_clips = len(pending_clips)
    completed_count = 0
    failed_count = 0
    start_time = time.time()

    # Create progress bar
    clip_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    )

    # Process clips with live display
    with Live(console=console, refresh_per_second=4) as live:
        clip_task = clip_progress.add_task("Processing Clips", total=total_clips)

        for clip_id in pending_clips:
            clip_folder = clip_base / str(clip_id)
            output_path = output_base / f"{clip_id}.mp4"

            # Update display
            current_status = create_status_table(
                total_clips=total_clips,
                completed=completed_count,
                failed=failed_count,
                current_clip=str(clip_id),
                elapsed_time=time.time() - start_time,
            )

            display_group = Table.grid()
            display_group.add_row(Panel(current_status, border_style="blue"))
            display_group.add_row(clip_progress)
            live.update(display_group)

            # Check if clip folder exists
            if not clip_folder.is_dir():
                failed_count += 1
                clip_progress.update(clip_task, advance=1)
                continue

            try:
                start_image, end_image = load_start_end_images(clip_folder, args.start_frame, args.end_frame)
                height, width = start_image.height, start_image.width
                num_frames = args.num_intermediate_frames + 2

                video = pipe(
                    prompt=args.prompt,
                    negative_prompt="",
                    input_image=start_image,
                    end_image=end_image,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    seed=args.seed,
                    tiled=True,
                )

                save_video(video, str(output_path), fps=args.fps, quality=args.quality)
                completed_count += 1

            except Exception as exc:
                failed_count += 1
                console.print(f"[red]✗ ERROR processing clip {clip_id}: {exc}[/red]")

            clip_progress.update(clip_task, advance=1)

    # Final summary
    elapsed_time = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    summary = Text()
    summary.append("\n")
    summary.append("PROCESSING COMPLETE\n\n", style="bold green")
    summary.append(f"Total time:      {elapsed_str}\n", style="cyan")
    summary.append(f"Total clips:     {total_clips}\n", style="bold")
    summary.append(f"✓ Successful:    {completed_count}\n", style="bold green")
    summary.append(f"✗ Failed:        {failed_count}\n", style="bold red")
    summary.append(f"\nOutput directory: {output_base}\n", style="cyan")

    border_style = "green" if failed_count == 0 else "yellow"
    console.print(Panel(summary, border_style=border_style, title="Summary"))


if __name__ == "__main__":
    main()
