#!/usr/bin/env python3
"""Multi-GPU batch video frame interpolation using Wan pipeline."""

import argparse
import os
import random
import time
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


@dataclass
class WorkerArgs:
    """Arguments passed to worker processes."""
    prompt: str
    start_frame: int
    end_frame: int
    num_intermediate_frames: int
    seed: int
    fps: int
    quality: int
    torch_dtype: str


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


def process_clip(
    clip_id: int,
    clip_base: Path,
    output_base: Path,
    pipe,
    worker_args: WorkerArgs,
) -> None:
    """Process a single clip through the Wan pipeline."""
    from diffsynth import save_video

    clip_folder = clip_base / str(clip_id)
    output_path = output_base / f"{clip_id}.mp4"

    start_image, end_image = load_start_end_images(clip_folder, worker_args.start_frame, worker_args.end_frame)
    height, width = start_image.height, start_image.width
    num_frames = worker_args.num_intermediate_frames + 2

    video = pipe(
        prompt=worker_args.prompt,
        negative_prompt="",
        input_image=start_image,
        end_image=end_image,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=worker_args.seed,
        tiled=True,
    )

    save_video(video, str(output_path), fps=worker_args.fps, quality=worker_args.quality)


def worker_process(
    gpu_id: int,
    task_queue: Queue,
    status_dict: dict,
    clip_base: Path,
    output_base: Path,
    worker_args: WorkerArgs,
    model_configs: List,
):
    """Worker process that processes clips on a specific GPU."""
    # Set GPU for this process BEFORE importing torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Import torch and pipeline AFTER setting CUDA_VISIBLE_DEVICES
    import torch
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline

    device = 'cuda'  # Always cuda since CUDA_VISIBLE_DEVICES is set

    # Random delay to avoid filesystem conflicts
    time.sleep(random.uniform(0, 1.0))

    # Initialize status for this GPU
    status_dict[gpu_id] = {
        'status': 'initializing',
        'clip': None,
        'completed': 0,
        'failed': 0,
        'start_time': None
    }

    try:
        # Load model
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map[worker_args.torch_dtype.lower()]

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
        )
        pipe.enable_vram_management()

        # Update status to idle
        status_dict[gpu_id] = {
            'status': 'idle',
            'clip': None,
            'completed': 0,
            'failed': 0,
            'start_time': None
        }

        # Process clips from queue
        while True:
            try:
                clip_id = task_queue.get(timeout=1)
            except:
                if task_queue.empty():
                    current = status_dict[gpu_id]
                    status_dict[gpu_id] = {
                        'status': 'finished',
                        'clip': None,
                        'completed': current['completed'],
                        'failed': current['failed'],
                        'start_time': None
                    }
                    break
                continue

            if clip_id is None:  # Poison pill
                current = status_dict[gpu_id]
                status_dict[gpu_id] = {
                    'status': 'finished',
                    'clip': None,
                    'completed': current['completed'],
                    'failed': current['failed'],
                    'start_time': None
                }
                break

            # Update status to processing
            current = status_dict[gpu_id]
            status_dict[gpu_id] = {
                'status': 'processing',
                'clip': str(clip_id),
                'completed': current['completed'],
                'failed': current['failed'],
                'start_time': time.time()
            }

            try:
                process_clip(clip_id, clip_base, output_base, pipe, worker_args)

                # Update status - success
                current = status_dict[gpu_id]
                status_dict[gpu_id] = {
                    'status': 'idle',
                    'clip': None,
                    'completed': current['completed'] + 1,
                    'failed': current['failed'],
                    'start_time': None
                }

                # Force cleanup
                import gc
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                # Update status - failed
                print(f"[GPU {gpu_id}] Error processing clip {clip_id}: {e}")
                import traceback
                traceback.print_exc()
                current = status_dict[gpu_id]
                status_dict[gpu_id] = {
                    'status': 'idle',
                    'clip': None,
                    'completed': current['completed'],
                    'failed': current['failed'] + 1,
                    'start_time': None
                }

    except Exception as e:
        # Model loading or other critical error
        print(f"[GPU {gpu_id}] CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        status_dict[gpu_id] = {
            'status': 'error',
            'clip': None,
            'completed': status_dict.get(gpu_id, {}).get('completed', 0),
            'failed': status_dict.get(gpu_id, {}).get('failed', 0),
            'start_time': None
        }


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds is None:
        return "N/A"
    return str(timedelta(seconds=int(seconds)))


def create_status_display(status_dict, total_clips, start_time, gpu_ids):
    """Create a rich layout for displaying multi-GPU status."""
    # Calculate overall statistics
    total_completed = sum(status_dict.get(gpu_id, {}).get('completed', 0) for gpu_id in gpu_ids)
    total_failed = sum(status_dict.get(gpu_id, {}).get('failed', 0) for gpu_id in gpu_ids)
    total_processed = total_completed + total_failed

    elapsed_time = time.time() - start_time

    # Calculate ETA
    if total_processed > 0:
        avg_time = elapsed_time / total_processed
        remaining = total_clips - total_processed
        eta = avg_time * remaining
    else:
        eta = None

    # Create summary panel
    summary_text = Text()
    summary_text.append("Progress: ", style="bold")
    summary_text.append(f"{total_processed}/{total_clips} ", style="bold cyan")
    summary_text.append("(", style="dim")
    summary_text.append(f"✓ {total_completed} ", style="bold green")
    summary_text.append(f"✗ {total_failed}", style="bold red")
    summary_text.append(")", style="dim")

    summary_text.append(" │ ", style="dim")
    summary_text.append("Elapsed: ", style="bold")
    summary_text.append(format_time(elapsed_time), style="yellow")

    if eta is not None:
        summary_text.append(" │ ", style="dim")
        summary_text.append("ETA: ", style="bold")
        summary_text.append(format_time(eta), style="yellow")

    summary_panel = Panel(
        summary_text,
        title="[bold magenta]Overall Progress[/bold magenta]",
        border_style="cyan",
        box=box.ROUNDED
    )

    # Create GPU status table
    gpu_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        padding=(0, 1)
    )
    gpu_table.add_column("GPU", style="cyan", width=6)
    gpu_table.add_column("Status", width=12)
    gpu_table.add_column("Current Clip", width=15)
    gpu_table.add_column("Completed", justify="right", width=10)
    gpu_table.add_column("Failed", justify="right", width=8)
    gpu_table.add_column("Processing Time", justify="right", width=16)

    for gpu_id in gpu_ids:
        info = status_dict.get(gpu_id, {})
        status = info.get('status', 'unknown')
        clip = info.get('clip')
        completed = info.get('completed', 0)
        failed = info.get('failed', 0)
        start_time_gpu = info.get('start_time')

        # Format status with colors
        if status == 'processing':
            status_str = "[bold yellow]Processing[/bold yellow]"
        elif status == 'idle':
            status_str = "[green]Idle[/green]"
        elif status == 'initializing':
            status_str = "[blue]Loading...[/blue]"
        elif status == 'finished':
            status_str = "[bold green]Finished[/bold green]"
        elif status == 'error':
            status_str = "[bold red]Error[/bold red]"
        else:
            status_str = "[dim]Unknown[/dim]"

        # Format clip name
        clip_str = clip if clip is not None else "-"

        # Format processing time
        if start_time_gpu is not None:
            proc_time = time.time() - start_time_gpu
            time_str = f"{proc_time:.1f}s"
        else:
            time_str = "-"

        gpu_table.add_row(
            f"GPU {gpu_id}",
            status_str,
            clip_str,
            f"[green]{completed}[/green]",
            f"[red]{failed}[/red]" if failed > 0 else "0",
            time_str
        )

    gpu_panel = Panel(
        gpu_table,
        title="[bold magenta]GPU Status[/bold magenta]",
        border_style="cyan",
        box=box.ROUNDED
    )

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(summary_panel, size=3),
        Layout(gpu_panel)
    )

    return layout


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU batch video frame interpolation")
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
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3'). If not specified, uses all available GPUs.")
    parser.add_argument("--prompt", type=str,
                        default="A car is driving on the road.",
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
    parser.add_argument("--torch-dtype", type=str, default="bfloat16",
                        help="Torch dtype (default: bfloat16)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Model configs (hardcoded)
    from diffsynth.pipelines.wan_video_new import ModelConfig
    model_configs = [
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
                    origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
                    origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
                    origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
    ]

    # Validate paths
    clip_base = args.clip_base.expanduser().resolve()
    output_base = args.output_base.expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    # Parse ranges
    parsed_ranges = parse_ranges(args.ranges) if args.ranges else None

    # Create worker args
    worker_args = WorkerArgs(
        prompt=args.prompt,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        num_intermediate_frames=args.num_intermediate_frames,
        seed=args.seed,
        fps=args.fps,
        quality=args.quality,
        torch_dtype=args.torch_dtype,
    )

    # Determine GPU IDs
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        import torch
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            console.print("[red]No GPUs available![/red]")
            return
        gpu_ids = list(range(gpu_count))

    # Print header
    console.print("\n[bold magenta]╔═══════════════════════════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║[/bold magenta]      [bold cyan]Wan Multi-GPU Frame Interpolation Pipeline[/bold cyan]      [bold magenta]║[/bold magenta]")
    console.print("[bold magenta]╚═══════════════════════════════════════════════════════════════╝[/bold magenta]\n")

    # Find pending clips
    console.print("[bold cyan]Scanning for clips to process...[/bold cyan]")
    skip_existing = not args.overwrite
    pending_clips = find_pending_clips(
        clip_base, output_base, ranges=parsed_ranges, skip_existing=skip_existing, descending=args.descending
    )

    if not pending_clips:
        console.print(f"[green]No clips to process. All clips already exist![/green]")
        return

    # Print info
    console.print(f"[bold]Found {len(pending_clips)} clips to process[/bold]")
    if parsed_ranges and parsed_ranges != [(None, None)]:
        ranges_str = ", ".join([f"[{s if s is not None else '0'}:{e if e is not None else '∞'})" for s, e in parsed_ranges])
        console.print(f"Range filter: {ranges_str}")
    if args.overwrite:
        console.print(f"Overwrite mode: [yellow]enabled (will reprocess existing clips)[/yellow]")
    else:
        console.print(f"Skip existing: [cyan]enabled (default)[/cyan]")

    order_str = "descending" if args.descending else "ascending"
    console.print(f"Processing order: [cyan]{order_str}[/cyan]")
    console.print(f"Clip base: [cyan]{clip_base}[/cyan]")
    console.print(f"Output: [cyan]{output_base}[/cyan]")
    console.print(f"Using GPUs: [cyan]{', '.join(map(str, gpu_ids))}[/cyan]\n")

    # Create shared structures
    manager = Manager()
    task_queue = manager.Queue()
    status_dict = manager.dict()

    # Populate task queue
    for clip_id in pending_clips:
        task_queue.put(clip_id)

    # Add poison pills
    for _ in gpu_ids:
        task_queue.put(None)

    # Start worker processes
    console.print(f"[bold cyan]Starting {len(gpu_ids)} worker processes...[/bold cyan]\n")

    import multiprocessing
    ctx = multiprocessing.get_context('spawn')

    workers = []
    for gpu_id in gpu_ids:
        p = ctx.Process(
            target=worker_process,
            args=(gpu_id, task_queue, status_dict, clip_base, output_base, worker_args, model_configs)
        )
        p.start()
        workers.append(p)

    # Monitor progress with live display
    start_time = time.time()

    with Live(
        create_status_display(status_dict, len(pending_clips), start_time, gpu_ids),
        refresh_per_second=2,
        console=console
    ) as live:
        while any(p.is_alive() for p in workers):
            live.update(create_status_display(status_dict, len(pending_clips), start_time, gpu_ids))
            time.sleep(0.5)

    # Wait for all workers to complete
    for p in workers:
        p.join()

    # Final summary
    total_time = time.time() - start_time
    total_completed = sum(status_dict.get(gpu_id, {}).get('completed', 0) for gpu_id in gpu_ids)
    total_failed = sum(status_dict.get(gpu_id, {}).get('failed', 0) for gpu_id in gpu_ids)

    console.print("\n")
    summary_table = Table(
        title="[bold magenta]Final Summary[/bold magenta]",
        box=box.DOUBLE,
        show_header=True,
        header_style="bold magenta"
    )
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", justify="right", style="green")

    summary_table.add_row("Total clips", str(len(pending_clips)))
    summary_table.add_row("Successful", f"[green]{total_completed}[/green]")
    summary_table.add_row("Failed", f"[red]{total_failed}[/red]" if total_failed > 0 else "0")
    summary_table.add_row("Success rate", f"{total_completed/(total_completed+total_failed)*100:.1f}%" if (total_completed+total_failed) > 0 else "N/A")
    summary_table.add_row("", "")
    summary_table.add_row("GPUs used", str(len(gpu_ids)))
    summary_table.add_row("Total time", format_time(total_time))
    summary_table.add_row("Avg per clip", f"{total_time/(total_completed+total_failed):.1f}s" if (total_completed+total_failed) > 0 else "N/A")

    console.print(summary_table)
    console.print(f"\n[bold green]✓ Multi-GPU batch processing completed![/bold green]")
    console.print(f"[cyan]Clips saved to: {output_base}[/cyan]\n")


if __name__ == "__main__":
    main()
