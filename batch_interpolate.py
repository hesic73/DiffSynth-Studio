#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import Dict, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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


def parse_args():
    parser = argparse.ArgumentParser(description="Batch video frame interpolation")
    parser.add_argument("--clip-base", type=Path, required=True,
                        help="Base directory containing numbered clip subdirectories")
    parser.add_argument("--clip-start", type=int, required=True,
                        help="Start clip ID (inclusive)")
    parser.add_argument("--clip-end", type=int, required=True,
                        help="End clip ID (exclusive)")
    parser.add_argument("--output-base", type=Path, required=True,
                        help="Base output directory")
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

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.torch_dtype.lower()]

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=_MODEL_CONFIGS,
    )
    pipe.enable_vram_management()

    args.output_base.mkdir(parents=True, exist_ok=True)

    for clip_id in range(args.clip_start, args.clip_end):
        clip_folder = args.clip_base / str(clip_id)
        if not clip_folder.is_dir():
            logging.warning(f"Skipping: {clip_folder} not found")
            continue

        output_path = args.output_base / f"{clip_id}.mp4"
        logging.info(f"Processing clip {clip_id} -> {output_path}")

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
        logging.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
