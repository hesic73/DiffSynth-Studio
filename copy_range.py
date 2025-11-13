#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
from typing import Optional


def copy_directory_range(
    source_dir: Path,
    start: int,
    end: int,
    output_dir: Path,
    verbose: bool = False
) -> None:
    """
    Copy subdirectories from source_dir/start to source_dir/end (inclusive) to output_dir.

    Args:
        source_dir: Parent directory containing numbered subdirectories
        start: Start index (inclusive)
        end: End index (inclusive)
        output_dir: Destination directory
        verbose: Print progress information
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    if start > end:
        raise ValueError(f"Start index ({start}) must be <= end index ({end})")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    missing_count = 0

    for idx in range(start, end + 1):
        src_subdir = source_dir / str(idx)
        dst_subdir = output_dir / str(idx)

        if not src_subdir.exists():
            if verbose:
                print(f"Warning: {src_subdir} does not exist, skipping")
            missing_count += 1
            continue

        if not src_subdir.is_dir():
            if verbose:
                print(f"Warning: {src_subdir} is not a directory, skipping")
            missing_count += 1
            continue

        if dst_subdir.exists():
            if verbose:
                print(f"Destination {dst_subdir} already exists, skipping")
            continue

        if verbose:
            print(f"Copying {src_subdir} -> {dst_subdir}")

        shutil.copytree(src_subdir, dst_subdir)
        copied_count += 1

    print(f"\nCompleted: Copied {copied_count} directories")
    if missing_count > 0:
        print(f"Skipped {missing_count} missing or invalid entries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a range of numbered subdirectories to a new location"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Source directory containing numbered subdirectories"
    )
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start index (inclusive)"
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="End index (inclusive)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory where subdirectories will be copied"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = args.dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    copy_directory_range(
        source_dir=source_dir,
        start=args.start,
        end=args.end,
        output_dir=output_dir,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
