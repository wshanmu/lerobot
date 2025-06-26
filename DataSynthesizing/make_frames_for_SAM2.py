#!/usr/bin/env python
"""
make_frames_for_SAM2.py

Usage:
    python make_frames_for_SAM2.py \
        --repo-id koenvanwijk/orange50-variation-2 \
        --camera phone           # second cam; omit to process every cam
        --quality 2              # ffmpeg -q:v <N> (0-9): 0=best, 9=worst
        --num-workers 8          # parallel extraction
        --root  ~/.cache/huggingface/lerobot   # optional custom cache
"""

import argparse, subprocess, multiprocessing as mp
from pathlib import Path
from typing import Iterator
from glob import glob

# --------------------------------------------------------------------------- #
# 1.  ── Load dataset and locate its local cache folder                       #
# --------------------------------------------------------------------------- #
def get_dataset_root(repo_id: str, root: str | None):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=root)        # downloads if missing
    return Path(ds.root)                           # e.g. ~/.cache/huggingface/lerobot/<repo_id>


# --------------------------------------------------------------------------- #
# 2.  ── Enumerate every MP4 that matches the camera filter                   #
# --------------------------------------------------------------------------- #

def find_videos(dataset_root: Path, camera_key: str | None) -> Iterator[Path]:
    """
    Yield every episode_<id>.mp4 inside   <root>/videos/chunk-XXX/observation.images.<camera>/.

    Parameters
    ----------
    dataset_root : Path
        e.g. ~/.cache/huggingface/lerobot/koenvanwijk/orange50-variation-2
    camera_key   : str | None
        * "phone", "laptop", …  to filter by camera
        * None                   to keep all cameras
    """
    videos_root = dataset_root / "videos"

    # 1) expand the wildcard *outside* Path construction
    for chunk_dir in videos_root.glob("chunk-*"):                    # chunk-000, chunk-001, …
        # 2) camera dirs look like observation.images.<camera>
        for cam_dir in chunk_dir.glob("observation.images.*"):
            if camera_key and not cam_dir.name.endswith(camera_key):
                continue                                             # skip other cameras
            yield from cam_dir.glob("episode_*.mp4")


# --------------------------------------------------------------------------- #
# 3.  ── Worker: run ffmpeg for one episode                                   #
# --------------------------------------------------------------------------- #
def _extract_worker(args):
    mp4_path, quality = args
    mp4_path = Path(mp4_path)
    out_dir  = mp4_path.parent / f"{mp4_path.stem}_frames"
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir                                 # already done

    out_dir.mkdir(exist_ok=True)
    # 5-digit zero-padded names keep them lexically sorted
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(mp4_path),
        "-q:v", str(quality),
        "-start_number", "0",
        str(out_dir / "%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return out_dir


# --------------------------------------------------------------------------- #
# 4.  ── CLI glue                                                             #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--camera",  default=None, help="camera key (e.g. phone)")
    ap.add_argument("--quality", default=2, type=int,
                    help="ffmpeg -q:v value (0-9, lower = better quality)")
    ap.add_argument("--num-workers", default=mp.cpu_count(), type=int)
    ap.add_argument("--root", default=None,
                    help="custom dataset cache dir (else LeRobot default)")
    args = ap.parse_args()

    root = get_dataset_root(args.repo_id, args.root)
    mp4s = list(find_videos(root, args.camera))
    if not mp4s:
        raise RuntimeError("No matching MP4 episodes found")

    print(f"Dataset path: {root}")
    print(f"Found {len(mp4s)} episode videos – extracting JPEG frames…")

    with mp.Pool(args.num_workers) as pool:
        frame_dirs = pool.map(_extract_worker,
                              [(p, args.quality) for p in mp4s])

    print("Finished. Created:", *frame_dirs[:3],
          "…" if len(frame_dirs) > 3 else "",
          sep="\n  • ")

if __name__ == "__main__":
    main()
