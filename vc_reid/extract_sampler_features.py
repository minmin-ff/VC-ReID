from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from .common import atomic_json_dump, load_paths, set_seed, stable_uniform


SAMPLERS = (
    "uniform",
    "random",
    "fixed_exponential",
    "adaptive_legacy",
    "adaptive_bidirectional",
    "boundary_jitter",
    "whole_tracklet_uniform",
)


def adaptive_base(interval_length: int, points_per_side: int = 8) -> float:
    """Paper-faithful base; it is not capped and can grow above two."""
    if interval_length <= 2:
        return 1.0
    return max(1.0, (interval_length - 1) ** (1.0 / (points_per_side - 1)))


def exponential_offsets(length: int, base: float, count: int = 8) -> List[int]:
    if length <= 1:
        return [0] * count
    return [min(int(round(base**power)), length - 1) for power in range(count)]


def sample_indices(
    start: int,
    end: int,
    sampler: str,
    key: str,
    seed: int,
) -> Tuple[List[int], float | None]:
    if end < start:
        start, end = end, start
    length = end - start + 1
    if sampler in {"uniform", "whole_tracklet_uniform"}:
        values = np.linspace(start, end, 16).round().astype(int).tolist()
        return values, None
    if sampler == "random":
        derived_seed = int(stable_uniform(key, seed) * (2**32 - 1))
        rng = np.random.default_rng(derived_seed)
        values = rng.choice(
            np.arange(start, end + 1), size=16, replace=length < 16
        )
        return sorted(int(value) for value in values), None
    if sampler == "fixed_exponential":
        base = 2.0
    elif sampler == "adaptive_legacy":
        base = min(adaptive_base(length), 2.0)
    elif sampler in {"adaptive_bidirectional", "boundary_jitter"}:
        base = adaptive_base(length)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
    offsets = exponential_offsets(length, base, 8)
    forward = [min(start + offset, end) for offset in offsets]
    reverse = [max(end - offset, start) for offset in offsets]
    values = forward + reverse
    if len(values) != 16:
        raise AssertionError(len(values))
    return values, base


def parse_tracklet(name: str) -> Tuple[str, str]:
    stem = Path(name).stem
    match = re.search(r"(cam\d+)_(track\d+)", stem)
    if not match:
        raise ValueError(f"Cannot parse camera/tracklet from {name}")
    return match.group(1), match.group(2)


def find_video(video_root: Path, pid: str, base_name: str) -> Path | None:
    candidates = [
        video_root / pid / f"{base_name}.mp4",
        video_root / pid / f"{pid}_{base_name}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted((video_root / pid).glob(f"*{base_name}*.mp4"))
    return matches[0] if matches else None


def collect_records(annotation_root: Path, video_root: Path) -> List[Dict[str, object]]:
    records: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for annotation_path in sorted(annotation_root.glob("**/*.json")):
        pid = annotation_path.parent.name
        try:
            camera, tracklet = parse_tracklet(annotation_path.name)
        except ValueError:
            continue
        base_name = f"{camera}_{tracklet}"
        key = (pid, camera, tracklet)
        if key not in records:
            video_path = find_video(video_root, pid, base_name)
            records[key] = {
                "pid": pid,
                "camera": camera,
                "tracklet": tracklet,
                "base_name": base_name,
                "video_path": str(video_path) if video_path else None,
                "actions": [],
            }
        with open(annotation_path, "r", encoding="utf-8") as handle:
            annotation = json.load(handle)
        for action_name, action in annotation.items():
            interval = action.get("time_interval")
            if action.get("is_present") and interval and len(interval) == 2:
                records[key]["actions"].append(
                    {
                        "name": action_name,
                        "start_seconds": float(interval[0]),
                        "end_seconds": float(interval[1]),
                    }
                )
    return list(records.values())


def video_information(path: str) -> Tuple[float, int]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}: fps={fps}, frames={frames}")
    return fps, frames


def read_frames(
    video_path: str,
    bbox_root: Path,
    pid: str,
    base_name: str,
    indices: Sequence[int],
) -> List[np.ndarray]:
    """Read all requested frames while opening the video at most once."""
    frames: Dict[int, np.ndarray] = {}
    missing: List[int] = []
    for index in sorted(set(indices)):
        bbox_candidates = [
            bbox_root / base_name / f"{index:06d}.jpg",
            bbox_root / f"{pid}_{base_name}" / f"{index:06d}.jpg",
        ]
        image = None
        for path in bbox_candidates:
            if path.exists():
                image = cv2.imread(str(path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    break
        if image is None:
            missing.append(index)
        else:
            frames[index] = image
    if missing:
        capture = cv2.VideoCapture(video_path)
        if capture.isOpened():
            for index in missing:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if ok:
                    frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        capture.release()
    output: List[np.ndarray] = []
    previous = None
    for index in indices:
        image = frames.get(index, previous)
        if image is None:
            return []
        output.append(image)
        previous = image
    return output


class ResNetEncoder:
    def __init__(self, device: torch.device):
        import torchvision.models as models

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    @torch.no_grad()
    def __call__(self, images: Sequence[np.ndarray]) -> np.ndarray:
        tensors = []
        for image in images:
            resized = np.asarray(Image.fromarray(image).resize((224, 224)))
            tensor = torch.from_numpy(resized.copy()).float().permute(2, 0, 1) / 255.0
            tensors.append((tensor - self.mean) / self.std)
        batch = torch.stack(tensors).to(self.device)
        features = self.model(batch).flatten(1)
        return features.mean(dim=0).cpu().numpy().astype(np.float32)


def jitter_interval(
    start: int,
    end: int,
    total_frames: int,
    fraction: float,
    key: str,
    seed: int,
) -> Tuple[int, int]:
    length = max(1, end - start + 1)
    maximum = max(1, int(round(length * fraction)))
    left = int(round((stable_uniform(key + ":left", seed) * 2 - 1) * maximum))
    right = int(round((stable_uniform(key + ":right", seed) * 2 - 1) * maximum))
    new_start = max(0, min(start + left, total_frames - 1))
    new_end = max(new_start, min(end + right, total_frames - 1))
    return new_start, new_end


def extract_split(args: argparse.Namespace, split: str) -> None:
    paths = load_paths(args.config)
    feature_root = paths.train_features if split == "train" else paths.test_features
    annotation_root = (
        paths.train_annotations if split == "train" else paths.test_annotations
    )
    video_root = paths.train_videos if split == "train" else paths.test_videos
    bbox_root = paths.train_bbox if split == "train" else paths.test_bbox
    output_root = paths.artifacts_root / "sampler_features" / args.sampler / split
    output_root.mkdir(parents=True, exist_ok=True)
    records = collect_records(annotation_root, video_root)
    if args.max_tracklets:
        records = records[: args.max_tracklets]
    if args.dry_run:
        print(
            f"dry-run split={split} sampler={args.sampler} records={len(records)} "
            f"output={output_root}"
        )
        for length in (16, 64, 129, 512):
            values, base = sample_indices(
                0, length - 1, args.sampler if args.sampler != "boundary_jitter" else "adaptive_bidirectional", "demo", args.seed
            )
            print(f"length={length} base={base} indices={values}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for feature extraction")
    encoder = ResNetEncoder(torch.device(f"cuda:{args.device}"))
    metadata: List[Dict[str, object]] = []
    for record in tqdm(records, desc=f"{split}:{args.sampler}"):
        video_path = record["video_path"]
        if not video_path:
            metadata.append({**record, "status": "missing_video"})
            continue
        source_path = feature_root / str(record["pid"]) / f"{record['base_name']}.npy"
        if not source_path.exists():
            metadata.append({**record, "status": "missing_source_feature"})
            continue
        destination = output_root / str(record["pid"]) / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not args.force:
            metadata.append({**record, "status": "skipped_existing"})
            continue
        try:
            fps, total_frames = video_information(str(video_path))
            action_features = []
            bases: List[float] = []
            intervals: List[Dict[str, object]] = []
            actions = list(record["actions"])
            if args.sampler == "whole_tracklet_uniform":
                actions = [{"name": "whole_tracklet", "start_seconds": 0.0, "end_seconds": (total_frames - 1) / fps}]
            for action_index, action in enumerate(actions):
                start = max(0, min(int(action["start_seconds"] * fps), total_frames - 1))
                end = max(start, min(int(action["end_seconds"] * fps), total_frames - 1))
                if args.sampler == "boundary_jitter":
                    start, end = jitter_interval(
                        start,
                        end,
                        total_frames,
                        args.jitter_fraction,
                        f"{record['pid']}/{record['base_name']}/{action_index}",
                        args.seed,
                    )
                indices, base = sample_indices(
                    start,
                    end,
                    args.sampler,
                    f"{record['pid']}/{record['base_name']}/{action_index}",
                    args.seed,
                )
                images = read_frames(
                    str(video_path),
                    bbox_root,
                    str(record["pid"]),
                    str(record["base_name"]),
                    indices,
                )
                if len(images) == 16:
                    action_features.append(encoder(images))
                    if base is not None:
                        bases.append(float(base))
                intervals.append(
                    {
                        "name": action["name"],
                        "start_frame": start,
                        "end_frame": end,
                        "length_frames": end - start + 1,
                        "indices": indices,
                        "base": base,
                    }
                )
            action_feature = (
                np.mean(action_features, axis=0).astype(np.float32)
                if action_features
                else np.zeros(2048, dtype=np.float32)
            )
            source = np.load(source_path).astype(np.float32).reshape(-1)
            if source.size != 4608:
                raise ValueError(f"Unexpected source feature shape {source.shape}")
            output = np.concatenate((source[:2048], action_feature, source[4096:4608]))
            np.save(destination, output)
            metadata.append(
                {
                    "pid": record["pid"],
                    "camera": record["camera"],
                    "tracklet": record["tracklet"],
                    "source": str(source_path),
                    "output": str(destination),
                    "sampler": args.sampler,
                    "mean_base": float(np.mean(bases)) if bases else None,
                    "intervals": intervals,
                    "status": "ok",
                }
            )
        except Exception as error:
            metadata.append({**record, "status": "error", "error": repr(error)})
    atomic_json_dump(
        {
            "sampler": args.sampler,
            "split": split,
            "seed": args.seed,
            "jitter_fraction": args.jitter_fraction,
            "paper_faithful_adaptive_base": "b=(L-1)^(1/7), no upper cap",
            "records": metadata,
        },
        paths.artifacts_root / "sampler_features" / args.sampler / f"{split}_meta.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild only the 2048-D action branch")
    parser.add_argument("--config", default="configs/paths.json")
    parser.add_argument("--sampler", choices=SAMPLERS, required=True)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--jitter-fraction", type=float, default=0.1)
    parser.add_argument("--max-tracklets", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    splits = ("train", "test") if args.split == "both" else (args.split,)
    for split in splits:
        extract_split(args, split)


if __name__ == "__main__":
    main()
