"""Zero-shot LS-VID evaluation with the MARS global+action model.

This evaluator deliberately does not train on LS-VID.  It transfers the
MARS-trained ResNet-50 and global+action projection head, represents every
LS-VID tracklet with 32 uniformly sampled global frames and 16 adaptive
bidirectional-exponential frames, and applies four rounds of signed Top-1
verification feedback.

LS-VID has no action-interval annotations.  The complete tracklet is therefore
used as the documented missing-interval fallback.  No static/text branch is
silently fabricated.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--resnet-checkpoint", required=True)
    parser.add_argument("--retriever-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--chunk-tracklets", type=int, default=128)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adaptive_base(interval_length: int, points_per_side: int = 8) -> float:
    if interval_length <= 2:
        return 1.0
    return max(1.0, (interval_length - 1) ** (1.0 / (points_per_side - 1)))


def bidirectional_indices(length: int) -> tuple[list[int], float]:
    if length <= 0:
        raise ValueError("Tracklet must contain at least one frame")
    base = adaptive_base(length)
    offsets = [min(int(round(base**power)), length - 1) for power in range(8)]
    forward = [min(offset, length - 1) for offset in offsets]
    reverse = [max(length - 1 - offset, 0) for offset in offsets]
    indices = forward + reverse
    if len(indices) != 16:
        raise AssertionError(len(indices))
    return indices, base


def uniform_indices(length: int) -> list[int]:
    if length <= 0:
        raise ValueError("Tracklet must contain at least one frame")
    return np.linspace(0, length - 1, 32).round().astype(int).tolist()


def load_split(path: Path) -> list[tuple[list[str], int, int, int]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = []
    for paths, pid, camid, trackid in data["tracklets"]:
        rows.append((list(paths), int(pid), int(camid), int(trackid)))
    if len(rows) != int(data["num_tracklets"]):
        raise RuntimeError("Split tracklet count mismatch: {}".format(path))
    return rows


class MarsResNetEncoder:
    def __init__(self, checkpoint_path: Path, device: torch.device):
        import torch.nn as nn
        import torchvision.models as models

        suite_src = checkpoint_path
        checkpoint = torch.load(suite_src, map_location="cpu")
        model = models.resnet50(weights=None)
        if int(checkpoint.get("last_stride", 1)) == 1:
            model.layer4[0].conv2.stride = (1, 1)
            model.layer4[0].downsample[0].stride = (1, 1)
        model.fc = nn.Identity()
        model.load_state_dict(checkpoint["backbone_state_dict"], strict=True)
        self.model = model.to(device).eval()
        self.device = device
        self.height = int(checkpoint.get("input_height", 256))
        self.width = int(checkpoint.get("input_width", 128))
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    @torch.inference_mode()
    def encode(self, images: list[Image.Image], batch_size: int) -> np.ndarray:
        outputs = []
        for offset in range(0, len(images), batch_size):
            selected = images[offset : offset + batch_size]
            buffer = np.empty(
                (len(selected), 3, self.height, self.width), dtype=np.float32
            )
            for index, image in enumerate(selected):
                array = np.asarray(
                    image.resize((self.width, self.height), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
                array = (array / 255.0 - self.mean) / self.std
                buffer[index] = array.transpose(2, 0, 1)
            batch = torch.from_numpy(buffer).to(self.device, non_blocking=True)
            outputs.append(self.model(batch).flatten(1).float().cpu())
        return torch.cat(outputs).numpy().astype(np.float32)


def extract_chunk(
    rows: list[tuple[list[str], int, int, int]],
    encoder: MarsResNetEncoder,
    frame_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    images: list[Image.Image] = []
    specs: list[tuple[list[int], list[int], int]] = []
    metadata: list[dict[str, object]] = []
    for paths, _, _, _ in rows:
        global_ids = uniform_indices(len(paths))
        action_ids, base = bidirectional_indices(len(paths))
        selected = global_ids + action_ids
        start = len(images)
        for index in selected:
            with Image.open(paths[index]) as image:
                images.append(image.convert("RGB"))
        specs.append((global_ids, action_ids, start))
        metadata.append(
            {
                "frames": len(paths),
                "adaptive_base": base,
                "global_indices": global_ids,
                "action_indices": action_ids,
            }
        )
    encoded = encoder.encode(images, frame_batch_size)
    descriptors = []
    for _, _, start in specs:
        global_feature = encoded[start : start + 32].mean(axis=0)
        action_feature = encoded[start + 32 : start + 48].mean(axis=0)
        descriptors.append(np.concatenate((global_feature, action_feature)))
    return (
        np.stack(descriptors).astype(np.float32),
        np.asarray([row[1] for row in rows], dtype=np.int64),
        np.asarray([row[2] for row in rows], dtype=np.int64),
        metadata,
    )


def extract_or_load(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    cache_path = output_dir / "global_action_features.pt"
    if cache_path.exists() and not args.force_extract:
        cached = torch.load(cache_path, map_location="cpu")
        expected_resnet = str(Path(args.resnet_checkpoint).resolve())
        if cached.get("resnet_checkpoint") != expected_resnet:
            raise RuntimeError("Cached feature extractor does not match requested checkpoint")
        print("Loaded {}".format(cache_path), flush=True)
        return cached

    query = load_split(Path(args.dataset_root) / "split_query.json")
    gallery = load_split(Path(args.dataset_root) / "split_gallery.json")
    if len(query) != 3504 or len(gallery) != 7829:
        raise RuntimeError("Expected official LS-VID 3504/7829 query/gallery split")
    rows = query + gallery
    encoder = MarsResNetEncoder(Path(args.resnet_checkpoint), device)
    chunk_dir = output_dir / "feature_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    chunks = []
    for start in range(0, len(rows), args.chunk_tracklets):
        end = min(start + args.chunk_tracklets, len(rows))
        path = chunk_dir / "{:05d}_{:05d}.npz".format(start, end)
        if path.exists() and not args.force_extract:
            payload = np.load(path)
            descriptors = payload["features"]
            pids = payload["pids"]
            camids = payload["camids"]
        else:
            descriptors, pids, camids, metadata = extract_chunk(
                rows[start:end], encoder, args.frame_batch_size
            )
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                features=descriptors,
                pids=pids,
                camids=camids,
                metadata=np.asarray([json.dumps(item) for item in metadata]),
            )
            temporary.replace(path)
        chunks.append((descriptors, pids, camids))
        print(
            "Feature extraction: {}/{} tracklets ({:.1f}s)".format(
                end, len(rows), time.monotonic() - started
            ),
            flush=True,
        )
    cached = {
        "features": torch.from_numpy(np.concatenate([item[0] for item in chunks])),
        "pids": torch.from_numpy(np.concatenate([item[1] for item in chunks])),
        "camids": torch.from_numpy(np.concatenate([item[2] for item in chunks])),
        "num_query": len(query),
        "dataset": "LS-VID",
        "feature_view": "global_action",
        "feature_dimension": 4096,
        "global_sampler": "32-frame uniform",
        "action_sampler": "16-frame adaptive bidirectional exponential",
        "missing_interval_fallback": "whole tracklet",
        "adaptive_base": "b_I=(L_I-1)^(1/7), no upper cap",
        "resnet_checkpoint": str(Path(args.resnet_checkpoint).resolve()),
        "extraction_seconds": time.monotonic() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    torch.save(cached, temporary)
    temporary.replace(cache_path)
    print("Saved {}".format(cache_path), flush=True)
    return cached


@torch.inference_mode()
def project_features(
    source: torch.Tensor, checkpoint_path: Path, suite_root: Path, device: torch.device
) -> torch.Tensor:
    sys.path.insert(0, str(suite_root))
    from vc_reid.retriever import load_retriever

    model = load_retriever(str(checkpoint_path), device).eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("feature_view") != "global_action" or int(
        checkpoint.get("input_dim", 0)
    ) != 4096:
        raise RuntimeError("Expected a 4096-D global_action projection checkpoint")
    outputs = []
    for offset in range(0, len(source), 512):
        feature, _ = model(source[offset : offset + 512].float().to(device))
        outputs.append(feature.cpu())
    return F.normalize(torch.cat(outputs), p=2, dim=1)


@torch.inference_mode()
def evaluate(
    query: torch.Tensor,
    gallery: torch.Tensor,
    q_pid: torch.Tensor,
    g_pid: torch.Tensor,
    junk: torch.Tensor,
) -> tuple[dict[str, float | int], torch.Tensor, torch.Tensor, torch.Tensor]:
    similarities = query @ gallery.T
    similarities.masked_fill_(junk, -float("inf"))
    order = similarities.argsort(dim=1, descending=True)
    matches = g_pid[order].eq(q_pid[:, None]) & ~junk.gather(1, order)
    cumulative = matches.cumsum(dim=1).float()
    ranks = torch.arange(1, gallery.shape[0] + 1, device=query.device)[None].float()
    relevant = matches.sum(dim=1)
    valid = relevant.gt(0)
    ap = ((cumulative / ranks) * matches).sum(dim=1) / relevant.clamp_min(1)
    metrics: dict[str, float | int] = {
        "mAP": 100.0 * float(ap[valid].mean()),
        "valid_queries": int(valid.sum()),
    }
    for rank in (1, 5, 10):
        metrics["Rank-{}".format(rank)] = 100.0 * float(
            matches[valid, :rank].any(dim=1).float().mean()
        )
    return metrics, order[:, 0], matches[:, 0], valid


@torch.inference_mode()
def run_interaction(
    query: torch.Tensor,
    gallery: torch.Tensor,
    q_pid: torch.Tensor,
    g_pid: torch.Tensor,
    junk: torch.Tensor,
    rounds: int,
    alpha: float,
    decay: float,
) -> dict[str, object]:
    current = query.clone()
    initial, _, initial_correct, valid = evaluate(current, gallery, q_pid, g_pid, junk)
    rows: list[dict[str, object]] = [{"round": 0, "metrics": initial}]
    final_correct = initial_correct
    for round_index in range(1, rounds + 1):
        _, top_index, oracle, _ = evaluate(current, gallery, q_pid, g_pid, junk)
        step = alpha * decay ** (round_index - 1)
        accepted = F.normalize(current + step * gallery[top_index], p=2, dim=1)
        rejected = F.normalize(current - step * gallery[top_index], p=2, dim=1)
        current = torch.where(oracle[:, None], accepted, rejected)
        metrics, _, final_correct, _ = evaluate(current, gallery, q_pid, g_pid, junk)
        rows.append({"round": round_index, "alpha": step, "metrics": metrics})
        print("Round {}: {}".format(round_index, metrics), flush=True)
    initially_wrong = valid & ~initial_correct
    initially_correct = valid & initial_correct
    repaired = initially_wrong & final_correct
    damaged = initially_correct & ~final_correct
    return {
        "rounds": rows,
        "repair_rate": 100.0 * float(repaired.sum()) / max(int(initially_wrong.sum()), 1),
        "damage_rate": 100.0 * float(damaged.sum()) / max(int(initially_correct.sum()), 1),
        "counts": {
            "initially_wrong": int(initially_wrong.sum()),
            "initially_correct": int(initially_correct.sum()),
            "repaired": int(repaired.sum()),
            "damaged": int(damaged.sum()),
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda:{}".format(args.device))
    cached = extract_or_load(args, device)
    projected = project_features(
        cached["features"],
        Path(args.retriever_checkpoint),
        Path(args.suite_root),
        device,
    ).to(device)
    pids = cached["pids"].long().to(device)
    camids = cached["camids"].long().to(device)
    num_query = int(cached["num_query"])
    query, gallery = projected[:num_query], projected[num_query:]
    q_pid, g_pid = pids[:num_query], pids[num_query:]
    q_cam, g_cam = camids[:num_query], camids[num_query:]
    junk = (q_pid[:, None] == g_pid[None]) & (q_cam[:, None] == g_cam[None])
    interaction = run_interaction(
        query, gallery, q_pid, g_pid, junk, args.rounds, args.alpha, args.decay
    )
    result = {
        "experiment": "LSVID_zero_shot_MARS_global_action_bidirectional",
        "protocol": {
            "query": len(query),
            "gallery": len(gallery),
            "gallery_excludes_query": True,
            "same_identity_same_camera_removed": True,
            "training_on_lsvid": False,
            "static_branch": False,
            "action_interval_source": "whole-tracklet missing-interval fallback",
            "verification": "identity labels simulate Top-1 user confirmation",
        },
        "configuration": vars(args),
        "feature_metadata": {
            key: value
            for key, value in cached.items()
            if key not in {"features", "pids", "camids"}
        },
        "interaction": interaction,
    }
    output_path = Path(args.output_dir) / "zero_shot_signed_feedback.json"
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    print("Wrote {}".format(output_path), flush=True)


if __name__ == "__main__":
    main()
