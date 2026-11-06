"""Cross-dataset verification evaluation on the official PRID2011 release.

The initializer is the frozen MARS-trained TF-CLIP checkpoint.  No PRID2011
identity is used for training or model selection.  The protocol follows the
official TU Graz A-to-B evaluation: the first 200 camera-A trajectories are
queries and all 749 camera-B trajectories form the gallery.

This experiment tests whether signed Top-1 verification transfers across
datasets.  It does not claim a supervised PRID2011 comparison.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/vit_clipreid.yml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_data_root(root):
    root = Path(root).expanduser().resolve()
    candidates = [
        root / "prid_2011" / "multi_shot",
        root / "multi_shot",
    ]
    for candidate in candidates:
        if (candidate / "cam_a").is_dir() and (candidate / "cam_b").is_dir():
            return candidate
    raise FileNotFoundError("Cannot find prid_2011/multi_shot/cam_a and cam_b under {}".format(root))


def numeric_pid(path):
    digits = "".join(character for character in path.name if character.isdigit())
    if not digits:
        raise ValueError("No numeric identity in {}".format(path.name))
    return int(digits)


def collect_tracklets(root):
    data_root = resolve_data_root(root)
    cam_a = {numeric_pid(path): path for path in (data_root / "cam_a").iterdir() if path.is_dir()}
    cam_b = {numeric_pid(path): path for path in (data_root / "cam_b").iterdir() if path.is_dir()}
    required = set(range(1, 201))
    if not required.issubset(cam_a) or not required.issubset(cam_b):
        raise RuntimeError("The official 200 overlapping identities are incomplete")

    def frames(directory):
        result = tuple(sorted(str(path) for path in directory.glob("*.png")))
        if not result:
            result = tuple(sorted(str(path) for path in directory.glob("*.jpg")))
        if not result:
            raise RuntimeError("No frames in {}".format(directory))
        return result

    query = [(frames(cam_a[pid]), pid, 0, 0) for pid in range(1, 201)]
    gallery = [(frames(cam_b[pid]), pid, 1, 0) for pid in sorted(cam_b)]
    return query, gallery, data_root


def collate_one(batch):
    images, pids, camids, trackids, paths = zip(*batch)
    return (
        torch.stack(images, dim=0),
        tuple(int(value) for value in pids),
        tuple(int(value) for value in camids),
        torch.tensor(camids, dtype=torch.long),
        torch.ones(len(batch), dtype=torch.long),
        paths,
    )


@torch.inference_mode()
def extract_features(args, device):
    cache = Path(args.feature_cache)
    if cache.exists() and not args.force_extract:
        return torch.load(cache, map_location="cpu")

    from config import cfg
    from datasets.video_loader_xh import VideoDataset
    from model.make_model_clipreid import make_model
    import utils.seqtransforms as SeqT

    query, gallery, data_root = collect_tracklets(args.dataset_root)
    cfg.defrost()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    transform = SeqT.Compose(
        [
            SeqT.RectScale(cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]),
            SeqT.ToTensor(),
            SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = VideoDataset(query + gallery, seq_len=args.seq_len, sample="dense", transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_one,
    )

    # The released checkpoint was trained on the official MARS split.
    model = make_model(cfg, num_class=625, camera_num=6, view_num=1)
    model.load_param(args.checkpoint)
    model = model.to(device).eval()
    features, pids, camids = [], [], []
    started = time.monotonic()
    for index, (images, pid, camid, camera_labels, view_labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        batch, clips, sequence, channels, height, width = images.shape
        if batch != 1:
            raise RuntimeError("Dense PRID2011 extraction requires batch size one")
        images = images.view(clips, sequence, channels, height, width)
        camera_labels = camera_labels.repeat(clips).to(device) if cfg.MODEL.SIE_CAMERA else None
        view_labels = view_labels.repeat(clips).to(device) if cfg.MODEL.SIE_VIEW else None
        feature = model(images, cam_label=camera_labels, view_label=view_labels)
        features.append(feature.float().mean(dim=0, keepdim=True).cpu())
        pids.append(pid[0])
        camids.append(camid[0])
        if (index + 1) % 100 == 0:
            print("Extracted {}/{} tracklets".format(index + 1, len(loader)), flush=True)

    value = {
        "features": F.normalize(torch.cat(features, dim=0), p=2, dim=1),
        "pids": torch.tensor(pids, dtype=torch.long),
        "camids": torch.tensor(camids, dtype=torch.long),
        "num_query": len(query),
        "data_root": str(data_root),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "extraction_seconds": time.monotonic() - started,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(cache)
    return value


@torch.inference_mode()
def evaluate(query, gallery, query_pid, gallery_pid):
    scores = query @ gallery.t()
    order = scores.argsort(dim=1, descending=True)
    matches = gallery_pid[order].eq(query_pid[:, None])
    ranks = torch.arange(1, gallery.shape[0] + 1, device=query.device).float()[None, :]
    cumulative = matches.cumsum(dim=1).float()
    relevant = matches.sum(dim=1)
    valid = relevant.gt(0)
    average_precision = ((cumulative / ranks) * matches).sum(dim=1) / relevant.clamp_min(1)
    metrics = {
        "mAP": 100.0 * average_precision[valid].mean().item(),
        "Rank-1": 100.0 * matches[valid, :1].any(dim=1).float().mean().item(),
        "Rank-5": 100.0 * matches[valid, :5].any(dim=1).float().mean().item(),
        "Rank-10": 100.0 * matches[valid, :10].any(dim=1).float().mean().item(),
        "valid_queries": int(valid.sum().item()),
    }
    return metrics, order[:, 0], matches[:, 0], valid


@torch.inference_mode()
def run(args, cached, device):
    features = F.normalize(cached["features"].float(), p=2, dim=1).to(device)
    pids = cached["pids"].long().to(device)
    boundary = int(cached["num_query"])
    query, gallery = features[:boundary], features[boundary:]
    query_pid, gallery_pid = pids[:boundary], pids[boundary:]

    current = query.clone()
    initial_metrics, _, initial_correct, valid = evaluate(current, gallery, query_pid, gallery_pid)
    first_success = torch.full((len(query),), -1, dtype=torch.long, device=device)
    first_success[initial_correct & valid] = 0
    rows = [{"round": 0, "metrics": initial_metrics}]
    for round_index in range(1, args.rounds + 1):
        _, top_index, correct, _ = evaluate(current, gallery, query_pid, gallery_pid)
        step = args.alpha * args.decay ** (round_index - 1)
        top_feature = gallery[top_index]
        accepted = F.normalize(current + step * top_feature, p=2, dim=1)
        rejected = F.normalize(current - step * top_feature, p=2, dim=1)
        current = torch.where(correct[:, None], accepted, rejected)
        metrics, _, correct_after, _ = evaluate(current, gallery, query_pid, gallery_pid)
        newly_successful = first_success.lt(0) & correct_after & valid
        first_success[newly_successful] = round_index
        rows.append({"round": round_index, "alpha": step, "metrics": metrics})
        print(
            "Round {}: mAP={:.2f} R1={:.2f} R5={:.2f} R10={:.2f}".format(
                round_index,
                metrics["mAP"],
                metrics["Rank-1"],
                metrics["Rank-5"],
                metrics["Rank-10"],
            ),
            flush=True,
        )

    final_metrics, _, final_correct, _ = evaluate(current, gallery, query_pid, gallery_pid)
    initially_wrong = valid & ~initial_correct
    initially_correct = valid & initial_correct
    repaired = initially_wrong & final_correct
    damaged = initially_correct & ~final_correct
    repair_steps = first_success[repaired]
    return {
        "experiment": "MARS-trained TF-CLIP to PRID2011, no adaptation",
        "protocol": "official TU Graz A-to-B: 200 queries, 749 gallery trajectories",
        "rounds": rows,
        "Repair_rate": 100.0 * repaired.sum().item() / initially_wrong.sum().item(),
        "Damage_rate": 100.0 * damaged.sum().item() / initially_correct.sum().item(),
        "Mean_rounds_to_success_initial_failures": (
            repair_steps.float().mean().item() if repair_steps.numel() else None
        ),
        "Rank1_gain_per_fixed_click": (
            final_metrics["Rank-1"] - initial_metrics["Rank-1"]
        ) / args.rounds,
        "counts": {
            "initially_wrong": int(initially_wrong.sum().item()),
            "initially_correct": int(initially_correct.sum().item()),
            "repaired": int(repaired.sum().item()),
            "damaged": int(damaged.sum().item()),
        },
        "feature_cache": {
            key: value
            for key, value in cached.items()
            if key not in {"features", "pids", "camids"}
        },
        "settings": vars(args),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:{}".format(args.device))
    cached = extract_features(args, device)
    result = run(args, cached, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    print("Saved {}".format(output), flush=True)


if __name__ == "__main__":
    main()
