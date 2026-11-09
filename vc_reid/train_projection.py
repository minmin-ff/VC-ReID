from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .common import add_parent_to_path, atomic_json_dump, load_paths, set_seed
from .feature_views import VIEW_DIMS, discover_feature_files, select_view


class ViewDataset(Dataset):
    def __init__(
        self,
        root: Path,
        view: str,
        allow_shared_static: bool = False,
        check_static_targets: bool = False,
    ):
        self.view = view
        raw = discover_feature_files(root)
        identities = sorted({pid for _, pid, _ in raw})
        self.pid_to_label = {pid: index for index, pid in enumerate(identities)}
        self.samples = [
            (path, self.pid_to_label[pid], pid) for path, pid, _ in raw
        ]
        self.indices_by_label: Dict[int, List[int]] = defaultdict(list)
        for index, (_, label, _) in enumerate(self.samples):
            self.indices_by_label[label].append(index)
        if "static" in view or view == "full" or check_static_targets:
            fingerprints: Dict[str, set[bytes]] = defaultdict(set)
            counts: Dict[str, int] = defaultdict(int)
            for path, _, pid in self.samples:
                feature = np.load(path).astype(np.float32).reshape(-1)
                if feature.size != 4608:
                    raise ValueError(f"Expected 4608-D feature at {path}")
                fingerprints[pid].add(feature[4096:4608].tobytes())
                counts[pid] += 1
            multi_tracklet = [pid for pid, count in counts.items() if count > 1]
            shared = [pid for pid in multi_tracklet if len(fingerprints[pid]) == 1]
            shared_ratio = len(shared) / max(1, len(multi_tracklet))
            if shared_ratio > 0.9 and not allow_shared_static:
                raise RuntimeError(
                    "Static-feature leakage guard failed: "
                    f"{len(shared)}/{len(multi_tracklet)} multi-tracklet identities "
                    "use one exactly identical static vector. Generate annotations "
                    "independently per tracklet. Use --allow-shared-static only for "
                    "a documented diagnostic, never a paper result."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, _ = self.samples[index]
        source = np.load(path).astype(np.float32).reshape(-1)
        feature = select_view(source, self.view)
        static_target = source[4096:4608]
        return (
            torch.from_numpy(feature),
            torch.tensor(label, dtype=torch.long),
            torch.from_numpy(static_target),
        )


class RandomIdentityBatchSampler(Sampler[List[int]]):
    """Deterministic P x K sampler with replacement for short identities."""

    def __init__(
        self,
        dataset: ViewDataset,
        identities_per_batch: int,
        instances_per_identity: int,
        seed: int,
    ):
        self.dataset = dataset
        self.p = identities_per_batch
        self.k = instances_per_identity
        self.seed = seed
        if self.p * self.k <= 0:
            raise ValueError("P and K must be positive")
        if self.p > len(dataset.indices_by_label):
            raise ValueError("P exceeds the number of training identities")
        self.batch_count = max(1, len(dataset) // (self.p * self.k))
        self.epoch = 0

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        labels = np.array(sorted(self.dataset.indices_by_label))
        self.epoch += 1
        for _ in range(self.batch_count):
            selected_labels = rng.choice(labels, size=self.p, replace=False)
            batch: List[int] = []
            for label in selected_labels:
                candidates = self.dataset.indices_by_label[int(label)]
                chosen = rng.choice(
                    candidates,
                    size=self.k,
                    replace=len(candidates) < self.k,
                )
                batch.extend(int(value) for value in chosen)
            rng.shuffle(batch)
            yield batch


def train(args: argparse.Namespace) -> Path:
    paths = load_paths(args.config)
    add_parent_to_path(paths)
    from .retriever import CenterLoss, CircleLoss, ProjectionHead, TripletLoss

    set_seed(args.seed)
    feature_root = Path(args.feature_root) if args.feature_root else paths.train_features
    dataset = ViewDataset(
        feature_root,
        args.view,
        allow_shared_static=args.allow_shared_static,
        check_static_targets=args.lambda_semantic > 0,
    )
    sampler = RandomIdentityBatchSampler(
        dataset,
        args.identities_per_batch,
        args.instances_per_identity,
        args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    input_dim = VIEW_DIMS[args.view]
    output_dir = (
        paths.artifacts_root
        / "projection"
        / args.artifact_name
        / args.view
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_record = vars(args).copy()
    config_record.update(
        {
            "input_dim": input_dim,
            "feature_root": str(feature_root),
            "training_samples": len(dataset),
            "num_classes": len(dataset.pid_to_label),
            "clip_aligned": args.lambda_semantic > 0,
        }
    )
    atomic_json_dump(config_record, output_dir / "config.json")
    if args.dry_run:
        features, labels, static_targets = next(iter(loader))
        model = ProjectionHead(input_dim, 512, len(dataset.pid_to_label))
        with torch.no_grad():
            embeddings, logits = model(features)
        print(
            f"dry-run ok view={args.view} batch={tuple(features.shape)} "
            f"embedding={tuple(embeddings.shape)} logits={tuple(logits.shape)}"
        )
        return output_dir

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; use --dry-run for CPU checks")
    device = torch.device(f"cuda:{args.device}")
    model = ProjectionHead(input_dim, 512, len(dataset.pid_to_label)).to(device)
    center_loss = CenterLoss(len(dataset.pid_to_label), 512, device)
    triplet_loss = TripletLoss(margin=0.3)
    circle_loss = CircleLoss(m=0.25, gamma=80.0)
    id_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(center_loss.parameters()),
        lr=args.learning_rate,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    history: List[Dict[str, float]] = []
    best_loss = math.inf
    for epoch in range(args.epochs):
        model.train()
        sums = defaultdict(float)
        batches = 0
        for features, labels, static_targets in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            static_targets = F.normalize(
                static_targets.to(device, non_blocking=True), p=2, dim=1
            )
            embeddings, logits = model(features)
            losses = {
                "triplet": triplet_loss(embeddings, labels),
                "id": id_loss(logits, labels),
                "center": center_loss(embeddings, labels),
                "circle": circle_loss(embeddings, labels),
                "semantic": (1.0 - F.cosine_similarity(
                    embeddings, static_targets, dim=1
                )).mean(),
            }
            total = (
                losses["triplet"]
                + losses["id"]
                + args.lambda_center * losses["center"]
                + args.lambda_circle * losses["circle"]
                + args.lambda_semantic * losses["semantic"]
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if args.lambda_center > 0:
                for parameter in center_loss.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(1.0 / args.lambda_center)
            optimizer.step()
            batches += 1
            sums["total"] += float(total.item())
            for name, value in losses.items():
                sums[name] += float(value.item())
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            **{name: value / batches for name, value in sums.items()},
        }
        history.append(row)
        atomic_json_dump(history, output_dir / "history.json")
        if row["total"] < best_loss:
            best_loss = row["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "proj_dim": 512,
                    "input_dim": input_dim,
                    "num_classes": len(dataset.pid_to_label),
                    "feature_view": args.view,
                    "seed": args.seed,
                    "training_config": config_record,
                    "clip_aligned": args.lambda_semantic > 0,
                    "lambda_semantic": args.lambda_semantic,
                },
                output_dir / "retriever_best.pth",
            )
        print(
            f"[{args.view} seed={args.seed}] epoch {epoch + 1:03d}/{args.epochs} "
            f"loss={row['total']:.5f}"
        )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a controlled projection ablation")
    parser.add_argument("--config", default="configs/paths.json")
    parser.add_argument("--view", choices=sorted(VIEW_DIMS), required=True)
    parser.add_argument("--feature-root")
    parser.add_argument("--artifact-name", default="default")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--identities-per-batch", type=int, default=32)
    parser.add_argument("--instances-per-identity", type=int, default=4)
    parser.add_argument("--lambda-center", type=float, default=5e-4)
    parser.add_argument("--lambda-circle", type=float, default=0.1)
    parser.add_argument(
        "--lambda-semantic",
        type=float,
        default=0.0,
        help=(
            "Cosine alignment weight between the 512-D retrieval embedding and "
            "the paired training-tracklet CLIP descriptor. It may also be used "
            "as a common distillation target for visual-only input-view "
            "ablations, and must be positive before direct CLIP-space feedback."
        ),
    )
    parser.add_argument(
        "--allow-shared-static",
        action="store_true",
        help="Bypass the leakage guard for diagnostics only.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
