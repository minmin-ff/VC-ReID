from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm

from .common import atomic_json_dump, set_seed
from .mars_protocol import read_names


def load_manifest(path: Path) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must be a list: {path}")
    return rows


def set_last_stride_one(model: nn.Module) -> None:
    model.layer4[0].conv2.stride = (1, 1)
    model.layer4[0].downsample[0].stride = (1, 1)


class MarsTrackletFrames(Dataset):
    def __init__(self, manifest: Path, names_file: Path, bbox_root: Path):
        self.rows = load_manifest(manifest)
        self.names = read_names(names_file)
        self.bbox_root = bbox_root
        identities = sorted({str(row["pid"]) for row in self.rows})
        self.pid_to_label = {pid: index for index, pid in enumerate(identities)}
        self.indices_by_label: Dict[int, List[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.indices_by_label[self.pid_to_label[str(row["pid"])]].append(index)
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 128), antialias=True),
                transforms.Pad(10),
                transforms.RandomCrop((256, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        start = int(row["start_frame_index"]) - 1
        end = int(row["end_frame_index"])
        if not 0 <= start < end <= len(self.names):
            raise ValueError(f"Invalid frame range for row {row['official_index']}")
        frame_index = random.randrange(start, end)
        name = self.names[frame_index]
        path = self.bbox_root / name[:4] / name
        image = self.transform(Image.open(path).convert("RGB"))
        label = self.pid_to_label[str(row["pid"])]
        return image, torch.tensor(label, dtype=torch.long)


class RandomIdentitySampler(Sampler[List[int]]):
    def __init__(self, dataset: MarsTrackletFrames, p: int, k: int, seed: int):
        self.dataset = dataset
        self.p = p
        self.k = k
        self.seed = seed
        self.epoch = 0
        self.batch_count = max(1, len(dataset) // (p * k))

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        labels = np.array(sorted(self.dataset.indices_by_label))
        for _ in range(self.batch_count):
            selected = rng.choice(labels, size=self.p, replace=False)
            batch = []
            for label in selected:
                candidates = self.dataset.indices_by_label[int(label)]
                batch.extend(
                    int(value)
                    for value in rng.choice(
                        candidates, size=self.k, replace=len(candidates) < self.k
                    )
                )
            rng.shuffle(batch)
            yield batch


class ReIDResNet50(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        set_last_stride_one(self.backbone)
        self.backbone.fc = nn.Identity()
        self.bnneck = nn.BatchNorm1d(2048)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, images: torch.Tensor):
        pooled = self.backbone(images)
        neck = self.bnneck(pooled)
        logits = self.classifier(neck)
        return pooled, neck, logits


def batch_hard_triplet(features: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    features = F.normalize(features.float(), p=2, dim=1)
    distance = torch.cdist(features, features, p=2)
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = ~labels[:, None].eq(labels[None, :])
    hardest_positive = distance.masked_fill(~same, -1.0).max(dim=1).values
    hardest_negative = distance.masked_fill(~different, float("inf")).min(dim=1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def learning_rate(epoch: int, base: float) -> float:
    if epoch < 10:
        return base * float(epoch + 1) / 10.0
    if epoch < 40:
        return base
    if epoch < 70:
        return base * 0.1
    return base * 0.01


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MARS ReID ResNet50 backbone")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--names-file", required=True)
    parser.add_argument("--bbox-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--identities-per-batch", type=int, default=16)
    parser.add_argument("--instances-per-identity", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.5e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume model, optimizer, scaler, and history from a last checkpoint",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional smoke-test cap per epoch; omit for the full experiment",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    dataset = MarsTrackletFrames(
        Path(args.manifest), Path(args.names_file), Path(args.bbox_root)
    )
    sampler = RandomIdentitySampler(
        dataset,
        args.identities_per_batch,
        args.instances_per_identity,
        args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed,
        generator=generator,
    )
    model = ReIDResNet50(len(dataset.pid_to_label)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "training_tracklets": len(dataset),
        "training_identities": len(dataset.pid_to_label),
        "input_height": 256,
        "input_width": 128,
        "last_stride": 1,
        "losses": "cross_entropy(label_smoothing=0.1)+batch_hard_triplet(margin=0.3)",
        "test_identity_access": False,
    }
    atomic_json_dump(config, output_dir / "config.json")
    history = []
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        state = torch.load(resume_path, map_location=device)
        if int(state["num_classes"]) != len(dataset.pid_to_label):
            raise ValueError("Resume checkpoint identity count does not match the manifest")
        model.backbone.load_state_dict(state["backbone_state_dict"], strict=True)
        model.bnneck.load_state_dict(state["bnneck_state_dict"], strict=True)
        model.classifier.load_state_dict(state["classifier_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"])
        history_path = output_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"Resumed {resume_path} after epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        lr = learning_rate(epoch, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        sums = defaultdict(float)
        count = 0
        progress = tqdm(loader, desc=f"ReID ResNet epoch {epoch + 1:03d}/{args.epochs}")
        for batch_index, (images, labels) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pooled, _, logits = model(images)
                id_loss = criterion(logits, labels)
                triplet = batch_hard_triplet(pooled, labels, args.margin)
                loss = id_loss + triplet
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(loss.item())
            sums["id"] += float(id_loss.item())
            sums["triplet"] += float(triplet.item())
            count += 1
            progress.set_postfix(loss=f"{sums['loss'] / count:.4f}")
            if args.max_batches is not None and batch_index + 1 >= args.max_batches:
                break
        row = {
            "epoch": epoch + 1,
            "learning_rate": lr,
            **{key: value / count for key, value in sums.items()},
        }
        history.append(row)
        atomic_json_dump(history, output_dir / "history.json")
        checkpoint = {
            "epoch": epoch + 1,
            "backbone_state_dict": model.backbone.state_dict(),
            "bnneck_state_dict": model.bnneck.state_dict(),
            "classifier_state_dict": model.classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "num_classes": len(dataset.pid_to_label),
            "input_height": 256,
            "input_width": 128,
            "last_stride": 1,
            "seed": args.seed,
            "training_config": config,
        }
        torch.save(checkpoint, output_dir / "reid_resnet50_last.pth")
        if epoch + 1 in {40, 70, 120}:
            torch.save(checkpoint, output_dir / f"reid_resnet50_epoch{epoch + 1:03d}.pth")
        print(json.dumps(row))


if __name__ == "__main__":
    main()
