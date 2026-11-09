"""Train the paper's ResNet-50 frame encoder on the official LS-VID train IDs.

Only ``split_train_dense_48.json`` is read.  Query/gallery files are never
opened by this trainer.  Each dense training tracklet contributes one randomly
sampled frame whenever selected by the deterministic P x K identity sampler.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--train-split", required=True)
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
    parser.add_argument("--resume")
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Run the forward/backward pass in FP32 for numerical stability.",
    )
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


class LSVIDTrackletFrames(Dataset):
    def __init__(self, split_path: Path):
        with split_path.open("r", encoding="utf-8") as handle:
            split = json.load(handle)
        self.rows = [
            (list(paths), int(pid), int(camid), int(trackid))
            for paths, pid, camid, trackid in split["tracklets"]
        ]
        if len(self.rows) != int(split["num_tracklets"]):
            raise RuntimeError("Training split count mismatch")
        identities = sorted({row[1] for row in self.rows})
        self.pid_to_label = {pid: index for index, pid in enumerate(identities)}
        self.indices_by_label: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.indices_by_label[self.pid_to_label[row[1]]].append(index)
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
        paths, pid, _, _ = self.rows[index]
        frame_path = paths[random.randrange(len(paths))]
        with Image.open(frame_path) as source:
            image = self.transform(source.convert("RGB"))
        return image, torch.tensor(self.pid_to_label[pid], dtype=torch.long)


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root).resolve()
    sys.path.insert(0, str(suite_root))
    from vc_reid.common import atomic_json_dump, set_seed
    from vc_reid.train_reid_resnet import (
        RandomIdentitySampler,
        ReIDResNet50,
        batch_hard_triplet,
        learning_rate,
        worker_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda:{}".format(args.device))
    dataset = LSVIDTrackletFrames(Path(args.train_split))
    if len(dataset) != 10106 or len(dataset.pid_to_label) != 842:
        raise RuntimeError(
            "Expected 10106 dense LS-VID train chunks and 842 train identities, "
            "got {} and {}".format(len(dataset), len(dataset.pid_to_label))
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
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.disable_amp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "dataset": "LS-VID",
        "training_tracklets": len(dataset),
        "training_identities": len(dataset.pid_to_label),
        "query_or_gallery_access": False,
        "initialization": "ImageNet-1K",
        "input_height": 256,
        "input_width": 128,
        "last_stride": 1,
        "loss": "cross_entropy(label_smoothing)+batch_hard_triplet",
    }
    atomic_json_dump(config, output_dir / "config.json")
    history = []
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        if int(state["num_classes"]) != len(dataset.pid_to_label):
            raise RuntimeError("Resume identity count mismatch")
        model.backbone.load_state_dict(state["backbone_state_dict"], strict=True)
        model.bnneck.load_state_dict(state["bnneck_state_dict"], strict=True)
        model.classifier.load_state_dict(state["classifier_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"])
        history_path = output_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print("Resumed after epoch {}".format(start_epoch), flush=True)

    for epoch in range(start_epoch, args.epochs):
        lr = learning_rate(epoch, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        sums: dict[str, float] = defaultdict(float)
        batches = 0
        progress = tqdm(loader, desc="LS-VID ResNet epoch {:03d}/{:03d}".format(epoch + 1, args.epochs))
        for batch_index, (images, labels) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=not args.disable_amp, dtype=torch.float16
            ):
                pooled, _, logits = model(images)
                id_loss = criterion(logits, labels)
                triplet = batch_hard_triplet(pooled, labels, args.margin)
                loss = id_loss + triplet
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss at epoch {} batch {}".format(
                        epoch + 1, batch_index + 1
                    )
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(loss)
            sums["id"] += float(id_loss)
            sums["triplet"] += float(triplet)
            batches += 1
            progress.set_postfix(loss="{:.4f}".format(sums["loss"] / batches))
            if args.max_batches is not None and batch_index + 1 >= args.max_batches:
                break
        row = {
            "epoch": epoch + 1,
            "learning_rate": lr,
            **{key: value / batches for key, value in sums.items()},
        }
        history.append(row)
        nonfinite_state = [
            name
            for name, value in model.state_dict().items()
            if torch.is_floating_point(value) and not torch.isfinite(value).all()
        ]
        if nonfinite_state:
            raise FloatingPointError(
                "Non-finite model tensors after epoch {}: {}".format(
                    epoch + 1, ", ".join(nonfinite_state[:20])
                )
            )
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
            torch.save(
                checkpoint,
                output_dir / "reid_resnet50_epoch{:03d}.pth".format(epoch + 1),
            )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
