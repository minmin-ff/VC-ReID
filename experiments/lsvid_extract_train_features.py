"""Extract leakage-free LS-VID training features for the visual projection head.

The script reads only ``split_train_dense_48.json``.  Each dense training
chunk is represented by a 2048-D 32-frame uniform descriptor and a 2048-D
16-frame adaptive bidirectional-exponential descriptor.  A zero-filled
512-D placeholder is appended solely to retain the experiment suite's
4608-D on-disk feature format; it is never used because the LS-VID projection
is trained with ``view=global_action`` and ``lambda_semantic=0``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lsvid_zero_shot_bidirectional import MarsResNetEncoder, extract_chunk, load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--resnet-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--chunk-tracklets", type=int, default=128)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def feature_path(root: Path, row_index: int, row: tuple[list[str], int, int, int]) -> Path:
    _, pid, camid, trackid = row
    return root / str(pid) / f"{camid:04d}_{trackid:06d}_{row_index:06d}.npy"


def atomic_save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, value)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    rows = load_split(Path(args.train_split))
    identities = {row[1] for row in rows}
    if len(rows) != 10106 or len(identities) != 842:
        raise RuntimeError(
            f"Expected 10106 dense train chunks and 842 identities, got "
            f"{len(rows)} and {len(identities)}"
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    encoder = MarsResNetEncoder(Path(args.resnet_checkpoint), device)
    started = time.monotonic()
    extracted = 0
    skipped = 0
    for start in range(0, len(rows), args.chunk_tracklets):
        end = min(start + args.chunk_tracklets, len(rows))
        chunk_rows = rows[start:end]
        targets = [
            feature_path(output_root, row_index, row)
            for row_index, row in enumerate(chunk_rows, start=start)
        ]
        if not args.force and all(path.exists() for path in targets):
            skipped += len(targets)
        else:
            descriptors, pids, camids, metadata = extract_chunk(
                chunk_rows, encoder, args.frame_batch_size
            )
            if descriptors.shape != (len(chunk_rows), 4096):
                raise RuntimeError(f"Unexpected descriptor shape {descriptors.shape}")
            for local_index, (path, descriptor) in enumerate(zip(targets, descriptors)):
                if path.exists() and not args.force:
                    skipped += 1
                    continue
                source = np.concatenate(
                    (descriptor.astype(np.float32, copy=False), np.zeros(512, np.float32))
                )
                atomic_save_array(path, source)
                extracted += 1
            metadata_dir = output_root / "_metadata"
            metadata_dir.mkdir(exist_ok=True)
            payload = {
                "start": start,
                "end": end,
                "pids": pids.tolist(),
                "camids": camids.tolist(),
                "sampling": metadata,
            }
            temporary = metadata_dir / f"{start:05d}_{end:05d}.tmp"
            final = metadata_dir / f"{start:05d}_{end:05d}.json"
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(final)
        print(
            f"Training extraction: {end}/{len(rows)} tracklets "
            f"(new={extracted}, skipped={skipped}, {time.monotonic() - started:.1f}s)",
            flush=True,
        )

    files = sorted(output_root.glob("*/*.npy"))
    if len(files) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} feature files, found {len(files)}")
    probes = [files[0], files[len(files) // 2], files[-1]]
    for path in probes:
        feature = np.load(path)
        if feature.shape != (4608,) or not np.isfinite(feature).all():
            raise RuntimeError(f"Invalid feature at {path}: {feature.shape}")
        if np.count_nonzero(feature[4096:]) != 0:
            raise RuntimeError(f"Nonzero static placeholder at {path}")
    config = {
        "dataset": "LS-VID",
        "split": str(Path(args.train_split).resolve()),
        "query_or_gallery_access": False,
        "training_tracklets": len(rows),
        "training_identities": len(identities),
        "resnet_checkpoint": str(Path(args.resnet_checkpoint).resolve()),
        "global_sampler": "32-frame uniform",
        "action_sampler": "16-frame adaptive bidirectional exponential",
        "missing_interval_fallback": "whole tracklet",
        "adaptive_base": "b_I=(L_I-1)^(1/7), no upper cap",
        "feature_dimension": 4608,
        "projection_view": "global_action",
        "static_placeholder": "512 zeros; excluded from training",
        "extraction_seconds": time.monotonic() - started,
    }
    (output_root / "extraction_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
