"""Evaluate the LS-VID-trained global+action model on the official split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from lsvid_zero_shot_bidirectional import (
    evaluate,
    extract_or_load,
    project_features,
    run_interaction,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--resnet-checkpoint", required=True)
    parser.add_argument("--retriever-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--chunk-tracklets", type=int, default=128)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    cached = extract_or_load(args, device)
    if args.extract_only:
        print(
            f"Extraction complete: {len(cached['features'])} query/gallery tracklets",
            flush=True,
        )
        return
    if not args.retriever_checkpoint:
        raise ValueError("--retriever-checkpoint is required unless --extract-only is set")
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
        "experiment": "LSVID_supervised_global_action_bidirectional",
        "protocol": {
            "query": len(query),
            "gallery": len(gallery),
            "gallery_excludes_query": True,
            "same_identity_same_camera_removed": True,
            "training_on_lsvid_train_ids": True,
            "query_or_gallery_used_for_training": False,
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
    output_path = Path(args.output_dir) / "supervised_signed_feedback.json"
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
