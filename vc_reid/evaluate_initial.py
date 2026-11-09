from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from .common import add_parent_to_path, atomic_json_dump, load_paths
from .feature_views import VIEW_DIMS, discover_feature_files, select_view
from .retrieval_metrics import evaluate_with_details


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an initial retrieval checkpoint")
    parser.add_argument("--config", default="configs/paths.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-root")
    parser.add_argument("--view", choices=sorted(VIEW_DIMS), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-queries", type=int)
    args = parser.parse_args()
    paths = load_paths(args.config)
    add_parent_to_path(paths)
    from .retriever import ProjectionHead

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    expected_dim = VIEW_DIMS[args.view]
    if int(checkpoint["input_dim"]) != expected_dim:
        raise ValueError(
            f"Checkpoint input_dim={checkpoint['input_dim']} but {args.view} is {expected_dim}"
        )
    model = ProjectionHead(
        int(checkpoint["input_dim"]),
        int(checkpoint.get("proj_dim", 512)),
        int(checkpoint["num_classes"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    feature_root = Path(args.feature_root) if args.feature_root else paths.test_features
    samples = discover_feature_files(feature_root)
    if args.max_queries:
        samples = samples[: args.max_queries]
    projected: List[torch.Tensor] = []
    with torch.no_grad():
        for offset in range(0, len(samples), args.batch_size):
            chunk = samples[offset : offset + args.batch_size]
            matrix = np.stack(
                [select_view(np.load(path), args.view) for path, _, _ in chunk]
            )
            features, _ = model(torch.from_numpy(matrix).to(device))
            projected.append(F.normalize(features, p=2, dim=1).cpu())
    embeddings = torch.cat(projected, dim=0)
    pids = [pid for _, pid, _ in samples]
    cameras = [camera for _, _, camera in samples]
    query_ids = [f"{pid}/{path.stem}" for path, pid, _ in samples]
    metrics, details = evaluate_with_details(
        embeddings,
        pids,
        cameras,
        embeddings,
        pids,
        cameras,
        query_ids,
    )
    output = {
        "experiment": "initial_retrieval",
        "feature_view": args.view,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "feature_root": str(feature_root),
        "metrics": metrics,
        "per_query": details,
    }
    atomic_json_dump(output, args.output)
    print(metrics)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
