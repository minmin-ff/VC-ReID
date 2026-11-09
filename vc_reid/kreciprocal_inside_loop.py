#!/usr/bin/env python3
"""Evaluate k-reciprocal ranking inside the four-round feedback loop.

At every round the candidate shown for verification is the Rank-1 item under
k-reciprocal distance.  The signed update still changes only the 512-D query;
the gallery and encoder remain frozen.  This differs from outside-loop
re-ranking, which applies k-reciprocal only when reporting a trajectory created
from cosine-ranked feedback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import atomic_json_dump
from .interactive_official import load_manifest
from .rerank_official import (
    evaluate_distances,
    k_reciprocal_rerank,
    squared_euclidean,
)
from .retrieval_metrics import evaluate_with_details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--gallery-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--k1", type=int, default=20)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--lambda-value", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    cache = torch.load(args.embedding_cache, map_location="cpu")
    current = F.normalize(cache["rounds"][0].float(), p=2, dim=1)
    gallery = F.normalize(cache["gallery"].float(), p=2, dim=1)
    query_rows = load_manifest(args.query_manifest)
    gallery_rows = load_manifest(args.gallery_manifest)
    if current.shape != (1980, 512) or gallery.shape != (9330, 512):
        raise RuntimeError(f"Unexpected official MARS shapes: {current.shape}, {gallery.shape}")

    qpid = np.asarray([str(row["pid"]) for row in query_rows])
    qcam = np.asarray([str(row["camera"]) for row in query_rows])
    gpid = np.asarray([str(row["pid"]) for row in gallery_rows])
    gcam = np.asarray([str(row["camera"]) for row in gallery_rows])
    junk = (qpid[:, None] == gpid[None, :]) & (qcam[:, None] == gcam[None, :])

    gallery_gallery = squared_euclidean(gallery.to(device), gallery.to(device))
    torch.cuda.empty_cache()
    records = []
    for step in range(args.rounds + 1):
        raw, _ = evaluate_with_details(
            current,
            qpid.tolist(),
            qcam.tolist(),
            gallery,
            gpid.tolist(),
            gcam.tolist(),
            [f"official:{row['official_index']}" for row in query_rows],
        )
        distances = k_reciprocal_rerank(
            current,
            gallery,
            gallery_gallery,
            args.k1,
            args.k2,
            args.lambda_value,
            device,
        )
        reranked, details = evaluate_distances(distances, query_rows, gallery_rows)
        records.append({"round": step, "raw_cosine": raw, "reranked": reranked})
        print(step, reranked)
        if step == args.rounds:
            break

        candidate_distances = distances.copy()
        candidate_distances[junk] = np.inf
        top1 = candidate_distances.argmin(axis=1)
        correct = torch.from_numpy(gpid[top1] == qpid)
        direction = gallery[torch.from_numpy(top1)].clone()
        direction[~correct] *= -1.0
        alpha = args.alpha * (args.decay**step)
        current = F.normalize(current + alpha * direction, p=2, dim=1)

    atomic_json_dump(
        {
            "experiment": "k_reciprocal_inside_feedback_loop",
            "protocol": {
                "query_tracklets": 1980,
                "gallery_tracklets": 9330,
                "valid_queries": 1840,
                "feedback_bits_per_round": 1,
                "rounds": args.rounds,
            },
            "configuration": {
                "alpha": args.alpha,
                "decay": args.decay,
                "k1": args.k1,
                "k2": args.k2,
                "lambda_value": args.lambda_value,
            },
            "rounds": records,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
