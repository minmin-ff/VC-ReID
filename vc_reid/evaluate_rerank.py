from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import atomic_json_dump
from .evaluate_manifest import embed_projection, embed_raw, load_manifest, select_available
from .rerank_official import evaluate_distances, k_reciprocal_rerank, squared_euclidean
from .retrieval_metrics import evaluate_with_details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an official MARS query/gallery split with k-reciprocal re-ranking"
    )
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--gallery-manifest", required=True)
    parser.add_argument("--view", default="full")
    parser.add_argument("--embedding", choices=("raw", "projection"), default="projection")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--k1", type=int, default=20)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--lambda-value", type=float, default=0.3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    query_rows = select_available(load_manifest(args.query_manifest), False, "query")
    gallery_rows = select_available(load_manifest(args.gallery_manifest), False, "gallery")
    if len(query_rows) != 1980 or len(gallery_rows) != 9330:
        raise RuntimeError(
            f"Refusing non-official MARS counts: {len(query_rows)} queries, "
            f"{len(gallery_rows)} gallery tracklets"
        )
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    if args.embedding == "projection":
        if not args.checkpoint:
            parser.error("--checkpoint is required for projection embeddings")
        query = embed_projection(
            query_rows, args.view, Path(args.checkpoint), device, args.batch_size
        )
        gallery = embed_projection(
            gallery_rows, args.view, Path(args.checkpoint), device, args.batch_size
        )
    else:
        query = embed_raw(query_rows, args.view)
        gallery = embed_raw(gallery_rows, args.view)

    query_pids = [str(row["pid"]) for row in query_rows]
    query_cameras = [str(row["camera"]) for row in query_rows]
    gallery_pids = [str(row["pid"]) for row in gallery_rows]
    gallery_cameras = [str(row["camera"]) for row in gallery_rows]
    query_ids = [f"official:{row['official_index']}" for row in query_rows]
    raw_metrics, raw_details = evaluate_with_details(
        query,
        query_pids,
        query_cameras,
        gallery,
        gallery_pids,
        gallery_cameras,
        query_ids,
    )

    gallery_gallery = squared_euclidean(gallery.to(device), gallery.to(device))
    distances = k_reciprocal_rerank(
        query,
        gallery,
        gallery_gallery,
        args.k1,
        args.k2,
        args.lambda_value,
        device,
    )
    rerank_metrics, rerank_details = evaluate_distances(
        distances, query_rows, gallery_rows
    )
    result = {
        "experiment": "official_mars_initial_k_reciprocal_reranking",
        "protocol": {
            "query_count": len(query_rows),
            "valid_queries": int(rerank_metrics["valid_queries"]),
            "gallery_count": len(gallery_rows),
        },
        "embedding": args.embedding,
        "view": args.view,
        "checkpoint": args.checkpoint,
        "configuration": {
            "k1": args.k1,
            "k2": args.k2,
            "lambda_value": args.lambda_value,
        },
        "raw": {"metrics": raw_metrics, "per_query": raw_details},
        "reranked": {"metrics": rerank_metrics, "per_query": rerank_details},
    }
    atomic_json_dump(result, args.output)
    print(json.dumps({"raw": raw_metrics, "reranked": rerank_metrics}, indent=2))


if __name__ == "__main__":
    main()
