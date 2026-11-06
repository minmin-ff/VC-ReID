#!/usr/bin/env python3
"""Replay saved interactions under different semantic-anchor weights.

Questions, answers, verified candidates, initial embeddings, and the four-round
budget are held fixed.  Only the coefficient multiplying the accumulated
positive QA anchor changes.  This isolates the anchor coefficient without
regenerating language or changing the candidate trajectory exogenously.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiment_suite.src.common import add_parent_to_path, load_paths
from experiment_suite.src.interactive_official import load_manifest
from experiment_suite.src.rerank_official import encode_positive_qa, reconstruct_rounds
from experiment_suite.src.retrieval_metrics import evaluate_with_details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--gallery-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    args = parser.parse_args()

    paths = load_paths(args.config)
    add_parent_to_path(paths)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    query_rows = load_manifest(args.query_manifest)
    gallery_rows = load_manifest(args.gallery_manifest)
    cached = torch.load(args.embedding_cache, map_location="cpu")
    query_initial = cached["rounds"][0]
    gallery = cached["gallery"]
    source = json.loads(args.interactions.read_text(encoding="utf-8"))
    interactions = source["interactions"]

    from transformers import CLIPModel, CLIPTokenizer

    clip = CLIPModel.from_pretrained(str(paths.clip_path), local_files_only=True).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(str(paths.clip_path), local_files_only=True)
    qa_features = encode_positive_qa(interactions, clip, tokenizer, device)

    qpid = [str(row["pid"]) for row in query_rows]
    qcam = [str(row["camera"]) for row in query_rows]
    gpid = [str(row["pid"]) for row in gallery_rows]
    gcam = [str(row["camera"]) for row in gallery_rows]
    qids = [f"official:{row['official_index']}" for row in query_rows]
    rows = []
    for weight in (0.0, 0.1, 0.3, 0.5, 0.7, 1.0):
        rounds = reconstruct_rounds(
            query_rows,
            gallery_rows,
            query_initial,
            gallery,
            args.feature_root,
            interactions,
            qa_features,
            device,
            weight,
        )
        trajectory = []
        for index, state in enumerate(rounds):
            metric, _ = evaluate_with_details(
                state, qpid, qcam, gallery, gpid, gcam, qids
            )
            trajectory.append({"round": index, "metrics": metric})
        rows.append({"anchor_weight": weight, "rounds": trajectory})

    result = {
        "experiment": "semantic_anchor_weight_sensitivity",
        "source_interactions": str(args.interactions),
        "controlled_factors": {
            "questions_answers": "fixed",
            "verified_candidate_records": "fixed",
            "alpha": 0.15,
            "decay": 0.8,
            "rounds": 4,
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "weights": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
