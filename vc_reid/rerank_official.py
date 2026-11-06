from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import add_parent_to_path, atomic_json_dump, load_paths
from .interactive_official import corrected_feature_path, embed_rows, load_manifest
from .interactive_sweep import apply_update, static_anchor
from .retrieval_metrics import evaluate_with_details


@torch.inference_mode()
def encode_positive_qa(
    interactions: Dict[str, List[Dict[str, object]]],
    clip,
    tokenizer,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, torch.Tensor]:
    texts = sorted(
        {
            f"{record['question']} {record['answer']}"
            for records in interactions.values()
            for record in records
            if record.get("question_type") == "P"
        }
    )
    encoded: Dict[str, torch.Tensor] = {}
    for offset in tqdm(range(0, len(texts), batch_size), desc="positive QA CLIP"):
        chunk = texts[offset : offset + batch_size]
        tokens = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        features = F.normalize(clip.get_text_features(**tokens).float(), p=2, dim=1).cpu()
        encoded.update(zip(chunk, features))
    return encoded


def reconstruct_rounds(
    query_rows: Sequence[Dict[str, object]],
    gallery_rows: Sequence[Dict[str, object]],
    query_initial: torch.Tensor,
    gallery: torch.Tensor,
    feature_root: Path,
    interactions: Dict[str, List[Dict[str, object]]],
    qa_features: Dict[str, torch.Tensor],
    device: torch.device,
    anchor_weight: float,
) -> List[torch.Tensor]:
    gallery_by_official = {
        int(row["official_index"]): index for index, row in enumerate(gallery_rows)
    }
    rounds: List[List[torch.Tensor]] = [[] for _ in range(5)]
    for row, initial in tqdm(
        zip(query_rows, query_initial), total=len(query_rows), desc="reconstruct rounds"
    ):
        query_id = f"official:{row['official_index']}"
        records = interactions[query_id]
        if len(records) != 4:
            raise ValueError(f"{query_id} has {len(records)} interaction records")
        raw = np.load(corrected_feature_path(row, feature_root)).astype(np.float32)
        current = F.normalize(initial.to(device), p=2, dim=0)
        positive_sum = static_anchor(raw, device)
        positive = positive_sum.clone()
        rounds[0].append(current.cpu())
        for expected_round, record in enumerate(records, start=1):
            if int(record["round"]) != expected_round:
                raise ValueError(f"Non-contiguous interaction history for {query_id}")
            if record["question_type"] == "P":
                text = f"{record['question']} {record['answer']}"
                positive_sum = positive_sum + qa_features[text].to(device)
                positive = F.normalize(positive_sum, p=2, dim=0)
            gallery_index = gallery_by_official[int(record["top1_official_index"])]
            current = apply_update(
                current,
                gallery[gallery_index].to(device),
                positive,
                bool(record["simulated_user_relevance"]),
                "full",
                float(record["alpha"]),
                anchor_weight,
            )
            rounds[expected_round].append(current.cpu())
    return [torch.stack(values) for values in rounds]


def squared_euclidean(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    left = F.normalize(left.float(), p=2, dim=1)
    right = F.normalize(right.float(), p=2, dim=1)
    return (2.0 - 2.0 * torch.mm(left, right.T)).clamp_min_(0).cpu().numpy()


def k_reciprocal_rerank(
    query: torch.Tensor,
    gallery: torch.Tensor,
    gallery_gallery: np.ndarray,
    k1: int,
    k2: int,
    lambda_value: float,
    device: torch.device,
) -> np.ndarray:
    """Zhong et al. CVPR 2017 k-reciprocal encoding for squared-L2 distances."""
    query_gpu = query.to(device)
    gallery_gpu = gallery.to(device)
    q_g = squared_euclidean(query_gpu, gallery_gpu)
    q_q = squared_euclidean(query_gpu, query_gpu)
    del query_gpu, gallery_gpu
    torch.cuda.empty_cache()

    top = np.concatenate((q_q, q_g), axis=1)
    bottom = np.concatenate((q_g.T, gallery_gallery), axis=1)
    original = np.concatenate((top, bottom), axis=0).astype(np.float32, copy=False)
    column_max = np.maximum(original.max(axis=0), 1e-12)
    original = (original / column_max[None, :]).T
    all_num = original.shape[0]
    query_num = len(query)
    initial_rank = np.argsort(original, axis=1).astype(np.int32)
    v = np.zeros_like(original, dtype=np.float32)
    half_k = int(np.around(k1 / 2))

    for index in tqdm(range(all_num), desc="k-reciprocal encode"):
        forward = initial_rank[index, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        reciprocal = forward[np.where(backward == index)[0]]
        expanded = reciprocal
        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, : half_k + 1]
            candidate_backward = initial_rank[candidate_forward, : half_k + 1]
            candidate_reciprocal = candidate_forward[
                np.where(candidate_backward == candidate)[0]
            ]
            if len(np.intersect1d(candidate_reciprocal, reciprocal)) > (2.0 / 3.0) * len(
                candidate_reciprocal
            ):
                expanded = np.append(expanded, candidate_reciprocal)
        expanded = np.unique(expanded)
        weights = np.exp(-original[index, expanded])
        v[index, expanded] = weights / np.maximum(weights.sum(), 1e-12)

    if k2 > 1:
        v_qe = np.zeros_like(v, dtype=np.float32)
        for index in range(all_num):
            v_qe[index] = np.mean(v[initial_rank[index, :k2]], axis=0)
        v = v_qe

    inverted = [np.where(v[:, column] != 0)[0] for column in range(all_num)]
    jaccard = np.zeros((query_num, all_num), dtype=np.float32)
    for index in tqdm(range(query_num), desc="Jaccard distance"):
        non_zero = np.where(v[index] != 0)[0]
        related = np.unique(np.concatenate([inverted[column] for column in non_zero]))
        minimum = np.zeros(all_num, dtype=np.float32)
        for candidate in related:
            minimum[candidate] = np.minimum(v[index, non_zero], v[candidate, non_zero]).sum()
        jaccard[index] = 1.0 - minimum / (2.0 - minimum)
    final = jaccard[:, query_num:] * (1.0 - lambda_value) + original[
        :query_num, query_num:
    ] * lambda_value
    return final


def evaluate_distances(
    distances: np.ndarray,
    query_rows: Sequence[Dict[str, object]],
    gallery_rows: Sequence[Dict[str, object]],
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    gallery_pids = np.asarray([str(row["pid"]) for row in gallery_rows])
    gallery_cameras = np.asarray([str(row["camera"]) for row in gallery_rows])
    details = []
    for index, row in enumerate(query_rows):
        pid = str(row["pid"])
        camera = str(row["camera"])
        junk = (gallery_pids == pid) & (gallery_cameras == camera)
        positives = (gallery_pids == pid) & ~junk
        if not positives.any():
            continue
        order = np.argsort(distances[index])
        order = order[~junk[order]]
        positions = np.flatnonzero(positives[order])
        ap = float((np.arange(1, len(positions) + 1) / (positions + 1)).mean())
        first = int(positions[0] + 1)
        details.append(
            {
                "query_id": f"official:{row['official_index']}",
                "pid": pid,
                "camera": camera,
                "ap": ap,
                "first_positive_rank": first,
                "rank_1": int(first <= 1),
                "rank_5": int(first <= 5),
                "rank_10": int(first <= 10),
            }
        )
    metrics = {"valid_queries": len(details), "mAP": 100.0 * float(np.mean([x["ap"] for x in details]))}
    for rank in (1, 5, 10):
        metrics[f"Rank-{rank}"] = 100.0 * float(np.mean([x[f"rank_{rank}"] for x in details]))
    return metrics, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct and re-rank official MARS rounds")
    parser.add_argument("--config", default="configs/paths.json")
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--gallery-manifest", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--k1", type=int, default=20)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--lambda-value", type=float, default=0.3)
    parser.add_argument("--anchor-weight", type=float, default=0.3)
    args = parser.parse_args()

    paths = load_paths(args.config)
    add_parent_to_path(paths)
    from retriever import load_retriever
    from transformers import CLIPModel, CLIPTokenizer

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    query_rows = load_manifest(Path(args.query_manifest))
    gallery_rows = load_manifest(Path(args.gallery_manifest))
    if len(query_rows) != 1980 or len(gallery_rows) != 9330:
        raise RuntimeError("Refusing non-official MARS counts")
    feature_root = Path(args.feature_root)
    retriever = load_retriever(args.checkpoint, device).eval()
    query_initial = embed_rows(query_rows, feature_root, retriever, device)
    gallery = embed_rows(gallery_rows, feature_root, retriever, device)
    with open(args.interactions, "r", encoding="utf-8") as handle:
        source_result = json.load(handle)
    interactions = source_result["interactions"]
    clip = CLIPModel.from_pretrained(str(paths.clip_path), local_files_only=True).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(str(paths.clip_path), local_files_only=True)
    qa_features = encode_positive_qa(interactions, clip, tokenizer, device)
    rounds = reconstruct_rounds(
        query_rows,
        gallery_rows,
        query_initial,
        gallery,
        feature_root,
        interactions,
        qa_features,
        device,
        args.anchor_weight,
    )
    query_pids = [str(row["pid"]) for row in query_rows]
    query_cameras = [str(row["camera"]) for row in query_rows]
    gallery_pids = [str(row["pid"]) for row in gallery_rows]
    gallery_cameras = [str(row["camera"]) for row in gallery_rows]
    query_ids = [f"official:{row['official_index']}" for row in query_rows]
    raw_rounds = []
    for index, features in enumerate(rounds):
        metrics, _ = evaluate_with_details(
            features,
            query_pids,
            query_cameras,
            gallery,
            gallery_pids,
            gallery_cameras,
            query_ids,
        )
        expected = source_result["rounds"][index]["metrics"]
        if max(abs(float(metrics[key]) - float(expected[key])) for key in ("mAP", "Rank-1", "Rank-5", "Rank-10")) > 1e-4:
            raise RuntimeError(f"Round {index} reconstruction mismatch: {metrics} vs {expected}")
        raw_rounds.append({"round": index, "metrics": metrics})
    torch.save({"rounds": rounds, "gallery": gallery}, args.embedding_cache)

    gallery_gpu = gallery.to(device)
    gallery_gallery = squared_euclidean(gallery_gpu, gallery_gpu)
    del gallery_gpu
    torch.cuda.empty_cache()
    reranked = []
    for index, features in enumerate(rounds):
        distances = k_reciprocal_rerank(
            features,
            gallery,
            gallery_gallery,
            args.k1,
            args.k2,
            args.lambda_value,
            device,
        )
        metrics, details = evaluate_distances(distances, query_rows, gallery_rows)
        reranked.append({"round": index, "metrics": metrics, "per_query": details})
        print(index, metrics)
    atomic_json_dump(
        {
            "experiment": "official_mars_k_reciprocal_reranking",
            "protocol": {"query_count": 1980, "valid_queries": 1840, "gallery_count": 9330},
            "configuration": {"k1": args.k1, "k2": args.k2, "lambda_value": args.lambda_value},
            "raw_rounds": raw_rounds,
            "rounds": reranked,
            "source_interactions": args.interactions,
            "checkpoint": args.checkpoint,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
