"""Four-question positive/negative ratio ablation with signed feedback.

This is a standalone evaluation entry point: it does not modify ``pipeline.py``.
Every schedule asks exactly one question in each of four rounds and uses the
same paper-defined update rule::

    correct top-1: normalize(q + alpha_t * top1)
    wrong top-1:   normalize(q - alpha_t * top1
                             + 0.3 * alpha_t * positive_evidence)

Positive evidence is the normalized accumulation of the initial static CLIP
branch and all positive question-answer embeddings observed so far.
"""

import argparse
import glob
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluator import InteractiveEvaluator, evaluate_retrieval
from pipeline import PipelineConfig, encode_clip_text, load_all_models
from questioner_answerer import (
    find_video_path,
    generate_answer,
    generate_negative_question,
    generate_positive_question,
)
from retriever import build_gallery, retrieve_topk


QUESTION_SCHEDULES: Dict[str, Tuple[str, ...]] = {
    "0:4": ("negative", "negative", "negative", "negative"),
    "1:3": ("negative", "positive", "negative", "negative"),
    "2:2": ("positive", "negative", "positive", "negative"),
    "3:1": ("positive", "negative", "positive", "positive"),
    "4:0": ("positive", "positive", "positive", "positive"),
}


def _validate_schedules() -> None:
    expected = {
        "0:4": (0, 4),
        "1:3": (1, 3),
        "2:2": (2, 2),
        "3:1": (3, 1),
        "4:0": (4, 0),
    }
    for ratio, schedule in QUESTION_SCHEDULES.items():
        counts = (schedule.count("positive"), schedule.count("negative"))
        if len(schedule) != 4 or counts != expected[ratio]:
            raise ValueError(f"Invalid schedule for {ratio}: {schedule}")


_validate_schedules()


def _normalize_feature(feature: torch.Tensor) -> torch.Tensor:
    if feature.norm().item() <= 1e-6:
        raise ValueError("Semantic evidence must not be the zero vector")
    return F.normalize(feature, p=2, dim=0)


def _static_clip_feature(
    raw_feature: np.ndarray, device: torch.device
) -> torch.Tensor:
    flat = raw_feature.reshape(-1)
    if flat.shape[0] < 512:
        raise ValueError(f"Expected at least 512 dimensions, got {flat.shape[0]}")
    return _normalize_feature(
        torch.tensor(flat[-512:], dtype=torch.float32, device=device)
    )


def run_single_query(
    query_feat_raw: np.ndarray,
    query_pid: str,
    query_cam: str,
    query_tracklet: str,
    gallery_feats_proj: torch.Tensor,
    gallery_pids: List[str],
    gallery_cams: List[str],
    gallery_paths: List[str],
    retriever,
    qwen_model,
    qwen_processor,
    clip_model,
    clip_tokenizer,
    config: PipelineConfig,
    device: torch.device,
    schedule: Sequence[str],
) -> List[Tuple[int, torch.Tensor]]:
    query_tensor = torch.tensor(
        query_feat_raw.reshape(-1), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        query_proj, _ = retriever(query_tensor.unsqueeze(0))
    query_proj = F.normalize(query_proj.squeeze(0), p=2, dim=0)
    round_features = [(0, query_proj.cpu().clone())]

    query_video_path = find_video_path(
        config.TEST_VIDEO_ROOT, query_pid, query_tracklet
    )
    if not query_video_path:
        for round_idx in range(1, config.MAX_ROUNDS + 1):
            round_features.append((round_idx, query_proj.cpu().clone()))
        return round_features

    # The stored 512-D static branch is the initial-description CLIP anchor.
    positive_evidence_sum = _static_clip_feature(query_feat_raw, device)
    positive_evidence = positive_evidence_sum.clone()
    qa_history = ""

    for round_idx, question_type in enumerate(schedule, start=1):
        _, _, _, top_paths = retrieve_topk(
            query_proj.cpu(),
            gallery_feats_proj,
            gallery_pids,
            gallery_cams,
            gallery_paths,
            query_pid,
            query_cam,
            topk=config.TOPK_DISPLAY,
        )

        top_video_paths = []
        for gallery_path in top_paths:
            gallery_idx = gallery_paths.index(gallery_path)
            gallery_pid = gallery_pids[gallery_idx]
            gallery_tracklet = os.path.splitext(os.path.basename(gallery_path))[0]
            video_path = find_video_path(
                config.TEST_VIDEO_ROOT, gallery_pid, gallery_tracklet
            )
            if video_path:
                top_video_paths.append(video_path)
        if not top_video_paths:
            round_features.append((round_idx, query_proj.cpu().clone()))
            continue

        top1_path = top_paths[0]
        top1_raw = np.load(top1_path).astype(np.float32)
        top1_tensor = torch.tensor(
            top1_raw.reshape(-1), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            top1_proj, _ = retriever(top1_tensor.unsqueeze(0))
        top1_proj = F.normalize(top1_proj.squeeze(0), p=2, dim=0)

        generator = (
            generate_positive_question
            if question_type == "positive"
            else generate_negative_question
        )
        question = generator(
            qwen_model,
            qwen_processor,
            top_video_paths,
            qa_history,
            device,
        )
        answer = generate_answer(
            qwen_model,
            qwen_processor,
            query_video_path,
            question,
            device,
        )
        qa_feature = _normalize_feature(
            encode_clip_text(
                clip_model,
                clip_tokenizer,
                [f"{question} {answer}"],
                device,
            )
        )

        if question_type == "positive":
            positive_evidence_sum = positive_evidence_sum + qa_feature
            positive_evidence = _normalize_feature(positive_evidence_sum)

        top1_pid = gallery_pids[gallery_paths.index(top1_path)]
        top1_is_correct = top1_pid == query_pid
        alpha_t = config.ALPHA * (0.8 ** (round_idx - 1))
        if top1_is_correct:
            query_proj = F.normalize(
                query_proj + alpha_t * top1_proj,
                p=2,
                dim=0,
            )
        else:
            query_proj = F.normalize(
                query_proj
                - alpha_t * top1_proj
                + 0.3 * alpha_t * positive_evidence,
                p=2,
                dim=0,
            )

        qa_history += (
            f"\n[Round {round_idx}] "
            f"Q({question_type}): {question} | A: {answer}"
        )
        round_features.append((round_idx, query_proj.cpu().clone()))

    return round_features


def select_initial_misses(
    all_queries: List[Dict[str, str]],
    gallery_feats: torch.Tensor,
    gallery_pids: List[str],
    gallery_cams: List[str],
    gallery_paths: List[str],
    retriever,
    device: torch.device,
    limit: int | None,
) -> List[Dict[str, str]]:
    """Select queries whose initial retrieval has an incorrect Top-1."""
    selected = []
    for query in tqdm(all_queries, desc="Selecting initial misses"):
        raw_feature = np.load(query["feature_path"]).astype(np.float32)
        query_tensor = torch.tensor(
            raw_feature.reshape(-1), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            query_proj, _ = retriever(query_tensor.unsqueeze(0))
        query_proj = F.normalize(query_proj.squeeze(0), p=2, dim=0)
        _, _, top_pids, _ = retrieve_topk(
            query_proj.cpu(),
            gallery_feats,
            gallery_pids,
            gallery_cams,
            gallery_paths,
            query["pid"],
            query["cam"],
            topk=1,
        )
        if top_pids and top_pids[0] != query["pid"]:
            selected.append(query)
            if limit is not None and len(selected) >= limit:
                break
    return selected


def run_evaluation(
    ratio: str,
    device_id: int,
    max_queries: int | None,
    output_tag: str | None,
    initial_misses_only: bool,
) -> List[Dict[str, float]]:
    config = PipelineConfig()
    # Skip the stale learned UpdateNet.  This script performs the signed update
    # directly and uses ABLATION_MODE only to control shared model loading.
    config.ABLATION_MODE = "no_negative"
    config.DEVICE_ID = device_id
    device = torch.device(f"cuda:{device_id}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    (
        qwen_model,
        qwen_processor,
        clip_model,
        clip_tokenizer,
        resnet,
        resnet_transform,
        retriever,
        _update_net,
    ) = load_all_models(config, device)
    del _update_net, resnet, resnet_transform

    if config.MAX_ROUNDS != len(QUESTION_SCHEDULES[ratio]):
        raise ValueError(
            f"MAX_ROUNDS={config.MAX_ROUNDS} does not match the four-question schedule"
        )

    gallery_feats, gallery_pids, gallery_cams, gallery_paths = build_gallery(
        config.TEST_FEAT_ROOT, retriever, device
    )
    all_queries = []
    for feature_path in sorted(
        glob.glob(os.path.join(config.TEST_FEAT_ROOT, "*", "*.npy"))
    ):
        pid = os.path.basename(os.path.dirname(feature_path))
        tracklet = os.path.splitext(os.path.basename(feature_path))[0]
        all_queries.append(
            {
                "pid": pid,
                "cam": tracklet.split("_")[0],
                "tracklet": tracklet,
                "feature_path": feature_path,
            }
        )
    if initial_misses_only:
        all_queries = select_initial_misses(
            all_queries,
            gallery_feats,
            gallery_pids,
            gallery_cams,
            gallery_paths,
            retriever,
            device,
            max_queries,
        )
    elif max_queries is not None:
        all_queries = all_queries[:max_queries]

    print(f"[RatioAblation] ratio={ratio}")
    print(f"[RatioAblation] schedule={QUESTION_SCHEDULES[ratio]}")
    print(f"[RatioAblation] queries={len(all_queries)}, device=cuda:{device_id}")
    print(f"[RatioAblation] initial_misses_only={initial_misses_only}")

    round_query_features = {
        round_idx: [] for round_idx in range(config.MAX_ROUNDS + 1)
    }
    query_pids = []
    query_cams = []
    for query in tqdm(all_queries, desc=f"Ratio {ratio}"):
        raw_feature = np.load(query["feature_path"]).astype(np.float32)
        features = run_single_query(
            raw_feature,
            query["pid"],
            query["cam"],
            query["tracklet"],
            gallery_feats,
            gallery_pids,
            gallery_cams,
            gallery_paths,
            retriever,
            qwen_model,
            qwen_processor,
            clip_model,
            clip_tokenizer,
            config,
            device,
            QUESTION_SCHEDULES[ratio],
        )
        query_pids.append(query["pid"])
        query_cams.append(query["cam"])
        for round_idx, feature in features:
            round_query_features[round_idx].append(feature)

    evaluator = InteractiveEvaluator(max_rounds=config.MAX_ROUNDS)
    for round_idx in range(config.MAX_ROUNDS + 1):
        metrics = evaluate_retrieval(
            torch.stack(round_query_features[round_idx], dim=0),
            query_pids,
            query_cams,
            gallery_feats,
            gallery_pids,
            gallery_cams,
        )
        evaluator.record(round_idx, metrics)
    evaluator.print_summary()

    ratio_tag = ratio.replace(":", "_")
    output_name = f"interactive_results_ratio_signed_{ratio_tag}"
    if output_tag:
        output_name += f"_{output_tag}"
    output_path = os.path.join(config.OUTPUT_DIR, output_name + ".json")
    evaluator.save(output_path)
    return evaluator.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", required=True, choices=sorted(QUESTION_SCHEDULES))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--output-tag")
    parser.add_argument("--initial-misses-only", action="store_true")
    args = parser.parse_args()
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries must be positive")
    run_evaluation(
        args.ratio,
        args.device,
        args.max_queries,
        args.output_tag,
        args.initial_misses_only,
    )


if __name__ == "__main__":
    main()
