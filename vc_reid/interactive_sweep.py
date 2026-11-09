from __future__ import annotations

import argparse
import glob
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import (
    add_parent_to_path,
    atomic_json_dump,
    load_paths,
    make_run_dir,
    require_paths,
    set_seed,
    stable_uniform,
)
from .retrieval_metrics import evaluate_with_details


UPDATE_MODES = (
    "no_update",
    "semantic_only",
    "attraction_only",
    "repulsion_only",
    "signed_no_anchor",
    "full",
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, function, *args, **kwargs):
    synchronize(device)
    start = time.perf_counter()
    value = function(*args, **kwargs)
    synchronize(device)
    return value, time.perf_counter() - start


def load_models(paths, checkpoint_path: Path, device: torch.device):
    from transformers import (
        AutoProcessor,
        CLIPModel,
        CLIPTokenizer,
        Qwen3VLForConditionalGeneration,
    )
    from .retriever import load_retriever

    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        str(paths.qwen_path),
        dtype=torch.bfloat16,
        device_map=f"cuda:{device.index}",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        str(paths.qwen_path), local_files_only=True
    )
    clip = CLIPModel.from_pretrained(
        str(paths.clip_path), local_files_only=True
    ).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(
        str(paths.clip_path), local_files_only=True
    )
    retriever = load_retriever(str(checkpoint_path), device).eval()
    return qwen, processor, clip, tokenizer, retriever


def encode_text(clip, tokenizer, text: str, device: torch.device) -> torch.Tensor:
    tokens = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device)
    with torch.no_grad():
        feature = clip.get_text_features(**tokens).squeeze(0)
    return F.normalize(feature.float(), p=2, dim=0)


def static_anchor(raw_feature: np.ndarray, device: torch.device) -> torch.Tensor:
    flat = np.asarray(raw_feature, dtype=np.float32).reshape(-1)
    if flat.size != 4608:
        raise ValueError(f"The interactive full model requires 4608-D features, got {flat.size}")
    anchor = torch.from_numpy(flat[4096:4608]).to(device)
    if float(anchor.norm()) <= 1e-8:
        raise ValueError("Static semantic anchor is zero")
    return F.normalize(anchor, p=2, dim=0)


def corrupt_answer(answer: str) -> str:
    stripped = answer.strip()
    lowered = stripped.lower()
    if lowered.startswith("yes"):
        return "No." + stripped[3:]
    if lowered.startswith("no"):
        suffix = stripped[2:].lstrip(".,:; ")
        return "Yes." + (f" {suffix}" if suffix else "")
    return "No. The described attribute is absent."


def apply_update(
    query: torch.Tensor,
    top1: torch.Tensor,
    positive_anchor: torch.Tensor,
    top1_is_correct: bool,
    mode: str,
    alpha: float,
    anchor_weight: float,
) -> torch.Tensor:
    if mode == "no_update":
        return query
    if mode == "semantic_only":
        return F.normalize(query + alpha * positive_anchor, p=2, dim=0)
    if mode == "attraction_only":
        return (
            F.normalize(query + alpha * top1, p=2, dim=0)
            if top1_is_correct
            else query
        )
    if mode == "repulsion_only":
        return (
            query
            if top1_is_correct
            else F.normalize(query - alpha * top1, p=2, dim=0)
        )
    if mode in {"signed_no_anchor", "full"}:
        if top1_is_correct:
            return F.normalize(query + alpha * top1, p=2, dim=0)
        update = query - alpha * top1
        if mode == "full":
            update = update + anchor_weight * alpha * positive_anchor
        return F.normalize(update, p=2, dim=0)
    raise ValueError(mode)


def run_query(
    raw_feature: np.ndarray,
    query: Dict[str, str],
    gallery_features: torch.Tensor,
    gallery_pids: List[str],
    gallery_cameras: List[str],
    gallery_paths: List[str],
    retriever,
    qwen,
    processor,
    clip,
    tokenizer,
    paths,
    feature_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[torch.Tensor], List[Dict[str, object]], Dict[str, float]]:
    from questioner_answerer import (
        find_video_path,
        generate_answer,
        generate_negative_question,
        generate_positive_question,
    )
    from .retriever import retrieve_topk

    raw_tensor = torch.from_numpy(raw_feature.reshape(-1)).float().to(device)
    with torch.no_grad():
        projected, _ = retriever(raw_tensor.unsqueeze(0))
    current = F.normalize(projected.squeeze(0), p=2, dim=0)
    round_features = [current.detach().cpu()]
    records: List[Dict[str, object]] = []
    timings = defaultdict(float)
    if args.mode == "no_update":
        return [round_features[0]] * (len(args.schedule) + 1), records, dict(timings)

    query_video = find_video_path(
        str(paths.test_videos), query["pid"], query["tracklet"]
    )
    positive_sum = static_anchor(raw_feature, device)
    positive = positive_sum.clone()
    history = ""
    for round_index, question_type in enumerate(args.schedule, start=1):
        retrieval, elapsed = timed_call(
            device,
            retrieve_topk,
            current.detach().cpu(),
            gallery_features,
            gallery_pids,
            gallery_cameras,
            gallery_paths,
            query["pid"],
            query["camera"],
            5,
        )
        timings["retrieval_seconds"] += elapsed
        _, _, top_pids, top_paths = retrieval
        top_video_paths: List[str] = []
        for gallery_path in top_paths:
            position = gallery_paths.index(gallery_path)
            tracklet = Path(gallery_path).stem
            video = find_video_path(
                str(paths.test_videos), gallery_pids[position], tracklet
            )
            if video:
                top_video_paths.append(video)
        if not top_paths or not top_video_paths or not query_video:
            records.append(
                {
                    "round": round_index,
                    "status": "missing_video_or_candidate",
                    "top1_pid": top_pids[0] if top_pids else None,
                }
            )
            round_features.append(current.detach().cpu())
            continue
        top1_path = top_paths[0]
        top1_raw = np.load(top1_path).astype(np.float32).reshape(-1)
        with torch.no_grad():
            top1, _ = retriever(
                torch.from_numpy(top1_raw).float().unsqueeze(0).to(device)
            )
        top1 = F.normalize(top1.squeeze(0), p=2, dim=0)
        generator = (
            generate_positive_question
            if question_type == "P"
            else generate_negative_question
        )
        question, elapsed = timed_call(
            device,
            generator,
            qwen,
            processor,
            top_video_paths,
            history,
            device,
        )
        timings["question_seconds"] += elapsed
        answer, elapsed = timed_call(
            device,
            generate_answer,
            qwen,
            processor,
            query_video,
            question,
            device,
        )
        timings["answer_seconds"] += elapsed
        original_answer = answer
        corruption_key = f"{query['query_id']}:{round_index}"
        corrupted = stable_uniform(corruption_key, args.seed) < args.answer_error_rate
        if corrupted:
            answer = corrupt_answer(answer)
        qa_feature, elapsed = timed_call(
            device,
            encode_text,
            clip,
            tokenizer,
            f"{question} {answer}",
            device,
        )
        timings["semantic_seconds"] += elapsed
        if question_type == "P":
            positive_sum = positive_sum + qa_feature
            positive = F.normalize(positive_sum, p=2, dim=0)
        top1_is_correct = top_pids[0] == query["pid"]
        alpha = args.alpha * (args.decay ** (round_index - 1))
        start = time.perf_counter()
        current = apply_update(
            current,
            top1,
            positive,
            top1_is_correct,
            args.mode,
            alpha,
            args.anchor_weight,
        )
        timings["update_seconds"] += time.perf_counter() - start
        history += (
            f"\n[Round {round_index}] Q({question_type}): {question} | A: {answer}"
        )
        records.append(
            {
                "round": round_index,
                "status": "ok",
                "question_type": question_type,
                "question": question,
                "answer": answer,
                "original_answer": original_answer,
                "answer_corrupted": corrupted,
                "top1_pid": top_pids[0],
                "top1_is_correct": top1_is_correct,
                "alpha": alpha,
            }
        )
        round_features.append(current.detach().cpu())
    return round_features, records, dict(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical signed-feedback experiments")
    parser.add_argument("--config", default="configs/paths.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--feature-root")
    parser.add_argument("--mode", choices=UPDATE_MODES, default="full")
    parser.add_argument("--schedule", default="PNPN")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--anchor-weight", type=float, default=0.3)
    parser.add_argument("--answer-error-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-end", type=int)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.schedule = args.schedule.upper()
    if len(args.schedule) != 4 or set(args.schedule) - {"P", "N"}:
        parser.error("--schedule must contain exactly four P/N characters")
    if not 0 <= args.answer_error_rate <= 1:
        parser.error("--answer-error-rate must be between zero and one")
    paths = load_paths(args.config)
    add_parent_to_path(paths)
    checkpoint = Path(args.checkpoint) if args.checkpoint else paths.full_checkpoint
    feature_root = Path(args.feature_root) if args.feature_root else paths.test_features
    require_paths(
        {
            "checkpoint": checkpoint,
            "feature_root": feature_root,
            "qwen": paths.qwen_path,
            "clip": paths.clip_path,
            "videos": paths.test_videos,
        }
    )
    if args.dry_run:
        files = sorted(feature_root.glob("*/*.npy"))
        print(
            f"dry-run ok mode={args.mode} schedule={args.schedule} "
            f"features={len(files)} checkpoint={checkpoint}"
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for interactive evaluation")
    device = torch.device(f"cuda:{args.device}")
    # PyTorch 2.4.0+cu118 on the experiment server rejects
    # reset_peak_memory_stats on an uninitialised masked CUDA device, and also
    # after manual_seed_all. Initialise the selected device explicitly, reset
    # the counter, and only then seed the run.
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    set_seed(args.seed)
    from .retriever import build_gallery

    qwen, processor, clip, tokenizer, retriever = load_models(
        paths, checkpoint, device
    )
    gallery, gallery_pids, gallery_cameras, gallery_paths = build_gallery(
        str(feature_root), retriever, device
    )
    queries = []
    for path in sorted(feature_root.glob("*/*.npy")):
        pid = path.parent.name
        tracklet = path.stem
        queries.append(
            {
                "pid": pid,
                "camera": tracklet.split("_")[0],
                "tracklet": tracklet,
                "feature_path": str(path),
                "query_id": f"{pid}/{tracklet}",
            }
        )
    total_query_count = len(queries)
    query_end = total_query_count if args.query_end is None else args.query_end
    if args.query_start < 0 or query_end < args.query_start:
        parser.error("invalid --query-start/--query-end range")
    if query_end > total_query_count:
        parser.error(
            f"--query-end {query_end} exceeds the {total_query_count} available queries"
        )
    queries = queries[args.query_start:query_end]
    if args.max_queries:
        queries = queries[: args.max_queries]
    if not queries:
        parser.error("selected query shard is empty")
    round_features: Dict[int, List[torch.Tensor]] = {
        index: [] for index in range(5)
    }
    interactions: Dict[str, List[Dict[str, object]]] = {}
    total_timings = defaultdict(float)
    started = time.perf_counter()
    for query in tqdm(queries, desc=f"interactive:{args.mode}"):
        raw = np.load(query["feature_path"]).astype(np.float32)
        features, records, timings = run_query(
            raw,
            query,
            gallery,
            gallery_pids,
            gallery_cameras,
            gallery_paths,
            retriever,
            qwen,
            processor,
            clip,
            tokenizer,
            paths,
            feature_root,
            device,
            args,
        )
        for index, feature in enumerate(features):
            round_features[index].append(feature)
        interactions[query["query_id"]] = records
        for name, value in timings.items():
            total_timings[name] += value
    pids = [query["pid"] for query in queries]
    cameras = [query["camera"] for query in queries]
    query_ids = [query["query_id"] for query in queries]
    rounds = []
    for round_index in range(5):
        metrics, details = evaluate_with_details(
            torch.stack(round_features[round_index]),
            pids,
            cameras,
            gallery,
            gallery_pids,
            gallery_cameras,
            query_ids,
        )
        rounds.append(
            {"round": round_index, "metrics": metrics, "per_query": details}
        )
        print(round_index, metrics)
    wall_seconds = time.perf_counter() - started
    run_dir = make_run_dir(paths, "interactive", f"{args.mode}_{args.tag}")
    result = {
        "experiment": "interactive_signed_feedback",
        "configuration": {
            **vars(args),
            "checkpoint": str(checkpoint),
            "feature_root": str(feature_root),
        },
        "query_count": len(queries),
        "query_universe_count": total_query_count,
        "query_range": {
            "start": args.query_start,
            "end": query_end,
        },
        "rounds": rounds,
        "interactions": interactions,
        "timing": {
            **dict(total_timings),
            "wall_seconds": wall_seconds,
            "seconds_per_query": wall_seconds / max(1, len(queries)),
        },
        "resources": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }
    atomic_json_dump(result, run_dir / "result.json")
    print(f"Result written to {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
