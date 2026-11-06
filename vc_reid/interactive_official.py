from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import add_parent_to_path, atomic_json_dump, load_paths, make_run_dir, set_seed
from .feature_views import select_view
from .interactive_sweep import apply_update, encode_text, static_anchor, timed_call
from .retrieval_metrics import evaluate_with_details


def load_manifest(path: Path) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must be a list: {path}")
    return rows


def corrected_feature_path(row: Dict[str, object], root: Path) -> Path:
    pid = str(row["pid"])
    if int(row["pid_numeric"]) > 0:
        name = str(row["custom_tracklet"])
    else:
        name = f"cam{int(row['camera_numeric'])}_official{int(row['official_index']):05d}"
    return root / pid / f"{name}.npy"


def video_path(
    row: Dict[str, object], positive_video_root: Path, distractor_video_root: Path
) -> Path:
    if int(row["pid_numeric"]) == 0:
        name = f"cam{int(row['camera_numeric'])}_official{int(row['official_index']):05d}.mp4"
        return distractor_video_root / "0000" / name
    pid = str(row["pid"])
    tracklet = str(row["custom_tracklet"])
    candidates = (
        positive_video_root / pid / f"{tracklet}.mp4",
        positive_video_root / pid / f"{pid}_{tracklet}.mp4",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted((positive_video_root / pid).glob(f"*{tracklet}*.mp4"))
    return matches[0] if matches else candidates[0]


def load_models(
    paths,
    checkpoint_path: Path,
    device: torch.device,
    needs_dialogue: bool,
    needs_text_encoder: bool,
    requires_alignment: bool,
):
    from retriever import load_retriever

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if requires_alignment and not bool(checkpoint.get("clip_aligned", False)):
        raise RuntimeError(
            "Semantic feedback requires a CLIP-aligned retrieval checkpoint; "
            "this checkpoint does not declare clip_aligned=true."
        )
    retriever = load_retriever(str(checkpoint_path), device).eval()
    # Keep the input feature view with the loaded projection.  The official
    # feature-composition ablation trains 2,048/2,560/4,096-D heads as well as
    # the 4,608-D full head, while all descriptors remain stored in one common
    # 4,608-D file format.
    retriever.feature_view = str(checkpoint.get("feature_view", "full"))
    if not needs_dialogue and not needs_text_encoder:
        return None, None, None, None, retriever
    from transformers import CLIPModel, CLIPTokenizer

    clip = CLIPModel.from_pretrained(str(paths.clip_path), local_files_only=True).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(str(paths.clip_path), local_files_only=True)
    if not needs_dialogue:
        return None, None, clip, tokenizer, retriever

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        str(paths.qwen_path),
        dtype=torch.bfloat16,
        device_map=f"cuda:{device.index}",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(str(paths.qwen_path), local_files_only=True)
    return qwen, processor, clip, tokenizer, retriever


def query_annotation_path(row: Dict[str, object], root: Path) -> Path:
    return root / str(row["subset"]) / f"official_{int(row['official_index']):05d}.json"


def load_initial_description(row: Dict[str, object], root: Path) -> str:
    path = query_annotation_path(row, root)
    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    description = str(record.get("annotation", {}).get("description", "")).strip()
    if not description:
        raise ValueError(f"Missing initial text description: {path}")
    return description


def description_variants(description: str, mode: str) -> List[str]:
    """Return fixed, label-free CLIP prompts for a user description.

    The variants contain no annotation fields beyond ``description``.  The
    ensemble mode is selected on held-out training identities, never on the
    official test split.
    """
    if mode == "raw":
        return [description]
    if mode == "retrieval2":
        return [
            description,
            f"A surveillance tracklet showing the target person: {description}",
        ]
    if mode == "retrieval3":
        return [
            description,
            f"A surveillance tracklet showing the target person: {description}",
            f"Visual attributes of the target person: {description}",
        ]
    raise ValueError(f"Unknown text prompt ensemble: {mode}")


@torch.inference_mode()
def embed_initial_texts(
    rows: Sequence[Dict[str, object]],
    annotation_root: Path,
    clip,
    tokenizer,
    text_adapter,
    device: torch.device,
    prompt_mode: str = "raw",
) -> tuple[torch.Tensor, List[str]]:
    """Encode only the user-visible initial descriptions.

    This function deliberately does not accept a feature root or video path.  It
    is the modality guard for the text-to-video protocol: query tracklet visual
    features cannot enter the initial retrieval state through this code path.
    """
    descriptions = [load_initial_description(row, annotation_root) for row in rows]
    outputs = []
    for description in descriptions:
        variants = [
            encode_text(clip, tokenizer, prompted, device)
            for prompted in description_variants(description, prompt_mode)
        ]
        feature = F.normalize(torch.stack(variants).mean(dim=0), p=2, dim=0)
        if text_adapter is not None:
            feature = text_adapter(feature.unsqueeze(0)).squeeze(0)
        outputs.append(F.normalize(feature, p=2, dim=0).cpu())
    return torch.stack(outputs), descriptions


@torch.inference_mode()
def embed_rows(
    rows: Sequence[Dict[str, object]],
    feature_root: Path,
    retriever,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    outputs = []
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        matrix = np.stack(
            [
                select_view(
                    np.load(corrected_feature_path(row, feature_root)).astype(np.float32),
                    getattr(retriever, "feature_view", "full"),
                )
                for row in batch_rows
            ]
        )
        projected, _ = retriever(torch.from_numpy(matrix).to(device))
        outputs.append(F.normalize(projected, p=2, dim=1).cpu())
    return torch.cat(outputs)


def embed_static_rows(
    rows: Sequence[Dict[str, object]], feature_root: Path
) -> torch.Tensor:
    matrix = np.stack(
        [
            np.load(corrected_feature_path(row, feature_root)).astype(np.float32)[4096:4608]
            for row in rows
        ]
    )
    return F.normalize(torch.from_numpy(matrix).float(), p=2, dim=1)


def retrieve(
    query: torch.Tensor,
    query_row: Dict[str, object],
    gallery: torch.Tensor,
    gallery_rows: Sequence[Dict[str, object]],
    topk: int,
) -> List[int]:
    similarities = torch.mv(gallery, F.normalize(query.float(), p=2, dim=0).cpu())
    query_pid = str(query_row["pid"])
    query_camera = str(query_row["camera"])
    junk = [
        index
        for index, row in enumerate(gallery_rows)
        if str(row["pid"]) == query_pid and str(row["camera"]) == query_camera
    ]
    if junk:
        similarities[torch.tensor(junk, dtype=torch.long)] = -float("inf")
    return similarities.argsort(descending=True)[:topk].tolist()


def run_query(
    query_row: Dict[str, object],
    query_initial: torch.Tensor,
    initial_positive_anchor: torch.Tensor,
    initial_description: str,
    query_video: Path,
    gallery: torch.Tensor,
    gallery_rows: Sequence[Dict[str, object]],
    gallery_videos: Sequence[Path],
    qwen,
    processor,
    clip,
    tokenizer,
    text_adapter,
    retriever,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[torch.Tensor], List[Dict[str, object]], Dict[str, float]]:
    from .dialogue_questioner import generate_negative_question, generate_positive_question

    current = F.normalize(query_initial.to(device), p=2, dim=0)
    rounds = [current.cpu()]
    records = []
    timings = defaultdict(float)
    if args.mode == "no_update":
        return [rounds[0]] * (len(args.schedule) + 1), records, dict(timings)
    # In the text-query protocol, the semantic anchor starts from exactly the
    # same user-visible text that produced q_0.  No query-tracklet feature is
    # loaded or used by retrieval/update.  query_video is hidden from retrieval
    # and is used only below to simulate a witness answer reproducibly.
    positive_sum = F.normalize(initial_positive_anchor.to(device), p=2, dim=0)
    positive = positive_sum.clone()
    history = f"[Initial description] {initial_description}"
    for round_index, question_type in enumerate(args.schedule, start=1):
        indices, elapsed = timed_call(
            device, retrieve, current, query_row, gallery, gallery_rows, args.topk
        )
        timings["retrieval_seconds"] += elapsed
        candidate_videos = [str(gallery_videos[index]) for index in indices]
        if any(not Path(path).exists() for path in candidate_videos) or not query_video.exists():
            raise FileNotFoundError("A query or top-k candidate video is missing")
        generator = generate_positive_question if question_type == "P" else generate_negative_question
        question, elapsed = timed_call(
            device, generator, qwen, processor, candidate_videos, history, device
        )
        timings["question_seconds"] += elapsed
        from .dialogue_questioner import generate_answer

        answer, elapsed = timed_call(
            device, generate_answer, qwen, processor, str(query_video), question, device
        )
        timings["answer_seconds"] += elapsed
        qa_feature, elapsed = timed_call(
            device, encode_text, clip, tokenizer, f"{question} {answer}", device
        )
        timings["semantic_seconds"] += elapsed
        if text_adapter is not None:
            qa_feature = text_adapter(qa_feature.unsqueeze(0)).squeeze(0)
        if question_type == "P":
            positive_sum = positive_sum + qa_feature
            positive = F.normalize(positive_sum, p=2, dim=0)
        top_index = indices[0]
        top_row = gallery_rows[top_index]
        # This is an explicit simulation of the user's binary relevance click.
        oracle_relevance = str(top_row["pid"]) == str(query_row["pid"])
        alpha = args.alpha * args.decay ** (round_index - 1)
        current = apply_update(
            current,
            gallery[top_index].to(device),
            positive,
            oracle_relevance,
            args.mode,
            alpha,
            args.anchor_weight,
        )
        history += f"\n[Round {round_index}] Q({question_type}): {question} | A: {answer}"
        records.append(
            {
                "round": round_index,
                "question_type": question_type,
                "question": question,
                "answer": answer,
                "top1_official_index": top_row["official_index"],
                "simulated_user_relevance": oracle_relevance,
                "feedback_source": "ground-truth identity label simulating an explicit user click",
                "alpha": alpha,
            }
        )
        rounds.append(current.detach().cpu())
    return rounds, records, dict(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official-manifest interactive MARS evaluation")
    parser.add_argument("--config", default="configs/paths.json")
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--gallery-manifest", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--query-modality",
        choices=("text", "tracklet"),
        default="text",
        help="Use text for the paper protocol. Tracklet is retained only for legacy diagnostics.",
    )
    parser.add_argument(
        "--query-annotation-root",
        help="Root containing query/official_XXXXX.json initial descriptions.",
    )
    parser.add_argument(
        "--text-adapter",
        help="Adapter from CLIP text space to the 512-D ReID retrieval space.",
    )
    parser.add_argument(
        "--text-prompt-ensemble",
        choices=("raw", "retrieval2", "retrieval3"),
        default="raw",
        help="Fixed label-free description prompting selected on train identities only.",
    )
    parser.add_argument(
        "--gallery-static-residual-weight",
        type=float,
        default=0.0,
        help="Weight of the normalized gallery static-CLIP residual; select on train identities only.",
    )
    parser.add_argument(
        "--gallery-semantic-source",
        choices=("stored_static", "description"),
        default="stored_static",
        help="Semantic residual source. Description uses independently generated gallery text.",
    )
    parser.add_argument("--distractor-video-root", required=True)
    parser.add_argument("--mode", default="full", choices=("no_update", "semantic_only", "attraction_only", "repulsion_only", "signed_no_anchor", "full"))
    parser.add_argument("--schedule", default="PNPN")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--anchor-weight", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-end", type=int)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--tag", default="official")
    args = parser.parse_args()
    args.schedule = args.schedule.upper()
    if len(args.schedule) != 4 or set(args.schedule) - {"P", "N"}:
        parser.error("--schedule must contain exactly four P/N characters")
    paths = load_paths(args.config)
    add_parent_to_path(paths)
    query_all = load_manifest(Path(args.query_manifest))
    gallery_rows = load_manifest(Path(args.gallery_manifest))
    if len(query_all) != 1980 or len(gallery_rows) != 9330:
        raise RuntimeError("Refusing a non-official MARS query/gallery count")
    feature_root = Path(args.feature_root)
    missing_features = [
        corrected_feature_path(row, feature_root)
        for row in query_all + gallery_rows
        if not corrected_feature_path(row, feature_root).exists()
    ]
    if missing_features:
        raise FileNotFoundError(f"Official protocol is missing {len(missing_features)} features")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    set_seed(args.seed)
    needs_dialogue = args.mode != "no_update"
    needs_text_encoder = args.query_modality == "text"
    requires_alignment = args.query_modality == "text" or args.mode in {"semantic_only", "full"}
    qwen, processor, clip, tokenizer, retriever = load_models(
        paths,
        Path(args.checkpoint),
        device,
        needs_dialogue,
        needs_text_encoder,
        requires_alignment,
    )
    text_adapter = None
    if args.query_modality == "text":
        if not args.text_adapter:
            parser.error("--text-adapter is required for --query-modality text")
        from .text_adapter import load_text_adapter

        text_adapter = load_text_adapter(args.text_adapter, device)
    if not 0.0 <= args.gallery_static_residual_weight <= 1.0:
        parser.error("--gallery-static-residual-weight must be in [0, 1]")
    gallery = embed_rows(gallery_rows, feature_root, retriever, device)
    if args.gallery_static_residual_weight > 0:
        if args.gallery_semantic_source == "description":
            if not args.query_annotation_root:
                parser.error("--query-annotation-root is required for description residuals")
            gallery_static, _ = embed_initial_texts(
                gallery_rows,
                Path(args.query_annotation_root),
                clip,
                tokenizer,
                text_adapter,
                device,
                args.text_prompt_ensemble,
            )
        else:
            gallery_static = embed_static_rows(gallery_rows, feature_root)
        weight = args.gallery_static_residual_weight
        gallery = F.normalize((1.0 - weight) * gallery + weight * gallery_static, p=2, dim=1)
    distractor_video_root = Path(args.distractor_video_root)
    gallery_videos = [video_path(row, paths.test_videos, distractor_video_root) for row in gallery_rows]
    start = args.query_start
    end = len(query_all) if args.query_end is None else min(args.query_end, len(query_all))
    query_rows = query_all[start:end]
    if args.max_queries is not None:
        query_rows = query_rows[: args.max_queries]
    if args.query_modality == "text":
        if not args.query_annotation_root:
            parser.error("--query-annotation-root is required for --query-modality text")
        query_initial, initial_descriptions = embed_initial_texts(
            query_rows,
            Path(args.query_annotation_root),
            clip,
            tokenizer,
            text_adapter,
            device,
            args.text_prompt_ensemble,
        )
        query_initial_source = "CLIP encoding of annotation.description only"
    else:
        query_initial = embed_rows(query_rows, feature_root, retriever, device)
        initial_descriptions = ["" for _ in query_rows]
        query_initial_source = "legacy projected query-tracklet feature"
    round_features = {index: [] for index in range(5)}
    interactions = {}
    total_timings = defaultdict(float)
    begun = time.perf_counter()
    for local_index, (row, initial, initial_description) in enumerate(
        tqdm(zip(query_rows, query_initial, initial_descriptions), total=len(query_rows))
    ):
        if args.query_modality == "text":
            initial_positive_anchor = initial
        else:
            raw = np.load(corrected_feature_path(row, feature_root)).astype(np.float32)
            initial_positive_anchor = static_anchor(raw, device).cpu()
        rounds, records, timings = run_query(
            row,
            initial,
            initial_positive_anchor,
            initial_description,
            video_path(row, paths.test_videos, distractor_video_root),
            gallery,
            gallery_rows,
            gallery_videos,
            qwen,
            processor,
            clip,
            tokenizer,
            text_adapter,
            retriever,
            device,
            args,
        )
        for round_index, feature in enumerate(rounds):
            round_features[round_index].append(feature)
        interactions[f"official:{row['official_index']}"] = records
        for key, value in timings.items():
            total_timings[key] += value
    query_pids = [str(row["pid"]) for row in query_rows]
    query_cameras = [str(row["camera"]) for row in query_rows]
    gallery_pids = [str(row["pid"]) for row in gallery_rows]
    gallery_cameras = [str(row["camera"]) for row in gallery_rows]
    query_ids = [f"official:{row['official_index']}" for row in query_rows]
    evaluated_rounds = []
    for round_index in range(5):
        metrics, details = evaluate_with_details(
            torch.stack(round_features[round_index]),
            query_pids,
            query_cameras,
            gallery,
            gallery_pids,
            gallery_cameras,
            query_ids,
        )
        evaluated_rounds.append({"round": round_index, "metrics": metrics, "per_query": details})
        print(round_index, metrics)
    result = {
        "experiment": "official_mars_interactive",
        "query_count": len(query_rows),
        "query_universe_count": len(query_all),
        "protocol": {
            "query_count": 1980,
            "gallery_count": 9330,
            "gallery_pid0_distractors": 3248,
            "query_modality": args.query_modality,
            "query_initial_source": query_initial_source,
            "text_adapter": str(Path(args.text_adapter).resolve()) if args.text_adapter else None,
            "text_prompt_ensemble": args.text_prompt_ensemble,
            "gallery_static_residual_weight": args.gallery_static_residual_weight,
            "gallery_semantic_source": args.gallery_semantic_source,
            "query_video_role": "hidden answer simulator only; never used by initial retrieval or query update",
            "feedback_source": "oracle binary relevance simulating an explicit user click",
        },
        "configuration": vars(args),
        "query_range": {"start": start, "end": end},
        "rounds": evaluated_rounds,
        "interactions": interactions,
        "timing": {**total_timings, "wall_seconds": time.perf_counter() - begun},
    }
    run_dir = make_run_dir(paths, "interactive_official", args.tag)
    atomic_json_dump(result, run_dir / "result.json")
    print(f"Result written to {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
