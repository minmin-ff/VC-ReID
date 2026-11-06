"""Replay cached Dialog-ReID turns under verification-click noise.

This experiment isolates the binary Top-1 verification channel.  Questions and
answers are held fixed to the preserved clean run, while false accepts and
false rejects are sampled independently.  No VLM generation is repeated.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .common import add_parent_to_path, atomic_json_dump, load_paths, set_seed
from .interactive_official import (
    corrected_feature_path,
    embed_rows,
    load_manifest,
    load_models,
)
from .retrieval_metrics import evaluate_with_details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a preserved interactive run with verification noise"
    )
    parser.add_argument("--config", default="configs/paths.json")
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rates", default="0,0.05,0.10,0.20")
    parser.add_argument("--noise-seeds", default="3407,3408,3409,3410,3411")
    parser.add_argument("--text-batch-size", type=int, default=256)
    return parser.parse_args()


def encode_text_batch(clip, tokenizer, texts: list[str], device, batch_size: int) -> torch.Tensor:
    features = []
    for offset in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[offset : offset + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)
        with torch.inference_mode():
            batch = clip.get_text_features(**tokens)
        features.append(F.normalize(batch.float(), p=2, dim=1).cpu())
    return torch.cat(features)


def mean_std(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def metric_row(
    metrics: dict,
    details: list[dict],
    initial_rank1: dict[str, int],
    event_counts: dict[str, int],
) -> dict:
    final = {str(row["query_id"]): int(row["rank_1"]) for row in details}
    valid_ids = sorted(set(initial_rank1) & set(final))
    initial_correct = sum(initial_rank1[key] for key in valid_ids)
    initial_wrong = len(valid_ids) - initial_correct
    repaired = sum(not initial_rank1[key] and final[key] for key in valid_ids)
    damaged = sum(initial_rank1[key] and not final[key] for key in valid_ids)
    return {
        "mAP_at_4": float(metrics["mAP"]),
        "Rank1_at_4": float(metrics["Rank-1"]),
        "Repair_rate": 100.0 * repaired / initial_wrong,
        "Damage_rate": 100.0 * damaged / initial_correct,
        "counts": {
            "valid_queries": len(valid_ids),
            "initially_wrong": initial_wrong,
            "initially_correct": initial_correct,
            "repaired": repaired,
            "damaged": damaged,
            **event_counts,
        },
    }


def summarize(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["noise_type"], run["error_rate"])].append(run)
    output = []
    for (noise_type, rate), items in sorted(grouped.items()):
        output.append(
            {
                "noise_type": noise_type,
                "error_rate": rate,
                "seeds": [item["noise_seed"] for item in items],
                **{
                    key: mean_std(item[key] for item in items)
                    for key in ("mAP_at_4", "Rank1_at_4", "Repair_rate", "Damage_rate")
                },
            }
        )
    return output


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_result)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = source["configuration"]
    paths = load_paths(args.config)
    add_parent_to_path(paths)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    set_seed(3407)

    query_rows = load_manifest(Path(config["query_manifest"]))
    gallery_rows = load_manifest(Path(config["gallery_manifest"]))
    feature_root = Path(config["feature_root"])
    if len(query_rows) != 1980 or len(gallery_rows) != 9330:
        raise RuntimeError("Expected the official 1,980-query / 9,330-gallery protocol")

    _, _, clip, tokenizer, retriever = load_models(
        paths,
        Path(config["checkpoint"]),
        device,
        needs_dialogue=False,
        needs_text_encoder=True,
        requires_alignment=True,
    )
    query_initial = embed_rows(query_rows, feature_root, retriever, device)
    gallery_cpu = embed_rows(gallery_rows, feature_root, retriever, device)
    static_cpu = F.normalize(
        torch.from_numpy(
            np.stack(
                [
                    np.load(corrected_feature_path(row, feature_root)).astype(np.float32)[
                        4096:4608
                    ]
                    for row in query_rows
                ]
            )
        ),
        p=2,
        dim=1,
    )

    schedule = str(config["schedule"]).upper()
    interactions = source["interactions"]
    query_ids = [f"official:{row['official_index']}" for row in query_rows]
    qa_features: dict[int, torch.Tensor] = {}
    for round_index, question_type in enumerate(schedule, start=1):
        if question_type != "P":
            continue
        texts = []
        for query_id in query_ids:
            record = interactions[query_id][round_index - 1]
            if int(record["round"]) != round_index:
                raise ValueError(f"Unexpected interaction order for {query_id}")
            texts.append(f"{record['question']} {record['answer']}")
        qa_features[round_index] = encode_text_batch(
            clip, tokenizer, texts, device, args.text_batch_size
        )

    del clip, tokenizer, retriever
    torch.cuda.empty_cache()
    query_initial = query_initial.to(device)
    gallery = gallery_cpu.to(device)
    static = static_cpu.to(device)
    q_pid = torch.tensor([int(row["pid_numeric"]) for row in query_rows], device=device)
    q_cam = torch.tensor([int(row["camera_numeric"]) for row in query_rows], device=device)
    g_pid = torch.tensor([int(row["pid_numeric"]) for row in gallery_rows], device=device)
    g_cam = torch.tensor([int(row["camera_numeric"]) for row in gallery_rows], device=device)
    junk = (q_pid[:, None] == g_pid[None, :]) & (q_cam[:, None] == g_cam[None, :])

    initial_details = source["rounds"][0]["per_query"]
    initial_rank1 = {str(row["query_id"]): int(row["rank_1"]) for row in initial_details}
    query_pids = [str(row["pid"]) for row in query_rows]
    query_cameras = [str(row["camera"]) for row in query_rows]
    gallery_pids = [str(row["pid"]) for row in gallery_rows]
    gallery_cameras = [str(row["camera"]) for row in gallery_rows]

    rates = [float(value) for value in args.rates.split(",")]
    noise_seeds = [int(value) for value in args.noise_seeds.split(",")]
    conditions = [("clean", 0.0, noise_seeds[0])]
    for rate in rates:
        if rate == 0:
            continue
        for noise_type in ("false_accept", "false_reject", "symmetric"):
            conditions.extend((noise_type, rate, seed) for seed in noise_seeds)

    runs = []
    for noise_type, rate, noise_seed in conditions:
        rng = np.random.default_rng(noise_seed)
        current = query_initial.clone()
        positive_sum = static.clone()
        event_counts = defaultdict(int)
        for round_index, question_type in enumerate(schedule, start=1):
            if question_type == "P":
                positive_sum = positive_sum + qa_features[round_index].to(device)
            positive = F.normalize(positive_sum, p=2, dim=1)
            similarities = torch.mm(current, gallery.T)
            similarities.masked_fill_(junk, -float("inf"))
            top_index = similarities.argmax(dim=1)
            top1 = gallery[top_index]
            oracle = g_pid[top_index].eq(q_pid)
            draws = torch.from_numpy(rng.random(len(query_rows))).to(device)
            observed = oracle.clone()
            false_accept = (~oracle) & (draws < rate)
            false_reject = oracle & (draws < rate)
            if noise_type in {"false_accept", "symmetric"}:
                observed[false_accept] = True
                event_counts["false_accept_events"] += int(false_accept.sum().item())
            if noise_type in {"false_reject", "symmetric"}:
                observed[false_reject] = False
                event_counts["false_reject_events"] += int(false_reject.sum().item())
            alpha = float(config["alpha"]) * float(config["decay"]) ** (round_index - 1)
            accepted = F.normalize(current + alpha * top1, p=2, dim=1)
            rejected = F.normalize(
                current
                - alpha * top1
                + float(config["anchor_weight"]) * alpha * positive,
                p=2,
                dim=1,
            )
            current = torch.where(observed[:, None], accepted, rejected)

        metrics, details = evaluate_with_details(
            current.cpu(),
            query_pids,
            query_cameras,
            gallery_cpu,
            gallery_pids,
            gallery_cameras,
            query_ids,
        )
        row = {
            "noise_type": noise_type,
            "error_rate": rate,
            "noise_seed": noise_seed,
            **metric_row(metrics, details, initial_rank1, dict(event_counts)),
        }
        runs.append(row)
        print(json.dumps(row), flush=True)

    clean = runs[0]
    expected = source["rounds"][-1]["metrics"]
    if abs(clean["mAP_at_4"] - float(expected["mAP"])) > 0.02 or abs(
        clean["Rank1_at_4"] - float(expected["Rank-1"])
    ) > 0.02:
        raise RuntimeError(
            "Clean replay does not reproduce the source endpoint: "
            f"replay={clean['mAP_at_4']:.4f}/{clean['Rank1_at_4']:.4f}, "
            f"source={expected['mAP']:.4f}/{expected['Rank-1']:.4f}"
        )

    output = {
        "experiment": "cached_dialogue_verification_noise_replay",
        "source_result": str(source_path.resolve()),
        "protocol": {
            "questions_and_answers": "fixed from the clean preserved run",
            "noise_target": "binary Top-1 relevance click only",
            "false_accept": "wrong Top-1 observed as accepted",
            "false_reject": "correct Top-1 observed as rejected",
            "noise_draw": "independent Bernoulli draw per query and round",
            "valid_query_denominator": len(initial_rank1),
        },
        "configuration": {
            "rates": rates,
            "noise_seeds": noise_seeds,
            "device": args.device,
            "schedule": schedule,
            "alpha": config["alpha"],
            "decay": config["decay"],
            "anchor_weight": config["anchor_weight"],
        },
        "runs": runs,
        "summary": summarize(runs),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(output, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
