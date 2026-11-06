"""Evaluate signed verification feedback on native TF-CLIP embeddings.

The evaluator supports the official MARS and LS-VID query/gallery splits.  It
keeps the query set outside the gallery and removes same-identity/same-camera
items before scoring.

The interaction is representation-agnostic and has no semantic anchor:

    accept: normalize(q + alpha_t * g_top1)
    reject: normalize(q - alpha_t * g_top1)

This is the ``signed_no_anchor`` condition requested for testing whether the
post-retrieval update remains useful with a stronger public backbone.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/vit_clipreid.yml")
    parser.add_argument("--dataset-name", choices=("mars", "lsvid"), default="mars")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--sampler", choices=("rrs_test", "dense"), default="rrs_test")
    parser.add_argument("--noise-rates", default="0.05,0.10,0.20")
    parser.add_argument("--noise-seeds", default="3407,3408,3409,3410,3411")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--skip-noise", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


@torch.inference_mode()
def extract_features(args, device):
    cache_path = Path(args.feature_cache)
    if cache_path.exists() and not args.force_extract:
        cached = torch.load(cache_path, map_location="cpu")
        cached_dataset = cached.get("dataset_name")
        if cached_dataset != args.dataset_name:
            raise RuntimeError(
                "Feature cache dataset mismatch: expected {}, found {}".format(
                    args.dataset_name, cached_dataset
                )
            )
        cached_checkpoint = Path(str(cached.get("checkpoint", ""))).resolve()
        requested_checkpoint = Path(args.checkpoint).resolve()
        if cached_checkpoint != requested_checkpoint:
            raise RuntimeError(
                "Feature cache checkpoint mismatch: expected {}, found {}".format(
                    requested_checkpoint, cached_checkpoint
                )
            )
        print("Loaded feature cache: {}".format(cache_path), flush=True)
        return cached

    from config import cfg
    from datasets.make_dataloader_clipreid import (
        make_eval_all_dataloader,
        make_eval_rrs_dataloader,
    )
    from model.make_model_clipreid import make_model

    cfg.defrost()
    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.NAMES = args.dataset_name
    cfg.DATASETS.ROOT_DIR = args.dataset_root
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.freeze()
    loader_factory = (
        make_eval_all_dataloader if args.sampler == "dense" else make_eval_rrs_dataloader
    )
    loader, num_query, num_classes, camera_num, view_num = loader_factory(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(args.checkpoint)
    model = model.to(device).eval()

    features = []
    pids = []
    camids = []
    started = time.monotonic()
    for batch_index, (images, pid, camid, camera_labels, view_labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        dense_tracklet = images.ndim == 6
        if dense_tracklet:
            batch_size, clips, sequence, channels, height, width = images.shape
            if batch_size != 1:
                raise RuntimeError("Dense evaluation expects one tracklet per batch")
            images = images.view(clips, sequence, channels, height, width)
            # The released dense evaluator flattens clips but leaves these
            # labels at batch size one.  TF-CLIP's SIE branch expects one label
            # per clip before it repeats each label across the temporal axis.
            camera_labels = camera_labels.repeat(clips)
            view_labels = view_labels.repeat(clips)
        camera_labels = camera_labels.to(device) if cfg.MODEL.SIE_CAMERA else None
        view_labels = view_labels.to(device) if cfg.MODEL.SIE_VIEW else None
        feature = model(images, cam_label=camera_labels, view_label=view_labels)
        if dense_tracklet:
            feature = feature.mean(dim=0, keepdim=True)
        features.append(feature.float().cpu())
        pids.extend(int(item) for item in pid)
        camids.extend(int(item) for item in camid)
        progress_period = 500 if args.sampler == "dense" else 50
        if (batch_index + 1) % progress_period == 0:
            print(
                "Feature extraction: {}/{} batches".format(batch_index + 1, len(loader)),
                flush=True,
            )

    matrix = F.normalize(torch.cat(features, dim=0), p=2, dim=1)
    if matrix.shape[0] != len(pids) or len(pids) != len(camids):
        raise RuntimeError("Feature, identity, and camera counts do not agree")
    if not 0 < num_query < matrix.shape[0]:
        raise RuntimeError("Invalid query boundary: {} / {}".format(num_query, matrix.shape[0]))
    expected_protocol = {
        "mars": (1980, 9330),
        "lsvid": (3504, 7829),
    }
    expected_query, expected_gallery = expected_protocol[args.dataset_name]
    actual_gallery = int(matrix.shape[0]) - int(num_query)
    if int(num_query) != expected_query or actual_gallery != expected_gallery:
        raise RuntimeError(
            "{} protocol mismatch: expected {}/{} query/gallery, got {}/{}".format(
                args.dataset_name,
                expected_query,
                expected_gallery,
                int(num_query),
                actual_gallery,
            )
        )
    cached = {
        "features": matrix,
        "pids": torch.tensor(pids, dtype=torch.long),
        "camids": torch.tensor(camids, dtype=torch.long),
        "num_query": num_query,
        "feature_dim": int(matrix.shape[1]),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "dataset_name": args.dataset_name,
        "sampler": args.sampler,
        "extraction_seconds": time.monotonic() - started,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(cached, temporary)
    temporary.replace(cache_path)
    print("Saved feature cache: {}".format(cache_path), flush=True)
    return cached


def retrieval_tensors(cached, device):
    num_query = int(cached["num_query"])
    features = F.normalize(cached["features"].float(), p=2, dim=1)
    pids = cached["pids"].long()
    camids = cached["camids"].long()
    query = features[:num_query].to(device)
    gallery = features[num_query:].to(device)
    q_pid = pids[:num_query].to(device)
    g_pid = pids[num_query:].to(device)
    q_cam = camids[:num_query].to(device)
    g_cam = camids[num_query:].to(device)
    junk = (q_pid[:, None] == g_pid[None, :]) & (q_cam[:, None] == g_cam[None, :])
    return query, gallery, q_pid, g_pid, junk


@torch.inference_mode()
def top1_state(query, gallery, q_pid, g_pid, junk):
    similarities = torch.mm(query, gallery.t())
    similarities.masked_fill_(junk, -float("inf"))
    top_index = similarities.argmax(dim=1)
    correct = g_pid[top_index].eq(q_pid)
    return top_index, correct


@torch.inference_mode()
def evaluate(query, gallery, q_pid, g_pid, junk):
    similarities = torch.mm(query, gallery.t())
    similarities.masked_fill_(junk, -float("inf"))
    order = similarities.argsort(dim=1, descending=True)
    del similarities
    matches = g_pid[order].eq(q_pid[:, None])
    # Masked same-ID/same-camera samples sort to the tail at -inf, but they
    # must also be removed from the relevance labels.  Otherwise they inflate
    # the valid-query count and enter the AP denominator at very low ranks.
    matches &= ~junk.gather(1, order)
    ranks = torch.arange(1, gallery.shape[0] + 1, device=query.device).float()[None, :]
    cumulative = matches.cumsum(dim=1).float()
    relevant = matches.sum(dim=1)
    valid = relevant.gt(0)
    ap = ((cumulative / ranks) * matches).sum(dim=1) / relevant.clamp_min(1)
    metrics = {
        "mAP": 100.0 * float(ap[valid].mean().item()),
        "valid_queries": int(valid.sum().item()),
    }
    for rank in (1, 5, 10):
        metrics["Rank-{}".format(rank)] = 100.0 * float(
            matches[valid, :rank].any(dim=1).float().mean().item()
        )
    top_index = order[:, 0]
    top_correct = matches[:, 0]
    return metrics, top_index, top_correct, valid


def update_query(current, top_feature, observed_correct, alpha):
    accepted = F.normalize(current + alpha * top_feature, p=2, dim=1)
    rejected = F.normalize(current - alpha * top_feature, p=2, dim=1)
    return torch.where(observed_correct[:, None], accepted, rejected)


def transition_metrics(initial_correct, final_correct, valid, initial_rank1, final_rank1):
    initial_correct_valid = valid & initial_correct
    initial_wrong = valid & ~initial_correct
    repaired = initial_wrong & final_correct
    damaged = initial_correct_valid & ~final_correct
    error_denominator = 100.0 - initial_rank1
    return {
        "Repair_rate": 100.0 * float(repaired.sum().item()) / int(initial_wrong.sum().item()),
        "Damage_rate": 100.0
        * float(damaged.sum().item())
        / int(initial_correct_valid.sum().item()),
        "Normalized_error_reduction": (final_rank1 - initial_rank1) / error_denominator,
        "counts": {
            "initially_wrong": int(initial_wrong.sum().item()),
            "initially_correct": int(initial_correct_valid.sum().item()),
            "repaired": int(repaired.sum().item()),
            "damaged": int(damaged.sum().item()),
        },
    }


@torch.inference_mode()
def run_fixed(query, gallery, q_pid, g_pid, junk, rounds, alpha, decay):
    current = query.clone()
    initial_metrics, _, initial_correct, valid = evaluate(
        current, gallery, q_pid, g_pid, junk
    )
    output_rounds = [{"round": 0, "metrics": initial_metrics}]
    for round_index in range(1, rounds + 1):
        top_index, oracle = top1_state(current, gallery, q_pid, g_pid, junk)
        step = alpha * decay ** (round_index - 1)
        current = update_query(current, gallery[top_index], oracle, step)
        metrics, _, correct, _ = evaluate(current, gallery, q_pid, g_pid, junk)
        output_rounds.append({"round": round_index, "alpha": step, "metrics": metrics})
        print(
            "Fixed round {}: mAP={:.4f} R1={:.4f} R5={:.4f} R10={:.4f}".format(
                round_index,
                metrics["mAP"],
                metrics["Rank-1"],
                metrics["Rank-5"],
                metrics["Rank-10"],
            ),
            flush=True,
        )
    output = transition_metrics(
        initial_correct,
        correct,
        valid,
        initial_metrics["Rank-1"],
        output_rounds[-1]["metrics"]["Rank-1"],
    )
    output["rounds"] = output_rounds
    return output, initial_correct


@torch.inference_mode()
def run_early_stop(query, gallery, q_pid, g_pid, junk, rounds, alpha, decay):
    current = query.clone()
    initial_metrics, _, initial_correct, valid = evaluate(
        current, gallery, q_pid, g_pid, junk
    )
    success_round = torch.full((len(query),), -1, dtype=torch.long, device=query.device)
    success_round[valid & initial_correct] = 0
    active = valid & ~initial_correct
    updates_executed = torch.zeros(len(query), dtype=torch.long, device=query.device)
    round_rows = []
    for round_index in range(1, rounds + 1):
        top_index, oracle = top1_state(current, gallery, q_pid, g_pid, junk)
        if bool((oracle & active).any()):
            success_round[oracle & active] = round_index - 1
            active &= ~oracle
        step = alpha * decay ** (round_index - 1)
        selected = active.nonzero(as_tuple=False).squeeze(1)
        if len(selected):
            # Active clean-protocol queries have just rejected an incorrect Top-1.
            current[selected] = F.normalize(
                current[selected] - step * gallery[top_index[selected]], p=2, dim=1
            )
            updates_executed[selected] += 1
        metrics, _, correct_after, _ = evaluate(current, gallery, q_pid, g_pid, junk)
        newly_correct = active & correct_after
        success_round[newly_correct] = round_index
        active &= ~newly_correct
        round_rows.append(
            {
                "round": round_index,
                "alpha": step,
                "active_queries_after_round": int(active.sum().item()),
                "metrics": metrics,
            }
        )
    final_metrics, _, final_correct, _ = evaluate(current, gallery, q_pid, g_pid, junk)
    successful = success_round.ge(0)
    repaired = (~initial_correct) & successful
    return {
        "rounds": round_rows,
        "final_metrics": final_metrics,
        "Mean_rounds_to_success_all_successes": float(
            success_round[successful].float().mean().item()
        ),
        "Mean_rounds_to_success_initial_failures": (
            float(success_round[repaired].float().mean().item()) if bool(repaired.any()) else None
        ),
        "Mean_updates_per_query": float(updates_executed[valid].float().mean().item()),
        "Unresolved_after_max_rounds": int(active.sum().item()),
        **transition_metrics(
            initial_correct,
            final_correct,
            valid,
            initial_metrics["Rank-1"],
            final_metrics["Rank-1"],
        ),
    }


@torch.inference_mode()
def run_noise_condition(
    query,
    gallery,
    q_pid,
    g_pid,
    junk,
    initial_correct,
    initial_rank1,
    rounds,
    alpha,
    decay,
    noise_type,
    rate,
    seed,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    current = query.clone()
    false_accept_events = 0
    false_reject_events = 0
    for round_index in range(1, rounds + 1):
        top_index, oracle = top1_state(current, gallery, q_pid, g_pid, junk)
        draws = torch.rand(len(query), generator=generator).to(query.device)
        observed = oracle.clone()
        false_accept = (~oracle) & draws.lt(rate)
        false_reject = oracle & draws.lt(rate)
        if noise_type in ("false_accept", "symmetric"):
            observed[false_accept] = True
            false_accept_events += int(false_accept.sum().item())
        if noise_type in ("false_reject", "symmetric"):
            observed[false_reject] = False
            false_reject_events += int(false_reject.sum().item())
        step = alpha * decay ** (round_index - 1)
        current = update_query(current, gallery[top_index], observed, step)
    metrics, _, final_correct, valid = evaluate(current, gallery, q_pid, g_pid, junk)
    return {
        "noise_type": noise_type,
        "error_rate": rate,
        "noise_seed": seed,
        "mAP_at_4": metrics["mAP"],
        "Rank1_at_4": metrics["Rank-1"],
        "false_accept_events": false_accept_events,
        "false_reject_events": false_reject_events,
        **transition_metrics(
            initial_correct, final_correct, valid, initial_rank1, metrics["Rank-1"]
        ),
    }


def summarize_noise(runs):
    grouped = defaultdict(list)
    for row in runs:
        grouped[(row["noise_type"], row["error_rate"])].append(row)
    summary = []
    for (noise_type, rate), rows in sorted(grouped.items()):
        item = {"noise_type": noise_type, "error_rate": rate, "seeds": len(rows)}
        for key in ("mAP_at_4", "Rank1_at_4", "Repair_rate", "Damage_rate"):
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            item[key] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        summary.append(item)
    return summary


def main():
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda:{}".format(args.device))
    cached = extract_features(args, device)
    query, gallery, q_pid, g_pid, junk = retrieval_tensors(cached, device)
    print(
        "Protocol: {} query / {} gallery / {}-D".format(
            len(query), len(gallery), query.shape[1]
        ),
        flush=True,
    )
    fixed, initial_correct = run_fixed(
        query, gallery, q_pid, g_pid, junk, args.rounds, args.alpha, args.decay
    )
    early_stop = run_early_stop(
        query, gallery, q_pid, g_pid, junk, args.rounds, args.alpha, args.decay
    )
    noise_runs = []
    if not args.skip_noise:
        rates = [float(value) for value in args.noise_rates.split(",") if value]
        seeds = [int(value) for value in args.noise_seeds.split(",") if value]
        initial_rank1 = fixed["rounds"][0]["metrics"]["Rank-1"]
        clean = run_noise_condition(
            query,
            gallery,
            q_pid,
            g_pid,
            junk,
            initial_correct,
            initial_rank1,
            args.rounds,
            args.alpha,
            args.decay,
            "clean",
            0.0,
            seeds[0],
        )
        noise_runs.append(clean)
        print(json.dumps(clean), flush=True)
        for rate in rates:
            for noise_type in ("false_accept", "false_reject", "symmetric"):
                for seed in seeds:
                    row = run_noise_condition(
                        query,
                        gallery,
                        q_pid,
                        g_pid,
                        junk,
                        initial_correct,
                        initial_rank1,
                        args.rounds,
                        args.alpha,
                        args.decay,
                        noise_type,
                        rate,
                        seed,
                    )
                    noise_runs.append(row)
                    print(json.dumps(row), flush=True)

    result = {
        "experiment": "TF-CLIP_native_signed_no_anchor",
        "protocol": {
            "dataset": args.dataset_name,
            "query_count": len(query),
            "gallery_count": len(gallery),
            "gallery_excludes_query_set": True,
            "same_identity_same_camera_removed": True,
            "sampler": cached.get("sampler"),
            "feature_dimension": int(query.shape[1]),
            "fixed_rounds": args.rounds,
            "update": "accept normalize(q+a*g); reject normalize(q-a*g)",
            "semantic_anchor": False,
            "verification": "identity labels simulate user Top-1 confirmation",
        },
        "configuration": vars(args),
        "feature_cache_metadata": {
            key: value for key, value in cached.items() if key not in {"features", "pids", "camids"}
        },
        "fixed_rounds": fixed,
        "early_stop": early_stop,
        "verification_noise": {
            "runs": noise_runs,
            "summary": summarize_noise(noise_runs),
        },
    }
    atomic_json_dump(result, args.output)
    print("Wrote {}".format(args.output), flush=True)


if __name__ == "__main__":
    main()
