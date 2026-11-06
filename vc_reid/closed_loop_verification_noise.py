"""Closed-loop verification-noise evaluation on the official MARS protocol.

Unlike ``replay_verification_noise.py``, this diagnostic never reuses a clean
candidate trajectory.  A corrupted click changes the query immediately; the
next Top-1 is then retrieved from that changed state.  The experiment isolates
the representation-orthogonal signed verification rule and deliberately uses
no dialogue/semantic anchor.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .common import atomic_json_dump, set_seed
from .evaluate_manifest import embed_projection, load_manifest, override_feature_root
from .retrieval_metrics import evaluate_with_details


def protocol_arrays(
    query_rows: Sequence[Dict[str, object]],
    gallery_rows: Sequence[Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query_pid = np.asarray([str(row["pid"]) for row in query_rows])
    query_cam = np.asarray([str(row["camera"]) for row in query_rows])
    gallery_pid = np.asarray([str(row["pid"]) for row in gallery_rows])
    gallery_cam = np.asarray([str(row["camera"]) for row in gallery_rows])
    junk = (query_pid[:, None] == gallery_pid[None, :]) & (
        query_cam[:, None] == gallery_cam[None, :]
    )
    relevant = (query_pid[:, None] == gallery_pid[None, :]) & ~junk
    valid = relevant.any(axis=1)
    return query_pid, query_cam, gallery_pid, gallery_cam, junk, valid


@torch.inference_mode()
def retrieve_top1(
    query: torch.Tensor,
    gallery: torch.Tensor,
    junk: torch.Tensor,
) -> torch.Tensor:
    scores = torch.mm(query, gallery.T)
    scores.masked_fill_(junk, -torch.inf)
    return scores.argmax(dim=1)


def corrupt_clicks(
    oracle: np.ndarray,
    noise_type: str,
    error_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_type == "clean" or error_rate <= 0:
        return oracle.copy()
    draw = rng.random(len(oracle)) < error_rate
    observed = oracle.copy()
    if noise_type == "symmetric":
        observed[draw] = ~observed[draw]
    elif noise_type == "false_accept":
        observed[draw & ~oracle] = True
    elif noise_type == "false_reject":
        observed[draw & oracle] = False
    else:
        raise ValueError(noise_type)
    return observed


def retrieval_metrics(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_rows: Sequence[Dict[str, object]],
    gallery_rows: Sequence[Dict[str, object]],
) -> Dict[str, float]:
    metrics, _ = evaluate_with_details(
        query.cpu(),
        [str(row["pid"]) for row in query_rows],
        [str(row["camera"]) for row in query_rows],
        gallery.cpu(),
        [str(row["pid"]) for row in gallery_rows],
        [str(row["camera"]) for row in gallery_rows],
        [f"official:{row['official_index']}" for row in query_rows],
    )
    return metrics


@torch.inference_mode()
def run_condition(
    initial: torch.Tensor,
    gallery: torch.Tensor,
    junk: torch.Tensor,
    valid: np.ndarray,
    query_pid: np.ndarray,
    gallery_pid: np.ndarray,
    rounds: int,
    alpha: float,
    decay: float,
    noise_type: str,
    error_rate: float,
    noise_seed: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    rng = np.random.default_rng(noise_seed)
    current = initial.clone()
    true_correctness: List[np.ndarray] = []
    observed_accepts: List[np.ndarray] = []
    false_accepts = 0
    false_rejects = 0

    for round_index in range(rounds + 1):
        top = retrieve_top1(current, gallery, junk)
        top_cpu = top.cpu().numpy()
        oracle = gallery_pid[top_cpu] == query_pid
        true_correctness.append(oracle.copy())
        if round_index == rounds:
            break
        observed = corrupt_clicks(oracle, noise_type, error_rate, rng)
        observed_accepts.append(observed.copy())
        false_accepts += int(np.sum(observed & ~oracle & valid))
        false_rejects += int(np.sum(~observed & oracle & valid))
        candidate = gallery[top]
        step = alpha * decay**round_index
        accepted = F.normalize(current + step * candidate, p=2, dim=1)
        rejected = F.normalize(current - step * candidate, p=2, dim=1)
        mask = torch.from_numpy(observed).to(current.device).unsqueeze(1)
        current = torch.where(mask, accepted, rejected)

    initial_correct = true_correctness[0][valid]
    final_correct = true_correctness[-1][valid]
    wrong = ~initial_correct
    repaired = wrong & final_correct
    damaged = initial_correct & ~final_correct
    first_success = np.full(int(valid.sum()), -1, dtype=np.int64)
    for round_index, correctness in enumerate(true_correctness):
        active = correctness[valid]
        first_success[(first_success < 0) & active] = round_index
    repaired_steps = first_success[wrong & (first_success >= 0)]
    diagnostics = {
        "noise_type": noise_type,
        "error_rate": error_rate,
        "noise_seed": noise_seed,
        "valid_queries": int(valid.sum()),
        "initially_correct": int(initial_correct.sum()),
        "initially_wrong": int(wrong.sum()),
        "repaired": int(repaired.sum()),
        "damaged": int(damaged.sum()),
        "Repair_rate": 100.0 * float(repaired.sum()) / max(1, int(wrong.sum())),
        "Damage_rate": 100.0 * float(damaged.sum()) / max(1, int(initial_correct.sum())),
        "Mean_rounds_to_success": (
            float(repaired_steps.mean()) if len(repaired_steps) else None
        ),
        "false_accept_clicks": false_accepts,
        "false_reject_clicks": false_rejects,
        "rank1_trajectory": [
            100.0 * float(values[valid].mean()) for values in true_correctness
        ],
    }
    return current, diagnostics


def aggregate_runs(runs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, object]]] = defaultdict(list)
    for run in runs:
        grouped[(str(run["noise_type"]), float(run["error_rate"]))].append(run)
    output = []
    scalar_keys = (
        "mAP",
        "Rank-1",
        "Rank-5",
        "Rank-10",
        "Repair_rate",
        "Damage_rate",
        "Mean_rounds_to_success",
    )
    for (noise_type, rate), items in sorted(grouped.items()):
        summary: Dict[str, object] = {
            "noise_type": noise_type,
            "error_rate": rate,
            "seeds": [int(item["noise_seed"]) for item in items],
            "runs": len(items),
        }
        for key in scalar_keys:
            values = [float(item[key]) for item in items if item.get(key) is not None]
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
        trajectories = np.asarray([item["rank1_trajectory"] for item in items], dtype=np.float64)
        summary["rank1_trajectory"] = {
            "mean": trajectories.mean(axis=0).tolist(),
            "sample_std": (
                trajectories.std(axis=0, ddof=1).tolist()
                if len(trajectories) > 1
                else np.zeros(trajectories.shape[1]).tolist()
            ),
        }
        output.append(summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--gallery-manifest", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--projection-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--memory-fraction", type=float, default=0.30)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--rates", default="0.1,0.2")
    parser.add_argument("--noise-seeds", default="3407,3408,3409,3410,3411")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, args.device)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    query_rows = override_feature_root(load_manifest(args.query_manifest), args.feature_root)
    gallery_rows = override_feature_root(load_manifest(args.gallery_manifest), args.feature_root)
    if len(query_rows) != 1980 or len(gallery_rows) != 9330:
        raise RuntimeError("Refusing a non-official MARS query/gallery protocol")
    query = embed_projection(
        query_rows, "full", Path(args.projection_checkpoint), device, 512
    ).to(device)
    gallery = embed_projection(
        gallery_rows, "full", Path(args.projection_checkpoint), device, 512
    ).to(device)
    query = F.normalize(query.float(), p=2, dim=1)
    gallery = F.normalize(gallery.float(), p=2, dim=1)
    query_pid, _, gallery_pid, _, junk_np, valid = protocol_arrays(query_rows, gallery_rows)
    junk = torch.from_numpy(junk_np).to(device)

    initial_metrics = retrieval_metrics(query, gallery, query_rows, gallery_rows)
    seeds = [int(value) for value in args.noise_seeds.split(",")]
    rates = [float(value) for value in args.rates.split(",")]
    conditions = [("clean", 0.0, seeds[0])]
    for noise_type in ("false_accept", "false_reject", "symmetric"):
        for rate in rates:
            conditions.extend((noise_type, rate, seed) for seed in seeds)

    runs = []
    for noise_type, rate, noise_seed in conditions:
        final, diagnostic = run_condition(
            query,
            gallery,
            junk,
            valid,
            query_pid,
            gallery_pid,
            args.rounds,
            args.alpha,
            args.decay,
            noise_type,
            rate,
            noise_seed,
        )
        metrics = retrieval_metrics(final, gallery, query_rows, gallery_rows)
        runs.append({**diagnostic, **metrics})
        print(
            f"{noise_type:12s} rate={rate:.2f} seed={noise_seed} "
            f"mAP={metrics['mAP']:.2f} R1={metrics['Rank-1']:.2f} "
            f"RR={diagnostic['Repair_rate']:.2f} DR={diagnostic['Damage_rate']:.2f}",
            flush=True,
        )

    result = {
        "experiment": "official_mars_closed_loop_verification_noise",
        "protocol": {
            "query_count": len(query_rows),
            "valid_query_count": int(valid.sum()),
            "gallery_count": len(gallery_rows),
            "candidate_trajectory": "recomputed after every corrupted update",
            "feedback": "binary Top-1 verification only",
            "language_anchor": False,
            "noise_draw": "independent Bernoulli per query and round",
        },
        "configuration": vars(args),
        "initial_metrics": initial_metrics,
        "runs": runs,
        "summary": aggregate_runs(runs),
    }
    atomic_json_dump(result, Path(args.output))
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
