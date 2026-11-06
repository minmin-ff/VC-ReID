from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def evaluate_with_details(
    query_features: torch.Tensor,
    query_pids: Sequence[str],
    query_cameras: Sequence[str],
    gallery_features: torch.Tensor,
    gallery_pids: Sequence[str],
    gallery_cameras: Sequence[str],
    query_ids: Sequence[str] | None = None,
    ranks: Tuple[int, ...] = (1, 5, 10),
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    q = F.normalize(query_features.float(), p=2, dim=1).cpu()
    g = F.normalize(gallery_features.float(), p=2, dim=1).cpu()
    similarities = torch.mm(q, g.T).numpy()
    gallery_pid_array = np.asarray(gallery_pids)
    gallery_camera_array = np.asarray(gallery_cameras)
    max_rank = max(ranks)
    details: List[Dict[str, object]] = []

    for index, (query_pid, query_camera) in enumerate(
        zip(query_pids, query_cameras)
    ):
        junk = (gallery_pid_array == query_pid) & (
            gallery_camera_array == query_camera
        )
        positives = (gallery_pid_array == query_pid) & ~junk
        if not positives.any():
            continue
        order = np.argsort(-similarities[index])
        order = order[~junk[order]]
        matches = positives[order]
        positive_positions = np.flatnonzero(matches)
        precisions = np.arange(1, len(positive_positions) + 1) / (
            positive_positions + 1
        )
        average_precision = float(precisions.mean())
        first_rank = int(positive_positions[0] + 1)
        details.append(
            {
                "query_id": (
                    query_ids[index]
                    if query_ids is not None
                    else f"{query_pid}/{query_camera}/{index}"
                ),
                "pid": str(query_pid),
                "camera": str(query_camera),
                "ap": average_precision,
                "first_positive_rank": first_rank,
                **{f"rank_{rank}": int(first_rank <= rank) for rank in ranks},
            }
        )

    if not details:
        return {
            "valid_queries": 0,
            "mAP": 0.0,
            **{f"Rank-{rank}": 0.0 for rank in ranks},
        }, []
    metrics: Dict[str, float] = {
        "valid_queries": len(details),
        "mAP": 100.0 * float(np.mean([row["ap"] for row in details])),
    }
    for rank in ranks:
        metrics[f"Rank-{rank}"] = 100.0 * float(
            np.mean([row[f"rank_{rank}"] for row in details])
        )
    return metrics, details
