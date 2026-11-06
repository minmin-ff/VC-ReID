from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .common import atomic_json_dump


def load_result(path: str | Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def final_round(result: Dict[str, object], round_index: int = 4) -> Dict[str, object]:
    for row in result["rounds"]:
        if int(row["round"]) == round_index:
            return row
    raise ValueError(f"Result does not contain round {round_index}")


def summarize(paths: List[str], output: str) -> None:
    rows = []
    for path in paths:
        result = load_result(path)
        configuration = result.get("configuration", {})
        for round_row in result["rounds"]:
            rows.append(
                {
                    "source": str(path),
                    "mode": configuration.get("mode", "unknown"),
                    "seed": configuration.get("seed", "unknown"),
                    "alpha": configuration.get("alpha", ""),
                    "decay": configuration.get("decay", ""),
                    "anchor_weight": configuration.get("anchor_weight", ""),
                    "answer_error_rate": configuration.get("answer_error_rate", ""),
                    "round": round_row["round"],
                    **round_row["metrics"],
                }
            )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {destination}")


def paired_bootstrap(
    left_path: str,
    right_path: str,
    round_index: int,
    replicates: int,
    seed: int,
) -> Dict[str, object]:
    left = final_round(load_result(left_path), round_index)
    right = final_round(load_result(right_path), round_index)
    left_by_id = {row["query_id"]: row for row in left["per_query"]}
    right_by_id = {row["query_id"]: row for row in right["per_query"]}
    common = sorted(set(left_by_id) & set(right_by_id))
    if not common:
        raise ValueError("No common query IDs")
    left_ap = np.asarray([left_by_id[key]["ap"] for key in common], dtype=float)
    right_ap = np.asarray([right_by_id[key]["ap"] for key in common], dtype=float)
    left_r1 = np.asarray([left_by_id[key]["rank_1"] for key in common], dtype=float)
    right_r1 = np.asarray([right_by_id[key]["rank_1"] for key in common], dtype=float)
    rng = np.random.default_rng(seed)
    map_differences = np.empty(replicates, dtype=float)
    rank1_differences = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = rng.integers(0, len(common), size=len(common))
        map_differences[index] = 100 * np.mean(right_ap[sample] - left_ap[sample])
        rank1_differences[index] = 100 * np.mean(right_r1[sample] - left_r1[sample])

    def describe(values: np.ndarray, observed: float) -> Dict[str, float]:
        return {
            "observed_difference": observed,
            "ci_2.5": float(np.quantile(values, 0.025)),
            "ci_97.5": float(np.quantile(values, 0.975)),
            "two_sided_p": float(
                min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0)))
            ),
        }

    return {
        "left": left_path,
        "right": right_path,
        "round": round_index,
        "common_queries": len(common),
        "replicates": replicates,
        "seed": seed,
        "mAP": describe(map_differences, 100 * float(np.mean(right_ap - left_ap))),
        "Rank-1": describe(
            rank1_differences, 100 * float(np.mean(right_r1 - left_r1))
        ),
    }


def seed_statistics(paths: List[str], round_index: int) -> Dict[str, object]:
    metrics: Dict[str, List[float]] = {}
    for path in paths:
        row = final_round(load_result(path), round_index)["metrics"]
        for name in ("mAP", "Rank-1", "Rank-5", "Rank-10"):
            metrics.setdefault(name, []).append(float(row[name]))
    return {
        "round": round_index,
        "runs": len(paths),
        "sources": paths,
        "metrics": {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "values": values,
            }
            for name, values in metrics.items()
        },
    }


def merge_shards(paths: List[str]) -> Dict[str, object]:
    shards = [load_result(path) for path in paths]
    if not shards:
        raise ValueError("No shard results supplied")
    expected_universe = int(shards[0].get("query_universe_count", 0))
    ranges = []
    interactions: Dict[str, object] = {}
    for path, shard in zip(paths, shards):
        if int(shard.get("query_universe_count", 0)) != expected_universe:
            raise ValueError(f"Mismatched query universe in {path}")
        query_range = shard.get("query_range")
        if not isinstance(query_range, dict):
            raise ValueError(f"Missing query_range in {path}")
        ranges.append((int(query_range["start"]), int(query_range["end"]), path))
        for query_id, records in shard.get("interactions", {}).items():
            if query_id in interactions:
                raise ValueError(f"Duplicate query ID across shards: {query_id}")
            interactions[query_id] = records
    ranges.sort()
    cursor = 0
    for start, end, path in ranges:
        if start != cursor:
            raise ValueError(
                f"Shard ranges are not contiguous at {cursor}: got [{start}, {end}) in {path}"
            )
        cursor = end
    if cursor != expected_universe:
        raise ValueError(f"Shard coverage ends at {cursor}, expected {expected_universe}")

    merged_rounds = []
    round_indices = [int(row["round"]) for row in shards[0]["rounds"]]
    for round_index in round_indices:
        details = []
        seen = set()
        for path, shard in zip(paths, shards):
            matches = [
                row for row in shard["rounds"] if int(row["round"]) == round_index
            ]
            if len(matches) != 1:
                raise ValueError(f"Round {round_index} missing or duplicated in {path}")
            for row in matches[0]["per_query"]:
                query_id = row["query_id"]
                if query_id in seen:
                    raise ValueError(
                        f"Duplicate query ID in round {round_index}: {query_id}"
                    )
                seen.add(query_id)
                details.append(row)
        metrics = {
            "valid_queries": len(details),
            "mAP": 100.0 * float(np.mean([row["ap"] for row in details])),
        }
        for rank in (1, 5, 10):
            metrics[f"Rank-{rank}"] = 100.0 * float(
                np.mean([row[f"rank_{rank}"] for row in details])
            )
        merged_rounds.append(
            {"round": round_index, "metrics": metrics, "per_query": details}
        )

    timing_keys = set().union(
        *(set(shard.get("timing", {})) for shard in shards)
    )
    timing = {
        key: float(sum(float(shard.get("timing", {}).get(key, 0.0)) for shard in shards))
        for key in timing_keys
        if key not in {"seconds_per_query", "wall_seconds"}
    }
    total_wall = float(
        sum(float(shard.get("timing", {}).get("wall_seconds", 0.0)) for shard in shards)
    )
    timing.update(
        {
            "sum_shard_wall_seconds": total_wall,
            "seconds_per_query": total_wall / expected_universe,
        }
    )
    return {
        "experiment": "interactive_signed_feedback_merged_shards",
        "configuration": shards[0].get("configuration", {}),
        "query_count": expected_universe,
        "query_universe_count": expected_universe,
        "query_range": {"start": 0, "end": expected_universe},
        "rounds": merged_rounds,
        "interactions": interactions,
        "timing": timing,
        "resources": {
            "shards": [shard.get("resources", {}) for shard in shards],
            "source_files": paths,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Dialog-ReID experiment results")
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--inputs", nargs="+", required=True)
    summary.add_argument("--output", required=True)
    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--left", required=True)
    bootstrap.add_argument("--right", required=True)
    bootstrap.add_argument("--round", type=int, default=4)
    bootstrap.add_argument("--replicates", type=int, default=10000)
    bootstrap.add_argument("--seed", type=int, default=2026)
    bootstrap.add_argument("--output", required=True)
    seeds = subparsers.add_parser("seeds")
    seeds.add_argument("--inputs", nargs="+", required=True)
    seeds.add_argument("--round", type=int, default=4)
    seeds.add_argument("--output", required=True)
    merge = subparsers.add_parser("merge-shards")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "summary":
        summarize(args.inputs, args.output)
    elif args.command == "bootstrap":
        atomic_json_dump(
            paired_bootstrap(
                args.left,
                args.right,
                args.round,
                args.replicates,
                args.seed,
            ),
            args.output,
        )
    elif args.command == "seeds":
        atomic_json_dump(seed_statistics(args.inputs, args.round), args.output)
    else:
        atomic_json_dump(merge_shards(args.inputs), args.output)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
