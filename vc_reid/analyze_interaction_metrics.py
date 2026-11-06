#!/usr/bin/env python3
"""Compute interaction-first and early-stop metrics from a merged result.

The input must contain round-level ``per_query`` records for rounds 0..T.  The
analysis is restricted to the evaluator's valid-query intersection, so every
reported denominator matches official MARS mAP/CMC evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def indexed_rounds(payload: dict) -> list[dict[str, dict]]:
    rounds = sorted(payload["rounds"], key=lambda item: int(item["round"]))
    if [int(item["round"]) for item in rounds] != list(range(len(rounds))):
        raise ValueError("Rounds must be contiguous and start at zero")
    indexed = []
    for item in rounds:
        records = {str(row["query_id"]): row for row in item["per_query"]}
        if len(records) != len(item["per_query"]):
            raise ValueError(f"Duplicate query ID in round {item['round']}")
        indexed.append(records)
    valid_ids = set(indexed[0])
    if any(set(records) != valid_ids for records in indexed[1:]):
        raise ValueError("Per-round valid-query sets differ")
    return indexed


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rounds = indexed_rounds(payload)
    query_ids = sorted(rounds[0])
    last = len(rounds) - 1

    rows = []
    first_success_histogram: Counter[int | str] = Counter()
    for query_id in query_ids:
        rank1 = [int(round_data[query_id]["rank_1"]) for round_data in rounds]
        ap = [float(round_data[query_id]["ap"]) for round_data in rounds]
        first_success = next((index for index, value in enumerate(rank1) if value), None)
        first_success_histogram[first_success if first_success is not None else "unresolved"] += 1

        # Deployment accounting: an initially correct result costs one
        # confirmation; an initial failure consumes updates until first success,
        # capped at the available round budget.
        active_rounds = 1 if rank1[0] else (first_success or last)
        stop_round = first_success if first_success is not None else last
        rows.append(
            {
                "query_id": query_id,
                "initial_rank1": rank1[0],
                "final_rank1": rank1[-1],
                "first_success_round": first_success,
                "active_rounds_with_confirmation": active_rounds,
                "early_stop_round": stop_round,
                "early_stop_ap": ap[stop_round],
            }
        )

    total = len(rows)
    initially_correct = sum(row["initial_rank1"] for row in rows)
    initially_wrong = total - initially_correct
    repaired = sum(
        not row["initial_rank1"] and row["final_rank1"] for row in rows
    )
    damaged = sum(
        row["initial_rank1"] and not row["final_rank1"] for row in rows
    )
    ever_repaired_rows = [
        row
        for row in rows
        if not row["initial_rank1"] and row["first_success_round"] is not None
    ]
    early_stop_correct = sum(row["first_success_round"] is not None for row in rows)
    early_stop_rounds = sum(row["active_rounds_with_confirmation"] for row in rows)
    fixed_rounds = total * last

    initial_r1 = 100.0 * initially_correct / total
    final_correct = sum(row["final_rank1"] for row in rows)
    final_r1 = 100.0 * final_correct / total
    result = {
        "source": str(args.input),
        "valid_queries": total,
        "round_budget": last,
        "counts": {
            "initially_correct": initially_correct,
            "initially_wrong": initially_wrong,
            "repaired_at_final_round": repaired,
            "damaged_at_final_round": damaged,
            "ever_repaired_by_budget": len(ever_repaired_rows),
            "unresolved_by_budget": first_success_histogram["unresolved"],
        },
        "interaction_metrics_percent": {
            "initial_rank1": initial_r1,
            "final_rank1": final_r1,
            "repair_rate": 100.0 * repaired / initially_wrong,
            "damage_rate": 100.0 * damaged / initially_correct,
            "normalized_error_reduction": 100.0
            * (final_r1 - initial_r1)
            / (100.0 - initial_r1),
        },
        "rounds_to_success": {
            "definition": "first post-update round with correct Top-1",
            "histogram_all_valid_queries": {
                str(key): first_success_histogram[key]
                for key in [0, 1, 2, 3, 4, "unresolved"]
            },
            "mean_among_initial_failures_eventually_repaired": mean(
                row["first_success_round"] for row in ever_repaired_rows
            ),
            "mean_including_round0_success_among_eventual_successes": mean(
                row["first_success_round"]
                for row in rows
                if row["first_success_round"] is not None
            ),
        },
        "early_stop_oracle": {
            "policy": (
                "one confirmation for an initially correct Top-1; otherwise stop "
                "at first correct post-update Top-1, or at the round budget"
            ),
            "rank1_percent": 100.0 * early_stop_correct / total,
            "mAP_percent": 100.0 * mean(row["early_stop_ap"] for row in rows),
            "damage_rate_percent": 0.0,
            "total_active_rounds": early_stop_rounds,
            "fixed_protocol_rounds": fixed_rounds,
            "mean_active_rounds_per_query": early_stop_rounds / total,
            "round_reduction_percent": 100.0 * (fixed_rounds - early_stop_rounds) / fixed_rounds,
            "queries_requiring_update_percent": 100.0 * initially_wrong / total,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
