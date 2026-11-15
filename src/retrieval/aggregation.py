from __future__ import annotations

from collections import defaultdict


def aggregate_by_investigation(
    chunk_results: list[dict],
    top_chunks_per_investigation: int = 3,
) -> list[dict]:
    grouped = defaultdict(list)

    for result in chunk_results:
        grouped[result["psi_id"]].append(result)

    aggregated = []

    for psi_id, rows in grouped.items():
        rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        top = rows[:top_chunks_per_investigation]

        aggregated.append(
            {
                "psi_id": psi_id,
                "score": top[0]["score"],  # MVP: max chunk score
                "matched_chunks": top,
            }
        )

    return sorted(aggregated, key=lambda x: x["score"], reverse=True)
