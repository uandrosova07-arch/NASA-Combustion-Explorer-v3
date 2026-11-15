from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.splade_encoder import LocalSpladeEncoder


QUERIES = [
    "flame extinction with carbon dioxide in microgravity",
    "solid material flammability at elevated oxygen concentration",
    "smoke particle size measurements on the ISS",
    "large scale spacecraft fire experiments",
    "cool flames of gaseous fuels",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/splade")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    encoder = LocalSpladeEncoder(args.model_dir, args.device)

    # warm-up
    encoder.encode_query(QUERIES[0])

    latencies = []
    for _ in range(args.rounds):
        for query in QUERIES:
            start = time.perf_counter()
            encoder.encode_query(query)
            latencies.append(time.perf_counter() - start)

    print(f"device={args.device}")
    print(f"queries={len(latencies)}")
    print(f"mean_seconds={statistics.mean(latencies):.4f}")
    print(f"median_seconds={statistics.median(latencies):.4f}")
    print(f"min_seconds={min(latencies):.4f}")
    print(f"max_seconds={max(latencies):.4f}")


if __name__ == "__main__":
    main()
