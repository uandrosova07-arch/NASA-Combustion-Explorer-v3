from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.search_engine import SpladeSearchEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/splade")
    args = parser.parse_args()

    engine = SpladeSearchEngine(
        model_dir=args.model_dir,
        index_dir=ROOT / "data/indexes/splade",
        chunks_path=ROOT / "data/processed/chunks.jsonl",
        investigations_path=ROOT / "data/raw/investigations.csv",
        device=args.device,
    )

    results = engine.search_investigations(
        args.query,
        top_k_investigations=args.top_k,
    )

    for rank, item in enumerate(results, start=1):
        inv = item["investigation"]
        print(
            f"{rank}. {item['psi_id']} | "
            f"{inv.get('acronym') or ''} | "
            f"{inv.get('full_name') or ''} | "
            f"score={item['score']:.4f}"
        )
        for chunk in item["matched_chunks"]:
            print(
                f"   - {chunk['chunk_type']}: "
                f"{chunk['score']:.4f}"
            )


if __name__ == "__main__":
    main()
