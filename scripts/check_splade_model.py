from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.splade_encoder import LocalSpladeEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/splade")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    encoder = LocalSpladeEncoder(args.model_dir, device=args.device)

    docs = encoder.encode_documents(
        [
            "Microgravity flame extinction with carbon dioxide suppression.",
            "Smoke particle size measurements aboard the International Space Station.",
        ],
        batch_size=2,
    )
    query = encoder.encode_query(
        "CO2 flame extinction in microgravity"
    )

    print("Model loaded successfully.")
    print("Document vectors:", docs.shape, "nnz =", docs.nnz)
    print("Query vector:", query.shape, "nnz =", query.nnz)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
