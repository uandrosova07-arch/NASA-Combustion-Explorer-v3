from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.splade_encoder import LocalSpladeEncoder
from src.retrieval.splade_index import save_index


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

            for field in ("chunk_id", "psi_id", "chunk_type", "text"):
                if not item.get(field):
                    raise ValueError(
                        f"Chunk at line {line_no} is missing required field {field!r}"
                    )
            chunks.append(item)

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode NASA PSI chunks with the trained local SPLADE model."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=ROOT / "data/processed/chunks.jsonl",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models/splade",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT / "data/indexes/splade",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    print(f"Chunks: {len(chunks)}")
    print(f"Device: {args.device}")
    print(f"Model directory: {args.model_dir}")

    encoder = LocalSpladeEncoder(
        model_dir=args.model_dir,
        device=args.device,
    )

    matrix = encoder.encode_documents(
        texts,
        batch_size=args.batch_size,
    )

    save_index(
        matrix=matrix,
        chunk_ids=chunk_ids,
        index_dir=args.index_dir,
    )

    print(f"Index shape: {matrix.shape}")
    print(f"Non-zero weights: {matrix.nnz}")
    print(f"Saved to: {args.index_dir}")


if __name__ == "__main__":
    main()
