from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Audit SPLADE chunk lengths with the exact local tokenizer."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models/splade",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=ROOT / "data/processed/chunks.jsonl",
    )
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if any chunk exceeds the token limit.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_fast=True,
    )

    chunks = load_chunks(args.chunks)
    lengths = []
    over = []

    for chunk in chunks:
        token_ids = tokenizer(
            chunk["text"],
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        n = len(token_ids)
        lengths.append(n)
        if n > args.limit:
            over.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "psi_id": chunk["psi_id"],
                    "chunk_type": chunk["chunk_type"],
                    "tokens": n,
                }
            )

    arr = np.asarray(lengths)

    print(f"chunks={len(chunks)}")
    print(f"limit={args.limit}")
    print(f"min={arr.min()}")
    print(f"median={np.percentile(arr, 50):.1f}")
    print(f"p90={np.percentile(arr, 90):.1f}")
    print(f"p95={np.percentile(arr, 95):.1f}")
    print(f"p99={np.percentile(arr, 99):.1f}")
    print(f"max={arr.max()}")
    print(f"over_limit={len(over)}")

    if over:
        print("\nLongest over-limit chunks:")
        for item in sorted(over, key=lambda x: x["tokens"], reverse=True)[:30]:
            print(
                f"{item['tokens']:4d}  "
                f"{item['chunk_id']}  "
                f"{item['chunk_type']}"
            )

    if args.strict and over:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
