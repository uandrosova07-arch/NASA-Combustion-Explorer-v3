from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .aggregation import aggregate_by_investigation
from .search import sparse_top_k
from .splade_encoder import LocalSpladeEncoder
from .splade_index import load_index


class SpladeSearchEngine:
    def __init__(
        self,
        model_dir: str | Path,
        index_dir: str | Path,
        chunks_path: str | Path,
        investigations_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self.encoder = LocalSpladeEncoder(model_dir=model_dir, device=device)
        self.doc_matrix, self.chunk_ids = load_index(index_dir)

        self.chunks = {}
        with Path(chunks_path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.chunks[item["chunk_id"]] = item

        if self.doc_matrix.shape[0] != len(self.chunk_ids):
            raise ValueError(
                "Index row count and chunk_ids length do not match."
            )

        missing = [cid for cid in self.chunk_ids if cid not in self.chunks]
        if missing:
            raise ValueError(
                f"{len(missing)} index chunk IDs are absent from chunks.jsonl."
            )

        self.investigations = pd.read_csv(investigations_path)
        self.investigations_by_id = (
            self.investigations[self.investigations["psi_id"].notna()]
            .drop_duplicates("psi_id")
            .set_index("psi_id")
        )

    def search_chunks(self, query: str, top_k: int = 20) -> list[dict]:
        q = self.encoder.encode_query(query)
        scored = sparse_top_k(self.doc_matrix, q, k=top_k)

        results = []
        for row_idx, score in scored:
            chunk_id = self.chunk_ids[row_idx]
            chunk = self.chunks[chunk_id]
            results.append(
                {
                    "chunk_id": chunk_id,
                    "psi_id": chunk["psi_id"],
                    "chunk_type": chunk["chunk_type"],
                    "score": score,
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                }
            )

        return results

    def search_investigations(
        self,
        query: str,
        top_k_chunks: int = 30,
        top_k_investigations: int = 10,
    ) -> list[dict]:
        chunk_results = self.search_chunks(query, top_k=top_k_chunks)
        grouped = aggregate_by_investigation(chunk_results)

        output = []
        for result in grouped[:top_k_investigations]:
            psi_id = result["psi_id"]
            meta = {}

            if psi_id in self.investigations_by_id.index:
                row = self.investigations_by_id.loc[psi_id]
                meta = row.where(pd.notna(row), None).to_dict()

            output.append(
                {
                    **result,
                    "investigation": meta,
                }
            )

        return output
