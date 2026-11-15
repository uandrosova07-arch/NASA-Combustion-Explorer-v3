from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz


def save_index(
    matrix: csr_matrix,
    chunk_ids: list[str],
    index_dir: str | Path,
) -> None:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    save_npz(index_dir / "sparse_docs.npz", matrix)
    np.save(index_dir / "chunk_ids.npy", np.asarray(chunk_ids, dtype=object))

    metadata = {
        "num_documents": int(matrix.shape[0]),
        "vocab_size": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
    }
    (index_dir / "index_meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_index(index_dir: str | Path):
    index_dir = Path(index_dir)
    matrix = load_npz(index_dir / "sparse_docs.npz").tocsr()
    chunk_ids = np.load(index_dir / "chunk_ids.npy", allow_pickle=True).tolist()
    return matrix, chunk_ids
