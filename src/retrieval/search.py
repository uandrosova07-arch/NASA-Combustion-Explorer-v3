from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def sparse_top_k(
    doc_matrix: csr_matrix,
    query_vector: csr_matrix,
    k: int = 10,
) -> list[tuple[int, float]]:
    """
    Return (row_index, score) pairs.

    For the current NASA PSI corpus this simple sparse matrix multiplication is
    enough. A dedicated inverted-index service is unnecessary at this scale.
    """
    if query_vector.shape[0] != 1:
        raise ValueError("query_vector must contain exactly one query.")
    if doc_matrix.shape[1] != query_vector.shape[1]:
        raise ValueError("Document and query vocabulary dimensions differ.")

    scores = (doc_matrix @ query_vector.T).toarray().ravel()

    if scores.size == 0:
        return []

    k = min(k, scores.size)
    candidate_idx = np.argpartition(scores, -k)[-k:]
    candidate_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]

    return [(int(i), float(scores[i])) for i in candidate_idx if scores[i] > 0]
