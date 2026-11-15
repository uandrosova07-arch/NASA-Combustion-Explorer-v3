from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.sparse import csr_matrix


@contextmanager
def _working_directory(path: Path):
    """
    Some self-contained model packages use relative paths internally.
    Initialize the user's Splade class from inside its own directory.
    """
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def torch_dense_to_csr(tensor: torch.Tensor) -> csr_matrix:
    """
    Convert SPLADE output [batch, vocab] to scipy CSR without changing weights.

    The provided model returns CPU FP32 torch.Tensor vectors with non-negative
    weights and exact zeros for inactive vocabulary coordinates.
    """
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected a 2-D SPLADE tensor [batch, vocab], got {tuple(tensor.shape)}"
        )

    array = tensor.detach().to("cpu").numpy()
    return csr_matrix(array, dtype=np.float32)


class LocalSpladeEncoder:
    """
    Adapter for the user's self-contained SPLADE-max package.

    Expected model directory (based on the supplied README):
        models/splade/
            splade.py
            model.pt
            config.json       (or Windows may display it simply as 'config')
            tokenizer.json
            tokenizer_config.json
            special_tokens_map.json
            vocab.txt
            ...

    The package must expose:
        from splade import Splade

        model = Splade(device="cpu")
        model.encode_queries([...], batch_size=...)
        model.encode_documents([...], batch_size=...)

    Output dimensionality is expected to be 30,522 for the supplied checkpoint.
    """

    EXPECTED_VOCAB_SIZE = 30_522
    MAX_TOKENS = 256

    def __init__(
        self,
        model_dir: str | Path = "models/splade",
        device: str = "cpu",
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.device = device

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"SPLADE model directory does not exist: {self.model_dir}"
            )

        splade_py = self.model_dir / "splade.py"
        if not splade_py.exists():
            raise FileNotFoundError(
                f"Expected {splade_py}. Copy the complete trained SPLADE package "
                "including splade.py and model/tokenizer files."
            )

        self._model = self._load_model()

    def _load_model(self):
        model_dir_str = str(self.model_dir)
        if model_dir_str not in sys.path:
            sys.path.insert(0, model_dir_str)

        # Avoid accidentally reusing a different module named `splade`.
        existing = sys.modules.get("splade")
        if existing is not None:
            existing_file = Path(getattr(existing, "__file__", "")).resolve()
            if existing_file.parent != self.model_dir:
                del sys.modules["splade"]

        with _working_directory(self.model_dir):
            module = importlib.import_module("splade")
            Splade = getattr(module, "Splade")
            model = Splade(device=self.device)

        return model

    def _check_shape(self, matrix: csr_matrix) -> None:
        if matrix.shape[1] != self.EXPECTED_VOCAB_SIZE:
            raise ValueError(
                "Unexpected SPLADE vocabulary dimension: "
                f"{matrix.shape[1]} (expected {self.EXPECTED_VOCAB_SIZE}). "
                "Check that the checkpoint and tokenizer belong together."
            )

    def encode_documents(
        self,
        texts: Iterable[str],
        batch_size: int = 8,
    ) -> csr_matrix:
        texts = list(texts)
        if not texts:
            return csr_matrix((0, self.EXPECTED_VOCAB_SIZE), dtype=np.float32)

        vectors = self._model.encode_documents(texts, batch_size=batch_size)
        matrix = torch_dense_to_csr(vectors)
        self._check_shape(matrix)
        return matrix

    def encode_query(self, text: str) -> csr_matrix:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Query must be a non-empty string.")

        vectors = self._model.encode_queries([text])
        matrix = torch_dense_to_csr(vectors)
        self._check_shape(matrix)
        return matrix

    def encode_queries(
        self,
        texts: Iterable[str],
        batch_size: int = 8,
    ) -> csr_matrix:
        texts = list(texts)
        if not texts:
            return csr_matrix((0, self.EXPECTED_VOCAB_SIZE), dtype=np.float32)

        vectors = self._model.encode_queries(texts, batch_size=batch_size)
        matrix = torch_dense_to_csr(vectors)
        self._check_shape(matrix)
        return matrix
