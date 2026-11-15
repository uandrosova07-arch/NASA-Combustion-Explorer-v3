from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("SPLADE_MODEL_DIR", ROOT / "models/splade"))
DEVICE = os.environ.get("SPLADE_DEVICE", "cpu")

@st.cache_resource(show_spinner="Loading SPLADE...")
def load_encoder():
    from src.retrieval.splade_encoder import LocalSpladeEncoder
    return LocalSpladeEncoder(MODEL_DIR, device=DEVICE)

st.title("SPLADE Explorer")
st.caption("Inspect the strongest vocabulary dimensions activated by a query.")


@st.cache_resource(show_spinner="Loading SPLADE...")
def load_encoder():
    from src.retrieval.splade_encoder import LocalSpladeEncoder
    return LocalSpladeEncoder(MODEL_DIR, device=DEVICE)


def load_vocab(model_dir: Path):
    vocab_txt = model_dir / "vocab.txt"
    if not vocab_txt.exists():
        return None
    return vocab_txt.read_text(encoding="utf-8").splitlines()


if not (MODEL_DIR / "splade.py").exists():
    st.error("Copy the complete trained SPLADE package to models/splade/.")
    st.stop()

encoder = load_encoder()
vocab = load_vocab(MODEL_DIR)

query = st.text_input(
    "Query",
    placeholder="spacecraft material flammability at elevated oxygen",
)

top_n = st.slider("Top activated terms", 5, 40, 20)

if query.strip():
    q = encoder.encode_query(query)
    row = q.getrow(0)

    pairs = sorted(
        zip(row.indices.tolist(), row.data.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    table = []
    for token_id, weight in pairs:
        token = (
            vocab[token_id]
            if vocab is not None and token_id < len(vocab)
            else f"token_id:{token_id}"
        )
        table.append(
            {
                "token_id": token_id,
                "token": token,
                "weight": weight,
            }
        )

    st.dataframe(table, use_container_width=True, hide_index=True)
