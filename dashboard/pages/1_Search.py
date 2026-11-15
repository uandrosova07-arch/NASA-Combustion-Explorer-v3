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

st.title("SPLADE Search")
st.caption("Neural sparse retrieval over NASA PSI combustion investigations")

index_path = ROOT / "data/indexes/splade/sparse_docs.npz"

if not index_path.exists():
    st.error(
        "SPLADE index is not built yet. Copy the complete trained model package "
        "to models/splade/ and run: python scripts/build_splade_index.py"
    )
    st.stop()


@st.cache_resource(show_spinner="Loading SPLADE model and sparse index...")
def load_engine():
    from src.retrieval.search_engine import SpladeSearchEngine

    return SpladeSearchEngine(
        model_dir=MODEL_DIR,
        index_dir=ROOT / "data/indexes/splade",
        chunks_path=ROOT / "data/processed/chunks.jsonl",
        investigations_path=ROOT / "data/raw/investigations.csv",
        device=DEVICE,
    )


engine = load_engine()

@st.cache_data
def load_insights():
    import pandas as pd
    path = ROOT / "data/raw/insights.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

insights_df = load_insights()

def related_insights(psi_id: str):
    if insights_df.empty:
        return insights_df
    import re
    return insights_df[
        insights_df["psi_ids"].fillna("").apply(
            lambda x: psi_id in re.findall(r"PSI-\d+", str(x))
        )
    ]

query = st.text_input(
    "Search NASA PSI combustion investigations",
    placeholder="e.g. flame extinction with CO2 in microgravity",
)

top_k = st.slider("Number of investigations", 3, 15, 8)

if query.strip():
    with st.spinner("Encoding query with SPLADE..."):
        results = engine.search_investigations(
            query,
            top_k_chunks=max(top_k * 4, 20),
            top_k_investigations=top_k,
        )

    if not results:
        st.info("No positive-scoring chunks were found.")

    for rank, item in enumerate(results, start=1):
        inv = item["investigation"]
        acronym = inv.get("acronym") or ""
        name = inv.get("full_name") or item["psi_id"]

        with st.container(border=True):
            st.subheader(f"{rank}. {acronym} — {name}")
            st.caption(
                f"{item['psi_id']} · SPLADE score {item['score']:.4f}"
            )

            cols = st.columns(3)
            cols[0].write(f"**Fuel:** {inv.get('fuel_phase') or '—'}")
            cols[1].write(
                f"**Topics:** {inv.get('fire_safety_topics') or '—'}"
            )
            cols[2].write(
                f"**Platform:** {inv.get('flight_platform') or '—'}"
            )

            st.write("**Why this result matched**")
            for chunk in item["matched_chunks"]:
                with st.expander(
                    f"{chunk['chunk_type']} · {chunk['score']:.4f}",
                    expanded=False,
                ):
                    st.write(chunk["text"])

            psi_url = inv.get("psi_url")
            if psi_url:
                st.link_button("Open NASA PSI", psi_url)

            linked = related_insights(item["psi_id"])
            if not linked.empty:
                with st.expander(
                    f"Related curated insights ({len(linked)})",
                    expanded=False,
                ):
                    for _, insight in linked.iterrows():
                        st.markdown(
                            f"**{insight['insight_id']} · "
                            f"{insight['confidence']} confidence**"
                        )
                        st.write(insight["statement"])
                        st.caption(
                            f"Evidence: {insight['evidence_type']} · "
                            f"Source: {insight['source_ref']}"
                        )
                        if (
                            "caveat" in insight
                            and insight["caveat"] is not None
                            and str(insight["caveat"]) != "nan"
                        ):
                            st.warning(f"Caveat: {insight['caveat']}")
