from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="NASA Combustion Explorer",
    page_icon="🔥",
    layout="wide",
)

st.title("NASA Combustion Explorer")
st.caption("SPLADE-powered retrieval over NASA PSI combustion investigations")

@st.cache_data
def load_data():
    investigations = pd.read_csv(ROOT / "data/raw/investigations.csv")
    chunks = [
        json.loads(line)
        for line in (ROOT / "data/processed/chunks.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    return investigations, chunks

investigations, chunks = load_data()

indexable = investigations[
    investigations["psi_id"].notna()
    & investigations["row_role"].isin(
        ["flight_investigation", "ground_investigation"]
    )
    & investigations["in_psi_search"].astype(str).str.lower().eq("yes")
]

c1, c2, c3 = st.columns(3)
c1.metric("PSI investigations in retrieval corpus", indexable["psi_id"].nunique())
c2.metric("Current chunks", len(chunks))
c3.metric("Source CSV rows", len(investigations))

st.info(
      "The data/chunk pipeline is active. SPLADE search will be enabled after "
    "the trained checkpoint format and pooling are connected. It can take some time! "
)

st.subheader("Current retrieval corpus")

role = st.selectbox(
    "Investigation scope",
    ["All", "flight_investigation", "ground_investigation"],
)

view = indexable.copy()
if role != "All":
    view = view[view["row_role"] == role]

columns = [
    "psi_id", "acronym", "full_name", "row_role",
    "fuel_phase", "fire_safety_topics", "flight_platform",
]
st.dataframe(view[columns], use_container_width=True, hide_index=True)
