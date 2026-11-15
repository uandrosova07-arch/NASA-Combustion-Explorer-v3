from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

@st.cache_data
def load_runs():
    return pd.read_csv(ROOT / "data/raw/runs.csv")

@st.cache_data
def load_investigations():
    return pd.read_csv(ROOT / "data/raw/investigations.csv")

runs = load_runs()
inv = load_investigations()
name_map = (
    inv.dropna(subset=["psi_id"])
    .drop_duplicates("psi_id")
    .set_index("psi_id")["full_name"]
    .to_dict()
)

st.title("Experimental Runs")
st.caption("Structured per-run conditions extracted from NASA PSI experiment tables.")

psi_options = sorted(runs["psi_id"].dropna().unique())
selected = st.selectbox(
    "Investigation",
    psi_options,
    format_func=lambda x: f"{x} — {name_map.get(x, '')}",
)

view = runs[runs["psi_id"] == selected].copy()

c1, c2, c3 = st.columns(3)
c1.metric("Runs", len(view))
c2.metric("Fuels", view["fuel"].nunique(dropna=True))
c3.metric("Rows with O₂", int(view["o2_pct"].notna().sum()))

fuel_options = sorted(view["fuel"].dropna().astype(str).unique())
fuel = st.multiselect("Fuel", fuel_options)
if fuel:
    view = view[view["fuel"].isin(fuel)]

columns = [
    "run_id", "run_label", "fuel", "flame_type", "o2_pct",
    "pressure_atm", "flow_speed_cm_s", "diluent", "suppressant",
    "gravity", "outcome", "source_file",
]
columns = [c for c in columns if c in view.columns]
st.dataframe(view[columns], use_container_width=True, hide_index=True)
