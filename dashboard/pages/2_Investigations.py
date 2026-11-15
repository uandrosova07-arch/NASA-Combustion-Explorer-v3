from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

@st.cache_data
def load_investigations():
    return pd.read_csv(ROOT / "data/raw/investigations.csv")

df = load_investigations()
df = df[
    df["psi_id"].notna()
    & df["row_role"].isin(["flight_investigation", "ground_investigation"])
    & df["in_psi_search"].astype(str).str.lower().eq("yes")
].copy()

st.title("Investigation Catalog")

role = st.multiselect(
    "Investigation type",
    ["flight_investigation", "ground_investigation"],
    default=["flight_investigation", "ground_investigation"],
)
if role:
    df = df[df["row_role"].isin(role)]

families = sorted(df["family"].dropna().astype(str).unique())
selected_family = st.selectbox("Family", ["All"] + families)
if selected_family != "All":
    df = df[df["family"] == selected_family]

st.dataframe(
    df[
        [
            "psi_id", "acronym", "full_name", "family", "fuel_phase",
            "fire_safety_topics", "flight_platform", "psi_url"
        ]
    ],
    use_container_width=True,
    hide_index=True,
)
