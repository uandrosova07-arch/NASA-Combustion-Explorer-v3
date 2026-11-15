from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

@st.cache_data
def load_data():
    insights = pd.read_csv(ROOT / "data/raw/insights.csv")
    investigations = pd.read_csv(ROOT / "data/raw/investigations.csv")
    return insights, investigations

insights, investigations = load_data()

st.title("Curated Fire-Safety Insights")
st.caption(
    "Evidence-linked interpretations derived from PSI pages, experiment tables, "
    "NASA sources, and publication abstracts. These records are not part of the "
    "SPLADE retrieval index."
)

c1, c2, c3 = st.columns(3)
c1.metric("Insights", len(insights))
c2.metric("High confidence", int((insights["confidence"] == "high").sum()))
c3.metric("Topics", insights["topic"].nunique())

topics = sorted(insights["topic"].dropna().astype(str).unique())
selected_topics = st.multiselect("Topic", topics)

confidence = st.multiselect(
    "Confidence",
    sorted(insights["confidence"].dropna().astype(str).unique()),
    default=sorted(insights["confidence"].dropna().astype(str).unique()),
)

view = insights.copy()
if selected_topics:
    view = view[view["topic"].isin(selected_topics)]
if confidence:
    view = view[view["confidence"].isin(confidence)]

for _, row in view.iterrows():
    with st.container(border=True):
        st.markdown(f"### {row['insight_id']} · {row['topic']}")
        st.write(row["statement"])

        meta = st.columns(3)
        meta[0].write(f"**Evidence:** {row['evidence_type']}")
        meta[1].write(f"**Confidence:** {row['confidence']}")
        meta[2].write(
            f"**PSI:** {row['psi_ids'] if pd.notna(row['psi_ids']) else '—'}"
        )

        st.write(f"**Source:** {row['source_ref']}")

        if pd.notna(row.get("caveat")) and str(row["caveat"]).strip():
            st.warning(f"Caveat: {row['caveat']}")

        if pd.notna(row.get("source_url")) and str(row["source_url"]).strip():
            st.link_button("Open source", str(row["source_url"]))
