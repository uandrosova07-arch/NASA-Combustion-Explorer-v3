from __future__ import annotations

from pathlib import Path
import pandas as pd


def validate_all(data_dir: str | Path = "data/raw") -> list[str]:
    data_dir = Path(data_dir)
    inv = pd.read_csv(data_dir / "investigations.csv")
    runs = pd.read_csv(data_dir / "runs.csv")
    pubs = pd.read_csv(data_dir / "publications.csv")
    files_df = pd.read_csv(data_dir / "files.csv")
    matrix = pd.read_csv(data_dir / "test_matrix.csv")
    insights = pd.read_csv(data_dir / "insights.csv")

    errors = []
    inv_ids = set(inv["psi_id"].dropna().astype(str))

    for name, df, id_col in [
        ("runs", runs, "run_id"),
        ("publications", pubs, "pub_id"),
        ("files", files_df, "file_id"),
        ("test_matrix", matrix, "matrix_id"),
    ]:
        unknown = sorted(set(df["psi_id"].dropna().astype(str)) - inv_ids)
        if unknown:
            errors.append(f"{name}: psi_id missing from investigations: {unknown}")
        if df[id_col].isna().any():
            errors.append(f"{name}: missing values in {id_col}")
        if df[id_col].duplicated().any():
            errors.append(f"{name}: duplicate values in {id_col}")

    actual_runs = runs.groupby("psi_id").size()
    for _, row in inv.dropna(subset=["psi_id"]).iterrows():
        psi_id = str(row["psi_id"])
        if psi_id in actual_runs.index and pd.notna(row.get("run_count")):
            expected = int(float(row["run_count"]))
            actual = int(actual_runs.loc[psi_id])
            if expected != actual:
                errors.append(
                    f"{psi_id}: run_count={expected}, but runs.csv has {actual}"
                )

    # Curated insights are a presentation/evidence layer, not retrieval documents.
    if insights["insight_id"].isna().any():
        errors.append("insights: missing insight_id")
    if insights["insight_id"].duplicated().any():
        errors.append("insights: duplicate insight_id")

    import re
    linked_ids = set()
    for value in insights["psi_ids"].dropna().astype(str):
        linked_ids.update(re.findall(r"PSI-\\d+", value))
    unknown_insight_ids = sorted(linked_ids - inv_ids)
    if unknown_insight_ids:
        errors.append(
            f"insights: psi_id missing from investigations: {unknown_insight_ids}"
        )

    return errors
