from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd

INDEXABLE_ROLES = {"flight_investigation", "ground_investigation"}
EMPTY_LIKE = {"", "nan", "none", "n/a", "na", "unknown", "null", "not stated"}

INV_CHUNK_SPECS = {
    "investigation_overview": [
        ("Investigation", "full_name"),
        ("Acronym", "acronym"),
        ("Official PSI title", "official_psi_title"),
        ("Investigation type", "psi_type"),
        ("Research area", "psi_research_area"),
        ("Sub-research area", "sub_research_area"),
        ("Family", "family"),
        ("Fuel phase", "fuel_phase"),
        ("Fire safety topics", "fire_safety_topics"),
        ("Flight platform", "flight_platform"),
        ("PSI keywords", "psi_keywords"),
    ],
    # Keep the full curated objective by itself. If it is absent, the builder
    # falls back to the shorter PSI-search excerpt.
    "investigation_objective": [
        ("Objective", "objective"),
    ],
    "investigation_hypothesis": [
        ("Hypothesis", "hypothesis"),
    ],
    # Separate setup from conditions/approach so scientific detail is not
    # truncated by the model's 256-token limit.
    "investigation_setup": [
        ("Fuel / samples", "fuel_samples"),
        ("Hardware / facility", "hardware_facility"),
    ],
    "investigation_conditions": [
        ("Test conditions", "test_conditions"),
    ],
    "investigation_approach": [
        ("PSI approach", "psi_approach_excerpt"),
    ],
    "investigation_evidence": [
        ("Measurements", "measurements"),
        ("Key findings", "key_findings"),
    ],
    "investigation_impacts": [
        ("Space benefits", "space_benefits"),
        ("Earth benefits", "earth_benefits"),
    ],
}

RUN_COLUMNS = [
    ("Run", "run_id", "text"),
    ("Label", "run_label", "text"),
    ("Flame type", "flame_type", "text"),
    ("Fuel", "fuel", "text"),
    ("Fuel description", "fuel_label_raw", "text"),
    ("Geometry", "fuel_geometry", "text"),
    ("Oxygen pct", "o2_pct", "number"),
    ("Pressure atm", "pressure_atm", "number"),
    ("Flow speed cm/s", "flow_speed_cm_s", "number"),
    ("Flow direction", "flow_direction", "text"),
    ("Diluent", "diluent", "text"),
    ("Suppressant", "suppressant", "text"),
    ("Suppressant mole fraction", "suppressant_level_mole_fraction", "number"),
    ("Gravity", "gravity", "text"),
    ("Fuel mole fraction", "fuel_mole_fraction", "number"),
    ("Hydrocarbon flow mg/s", "hc_flow_mg_s", "number"),
    ("O2 flow mg/s", "o2_flow_mg_s", "number"),
    ("N2 flow mg/s", "n2_flow_mg_s", "number"),
    ("Droplet initial diameter mm", "droplet_d0_mm", "number"),
    ("Extinction diameter mm", "extinction_diameter_mm", "number"),
    ("Fuel thickness mm", "fuel_thickness_mm", "number"),
    ("Fuel width cm", "fuel_width_cm", "number"),
    ("Fuel diameter mm", "fuel_diameter_mm", "number"),
    ("Nozzle diameter mm", "nozzle_d_mm", "number"),
    ("Burn time s", "burn_time_s", "number"),
    ("Cool flame recorded", "cool_flame_recorded", "text"),
    ("Hot ignition", "hot_ignition", "text"),
    ("Outcome", "outcome", "text"),
    ("End mode", "end_mode", "text"),
    ("Notes", "notes", "text"),
]

MATRIX_COLUMNS = [
    ("Matrix row", "matrix_id"), ("Kind", "row_kind"), ("Label", "label"),
    ("Date", "test_date"), ("Day code", "test_day_code"), ("Test type", "test_type"),
    ("Hardware", "hardware"), ("Fuel", "fuel"), ("Fuel description", "fuel_label_raw"),
    ("Fuel concentration", "fuel_conc_raw"), ("Oxygen", "o2_pct_raw"),
    ("Atmosphere", "atmosphere_raw"), ("Pressure", "pressure_raw"),
    ("Flow", "flow_raw"), ("Flames ignited", "flames_ignited"),
    ("Other", "other_raw"), ("Notes", "notes"),
]


def clean(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = " ".join(str(value).strip().split())
    if value.lower() in EMPTY_LIKE or value.lower().startswith("not stated"):
        return None
    return value


def fmt_number(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        x = float(value)
        if math.isfinite(x):
            return f"{x:.4g}"
    except Exception:
        pass
    return clean(value)


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def indexable_investigations(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["psi_id"].notna()
        & df["row_role"].isin(INDEXABLE_ROLES)
        & df["in_psi_search"].astype(str).str.strip().str.lower().eq("yes")
    ].copy()


def build_investigation_chunks(df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    selected = indexable_investigations(df)
    chunks: list[dict] = []

    for _, row in selected.iterrows():
        psi_id = clean(row.get("psi_id"))
        for chunk_type, fields in INV_CHUNK_SPECS.items():
            parts, source_fields, seen = [], [], set()

            # Prefer the fully curated objective. Use the shorter PSI search
            # excerpt only as a fallback when the objective field is empty.
            if chunk_type == "investigation_objective":
                objective = clean(row.get("objective"))
                if objective:
                    parts.append(f"Objective: {objective}")
                    source_fields.append("objective")
                    fields_to_read = []
                else:
                    fields_to_read = [("PSI objective", "psi_objective_excerpt")]
            else:
                fields_to_read = fields

            for label, column in fields_to_read:
                value = clean(row.get(column))
                if value and value not in seen:
                    parts.append(f"{label}: {value}")
                    source_fields.append(column)
                    seen.add(value)

            text = " ".join(parts)
            if len(text) < 40:
                continue

            chunks.append({
                "chunk_id": f"{psi_id}__{chunk_type.replace('investigation_', '')}",
                "psi_id": psi_id,
                "run_id": None,
                "matrix_id": None,
                "pub_id": None,
                "chunk_type": chunk_type,
                "text": text,
                "source_fields": source_fields,
                "metadata": {
                    "acronym": clean(row.get("acronym")),
                    "full_name": clean(row.get("full_name")),
                    "row_role": clean(row.get("row_role")),
                    "family": clean(row.get("family")),
                    "fuel_phase": clean(row.get("fuel_phase")),
                    "fire_safety_topics": clean(row.get("fire_safety_topics")),
                    "flight_platform": clean(row.get("flight_platform")),
                    "sub_research_area": clean(row.get("sub_research_area")),
                    "psi_url": clean(row.get("psi_url")),
                    "doi": clean(row.get("doi")),
                    "page_version": clean(row.get("page_version")),
                },
            })

    return chunks, selected


def make_inv_lookup(selected: pd.DataFrame) -> dict[str, dict]:
    lookup = {}
    for _, row in selected.iterrows():
        lookup[str(row["psi_id"])] = {
            "acronym": clean(row.get("acronym")),
            "full_name": clean(row.get("full_name")),
            "psi_url": clean(row.get("psi_url")),
        }
    return lookup


def build_run_chunks(runs: pd.DataFrame, lookup: dict[str, dict]) -> list[dict]:
    chunks = []
    for _, row in runs.iterrows():
        psi_id = clean(row.get("psi_id"))
        run_id = clean(row.get("run_id"))
        inv = lookup.get(psi_id, {})

        parts = []
        title = inv.get("full_name") or inv.get("acronym")
        if title:
            parts.append(f"Investigation: {title}")

        source_fields = []
        for label, column, kind in RUN_COLUMNS:
            value = fmt_number(row.get(column)) if kind == "number" else clean(row.get(column))
            if value:
                parts.append(f"{label}: {value}")
                source_fields.append(column)

        chunks.append({
            "chunk_id": f"{psi_id}__run__{run_id}",
            "psi_id": psi_id,
            "run_id": run_id,
            "matrix_id": None,
            "pub_id": None,
            "chunk_type": "run",
            "text": " ".join(parts),
            "source_fields": source_fields,
            "metadata": {
                "acronym": inv.get("acronym"),
                "full_name": inv.get("full_name"),
                "fuel": clean(row.get("fuel")),
                "o2_pct": None if pd.isna(row.get("o2_pct")) else float(row.get("o2_pct")),
                "pressure_atm": None if pd.isna(row.get("pressure_atm")) else float(row.get("pressure_atm")),
                "flow_speed_cm_s": None if pd.isna(row.get("flow_speed_cm_s")) else float(row.get("flow_speed_cm_s")),
                "gravity": clean(row.get("gravity")),
                "source_file": clean(row.get("source_file")),
                "psi_url": inv.get("psi_url"),
            },
        })
    return chunks


def build_matrix_chunks(matrix: pd.DataFrame, lookup: dict[str, dict]) -> list[dict]:
    chunks = []
    for _, row in matrix.iterrows():
        psi_id = clean(row.get("psi_id"))
        matrix_id = clean(row.get("matrix_id"))
        inv = lookup.get(psi_id, {})

        parts = []
        title = inv.get("full_name") or inv.get("acronym")
        if title:
            parts.append(f"Investigation: {title}")

        source_fields = []
        for label, column in MATRIX_COLUMNS:
            value = clean(row.get(column))
            if value:
                parts.append(f"{label}: {value}")
                source_fields.append(column)

        chunks.append({
            "chunk_id": f"{psi_id}__matrix__{matrix_id}",
            "psi_id": psi_id,
            "run_id": None,
            "matrix_id": matrix_id,
            "pub_id": None,
            "chunk_type": "test_matrix",
            "text": " ".join(parts),
            "source_fields": source_fields,
            "metadata": {
                "acronym": inv.get("acronym"),
                "full_name": inv.get("full_name"),
                "row_kind": clean(row.get("row_kind")),
                "source_file": clean(row.get("source_file")),
                "psi_url": inv.get("psi_url"),
            },
        })
    return chunks


def build_publication_chunks(publications: pd.DataFrame, lookup: dict[str, dict]) -> list[dict]:
    chunks = []
    for _, row in publications.iterrows():
        psi_id = clean(row.get("psi_id"))
        pub_id = clean(row.get("pub_id"))
        inv = lookup.get(psi_id, {})

        parts = []
        for label, column in [
            ("Publication title", "title"),
            ("Authors", "authors"),
            ("Year", "year"),
            ("DOI", "doi"),
        ]:
            value = clean(row.get(column))
            if value:
                parts.append(f"{label}: {value}")

        chunks.append({
            "chunk_id": f"{psi_id}__publication__{pub_id}",
            "psi_id": psi_id,
            "run_id": None,
            "matrix_id": None,
            "pub_id": pub_id,
            "chunk_type": "publication",
            "text": " ".join(parts),
            "source_fields": ["title", "authors", "year", "doi"],
            "metadata": {
                "acronym": inv.get("acronym"),
                "full_name": inv.get("full_name"),
                "doi": clean(row.get("doi")),
                "psi_url": inv.get("psi_url"),
            },
        })
    return chunks


def validate_links(investigations, runs, publications, files_df, matrix):
    inv_ids = set(investigations["psi_id"].dropna().astype(str))
    errors = []

    for name, df in [
        ("runs", runs),
        ("publications", publications),
        ("files", files_df),
        ("test_matrix", matrix),
    ]:
        unknown = sorted(set(df["psi_id"].dropna().astype(str)) - inv_ids)
        if unknown:
            errors.append(f"{name}: unknown psi_id values: {unknown}")

    for name, df, id_col in [
        ("runs", runs, "run_id"),
        ("publications", publications, "pub_id"),
        ("files", files_df, "file_id"),
        ("test_matrix", matrix, "matrix_id"),
    ]:
        if df[id_col].isna().any():
            errors.append(f"{name}: missing {id_col}")
        if df[id_col].duplicated().any():
            errors.append(f"{name}: duplicate {id_col}")

    if errors:
        raise ValueError("\n".join(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument(
        "--publication-output",
        type=Path,
        default=Path("data/processed/publication_chunks.jsonl"),
    )
    args = parser.parse_args()

    inv = pd.read_csv(args.data_dir / "investigations.csv")
    runs = pd.read_csv(args.data_dir / "runs.csv")
    publications = pd.read_csv(args.data_dir / "publications.csv")
    files_df = pd.read_csv(args.data_dir / "files.csv")
    matrix = pd.read_csv(args.data_dir / "test_matrix.csv")

    validate_links(inv, runs, publications, files_df, matrix)

    inv_chunks, selected = build_investigation_chunks(inv)
    lookup = make_inv_lookup(selected)
    run_chunks = build_run_chunks(runs, lookup)
    matrix_chunks = build_matrix_chunks(matrix, lookup)
    pub_chunks = build_publication_chunks(publications, lookup)

    primary = inv_chunks + run_chunks + matrix_chunks
    write_jsonl(args.output, primary)
    write_jsonl(args.publication_output, pub_chunks)

    print(f"Indexable investigations: {selected['psi_id'].nunique()}")
    print(f"Investigation chunks: {len(inv_chunks)}")
    print(f"Run chunks: {len(run_chunks)}")
    print(f"Test-matrix chunks: {len(matrix_chunks)}")
    print(f"Primary SPLADE chunks: {len(primary)}")
    print(f"Optional publication chunks: {len(pub_chunks)}")


if __name__ == "__main__":
    main()
