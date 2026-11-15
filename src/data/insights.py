from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def load_insights(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def parse_psi_ids(value) -> list[str]:
    if pd.isna(value):
        return []
    return re.findall(r"PSI-\d+", str(value))


def insights_for_psi_ids(
    insights: pd.DataFrame,
    psi_ids: list[str] | set[str],
) -> pd.DataFrame:
    wanted = set(psi_ids)
    mask = insights["psi_ids"].apply(
        lambda value: bool(set(parse_psi_ids(value)) & wanted)
    )
    return insights[mask].copy()
