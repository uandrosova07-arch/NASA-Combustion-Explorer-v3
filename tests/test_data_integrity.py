from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_foreign_keys_and_ids():
    inv = pd.read_csv(ROOT / "data/raw/investigations.csv")
    inv_ids = set(inv["psi_id"].dropna().astype(str))

    for filename, id_col in [
        ("runs.csv", "run_id"),
        ("publications.csv", "pub_id"),
        ("files.csv", "file_id"),
        ("test_matrix.csv", "matrix_id"),
    ]:
        df = pd.read_csv(ROOT / "data/raw" / filename)
        assert set(df["psi_id"].dropna().astype(str)) <= inv_ids
        assert not df[id_col].isna().any()
        assert not df[id_col].duplicated().any()
