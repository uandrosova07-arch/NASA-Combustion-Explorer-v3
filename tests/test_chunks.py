import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_primary_chunks_are_unique_and_well_formed():
    path = ROOT / "data/processed/chunks.jsonl"
    chunks = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    ids = [x["chunk_id"] for x in chunks]
    assert chunks
    assert len(ids) == len(set(ids))
    allowed = {
        "investigation_overview", "investigation_objective",
        "investigation_hypothesis", "investigation_setup",
        "investigation_conditions", "investigation_approach",
        "investigation_evidence", "investigation_impacts",
        "run", "test_matrix"
    }
    for c in chunks:
        assert c["psi_id"]
        assert c["chunk_type"] in allowed
        assert c["text"].strip()
