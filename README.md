# NASA Combustion Explorer

MVP repository for SPLADE-powered retrieval over NASA Physical Sciences
Informatics (PSI) combustion investigations.

## Current status

This snapshot is built from the supplied draft `investigations.csv`.

- Source CSV rows: **34**
- PSI investigations included in the retrieval corpus: **24**
- Generated investigation-level chunks: **97**
- BM25: **not included by design**

The main retrieval corpus currently includes only rows that:

1. have a non-empty `psi_id`;
2. have `row_role` equal to `flight_investigation` or `ground_investigation`;
3. have `in_psi_search == yes`.

`context_not_in_psi_search` and `exclude_non_combustion` rows remain in the
source CSV but are not indexed.

## Repository

```text
NASA-Combustion-Explorer/
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .streamlit/
│   └── config.toml
├── config/
│   └── splade.example.yaml
├── data/
│   ├── raw/
│   │   └── investigations.csv
│   ├── processed/
│   │   ├── chunks.jsonl
│   │   └── chunk_stats.json
│   └── indexes/
│       └── splade/
├── models/
│   └── README.md
├── src/
│   ├── data/
│   │   ├── build_chunks.py
│   │   └── validate.py
│   ├── retrieval/
│   │   ├── splade_encoder.py
│   │   ├── splade_index.py
│   │   ├── search.py
│   │   └── aggregation.py
│   └── evaluation/
├── scripts/
│   ├── build_chunks.py
│   ├── validate_data.py
│   └── build_splade_index.py
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── 1_Search.py
│       ├── 2_Investigations.py
│       └── 3_SPLADE_Explorer.py
└── tests/
    └── test_chunks.py
```

## Build chunks

From the repository root:

```bash
python scripts/validate_data.py
python scripts/build_chunks.py
```

Or choose custom paths:

```bash
python scripts/build_chunks.py \
  --input data/raw/investigations.csv \
  --output data/processed/chunks.jsonl
```

## Chunk types

Each investigation can generate up to five semantic documents:

- `overview`
- `objectives`
- `experiment`
- `evidence`
- `impacts`

Every JSONL record contains at least:

```json
{
  "chunk_id": "PSI-69__experiment",
  "psi_id": "PSI-69",
  "run_id": null,
  "chunk_type": "experiment",
  "text": "...",
  "source_fields": ["fuel_samples", "hardware_facility", "test_conditions"],
  "metadata": { "...": "..." }
}
```

The text is generated only from source CSV fields. Missing values are skipped;
the script does not synthesize experimental facts.

## Run the current dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

The Investigation Catalog works now. SPLADE Search and SPLADE Explorer are
wired as the next step but intentionally do not pretend to know the checkpoint
format or pooling.

## Next step: connect the trained SPLADE checkpoint

Before implementing `src/retrieval/splade_encoder.py`, confirm:

- checkpoint format (`save_pretrained`, `.pt/.pth`, Sentence Transformers, etc.);
- tokenizer path;
- model class used during training;
- document/query encoder sharing;
- activation and pooling used to produce sparse vectors.

Then:

```text
chunks.jsonl
  -> encode documents once
  -> data/indexes/splade/sparse_docs.npz
  -> query encoder at runtime
  -> sparse matrix scoring
  -> aggregate chunks by psi_id
  -> join investigations.csv metadata
  -> Streamlit
```

For this corpus, document vectors should be precomputed. The deployed app only
needs to encode the user's query and multiply it by the sparse document matrix.


---

## Trained SPLADE integration

The supplied model documentation identifies the checkpoint as a custom
SPLADE-max on DistilBERT with a local `.pt` state dict. Its public package API is:

```python
from splade import Splade

model = Splade()
query_vectors = model.encode_queries(["query"])
document_vectors = model.encode_documents(["document"])
```

For this checkpoint:

- vector dimension: `30522`;
- maximum input length: `256` tokens for queries and documents;
- default batch size: `8`;
- CPU is supported by default;
- CUDA can be selected with `Splade(device="cuda")`.

### 1. Copy the complete model package

```text
models/splade/
├── splade.py
├── model.pt
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.txt
```

Do not commit `model.pt` to Git unless you intentionally use Git LFS or another
model storage strategy.

If the model package includes its own `requirements.txt`, install those exact
dependencies too, because they are part of the original trained-model runtime.

### 2. Smoke-test the model on CPU

```bash
python scripts/check_splade_model.py --device cpu
```

Expected vector dimensions:

```text
documents: (2, 30522)
query:     (1, 30522)
```

### 3. Build the document index once

```bash
python scripts/build_splade_index.py --device cpu
```

This writes:

```text
data/indexes/splade/
├── sparse_docs.npz
├── chunk_ids.npy
└── index_meta.json
```

The dashboard never re-encodes all documents during normal search.

### 4. Test retrieval from the command line

```bash
python scripts/search_splade.py \
  "flame extinction with carbon dioxide in microgravity"
```

### 5. Benchmark CPU query latency on your laptop

```bash
python scripts/benchmark_splade.py --device cpu
```

This benchmark matters more than guessing whether a GPU is required.

### 6. Start Streamlit

```bash
streamlit run dashboard/app.py
```

The Search page loads the model once with `st.cache_resource`, encodes only the
new user query, performs sparse inner-product scoring against
`sparse_docs.npz`, aggregates chunks by `psi_id`, and joins the investigation
metadata.

### CPU-first deployment

The intended MVP deployment is CPU-first:

```text
offline:
chunks.jsonl -> SPLADE document encoding -> sparse_docs.npz

online:
user query -> SPLADE query encoding -> sparse inner product -> top PSI records
```

With the current small PSI corpus, sparse scoring itself is negligible. The
main runtime cost is one DistilBERT/SPLADE query forward pass. Measure it with
`benchmark_splade.py`; move to a GPU host only if the observed CPU latency is
unacceptable for the demo.


---

## Multi-table NASA PSI corpus (current snapshot)

The repository now includes the current five-table snapshot:

- `investigations.csv`: 34 rows, 26 PSI IDs;
- `runs.csv`: 1862 structured run rows across 11 investigations;
- `test_matrix.csv`: 135 planned/test-day/session rows across 11 investigations;
- `publications.csv`: 352 publication links across 23 investigations;
- `files.csv`: 316 source-file/category metadata rows across 24 investigations.

All foreign `psi_id` values in the four supporting tables are present in
`investigations.csv`. IDs in the supporting tables are unique.

### Retrieval corpus

`data/processed/chunks.jsonl` is the **primary SPLADE corpus** and currently
contains:

- investigation-level semantic chunks;
- one `run` chunk for every row of `runs.csv`;
- one `test_matrix` chunk for every row of `test_matrix.csv`.

Total primary chunks in this snapshot: **2117**.

Publication titles are generated separately into
`data/processed/publication_chunks.jsonl` (352 chunks). They are
kept optional for now so publication-title overlap does not distort the main
investigation/run ranking before evaluation.

`files.csv` is intentionally **not** indexed by SPLADE. It is provenance and
download/catalog metadata for the dashboard.

### Why both `runs` and `test_matrix` exist

`runs.csv` represents actual per-run/per-sample experimental records where PSI
provides structured run tables. `test_matrix.csv` preserves planned matrices,
test-day summaries, sessions, and data-item indexes when PSI does not expose the
same per-run structure. They must not be silently merged as if they had the same
semantic meaning.


---

## Curated insights layer

`data/raw/insights.csv` contains **71** evidence-linked insights.

This file is deliberately **not encoded into the primary SPLADE index**.
The primary retrieval system should rank NASA PSI investigation/run/test-matrix
documents, not rank our own derived conclusions.

Instead, insights are joined after retrieval by `psi_id` and can be displayed
beneath a retrieved investigation with:

- `statement`
- `evidence_type`
- `source_ref`
- `source_url`
- `confidence`
- `caveat`

Current snapshot:

- high-confidence insights: **44**
- medium-confidence insights: **27**
- distinct linked PSI IDs: **23**

This separation is important for scientific provenance: SPLADE answers
"which source records are relevant?", while the insights layer answers
"what curated interpretation is associated with those records, and how strong
is its evidence?"

### Token-length audit

The trained SPLADE model has a 256-token limit. After copying the model package
into `models/splade/`, run:

```bash
python scripts/audit_chunk_lengths.py
```

Only after checking the number of over-limit chunks should the final SPLADE
document index be built:

```bash
python scripts/check_splade_model.py --device cpu
python scripts/build_splade_index.py --device cpu
```
