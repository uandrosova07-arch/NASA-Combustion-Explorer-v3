# NASA Combustion Explorer

This is the sumbission for Nasa Space Apps Challange "Flame in Freefall: SI-Powered Fire Safety Insights from Microgravity Combustion Data", team lmsiki. 
MVP repository for SPLADE-powered retrieval over NASA Physical Sciences
Informatics (PSI) combustion investigations.
Our solution uses a trained SPLADE neural sparse retrieval model to make NASA PSI combustion investigations and experimental runs searchable, comparable, and easier to interpret for spacecraft fire-safety research. 
The main technical contribution of the project is the use of a trained SPLADE neural sparse retrieval model for scientific search.

SPLADE combines advantages of lexical and neural retrieval. It produces sparse vocabulary-based representations while also expanding queries and documents toward semantically related concepts.

For example, if a user types in:

"cool flames"

The model can retrieve investigations and experimental runs involving low-temperature combustion even when the exact wording differs across NASA records.

Because the representation remains sparse, the system also provides a degree of interpretability as the users can inspect the vocabulary dimensions activated by a query in the SPLADE Explorer. It also serves as a way to show the transparency of our model and how the searches work, this was users can understand what words are better to use in order to get the most relevant data. 


## But how does it work? 

## Current status

This snapshot is built from the supplied draft `investigations.csv`.

- Source CSV rows: **34**
- PSI investigations included in the retrieval corpus: **24**
- Generated investigation-level chunks: **97**



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


### Smoke-test the model on CPU

```bash
python scripts/check_splade_model.py --device cpu
```

Expected vector dimensions:

```text
documents: (2, 30522)
query:     (1, 30522)
```

### Build the document index once

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

### Test retrieval from the command line

```bash
python scripts/search_splade.py \
  "flame extinction with carbon dioxide in microgravity"
```

### Benchmark CPU query latency on your laptop

```bash
python scripts/benchmark_splade.py --device cpu
```

This benchmark matters more than guessing whether a GPU is required.

### Start Streamlit

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

Super Intelligence tools, including ChatGPT, were used as development assistants during this project.

Code: SI tools assisted with software architecture, Python and Streamlit implementation, data-validation scripts,  and deployment preparation. The final code was reviewed, tested, and executed by the project team.
