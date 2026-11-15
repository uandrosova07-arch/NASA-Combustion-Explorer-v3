# Put the trained SPLADE package here

Copy the COMPLETE package supplied with the trained model into this directory.

Expected minimum layout:

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

Windows Explorer can hide known extensions, so `config`, `tokenizer`,
`tokenizer_config`, `special_tokens_map`, and `vocab` may actually be
`.json` / `.txt` files.

This repository does not include the model weights.

The integration uses the model package's documented API:

```python
from splade import Splade

model = Splade(device="cpu")
docs = model.encode_documents(["..."], batch_size=8)
query = model.encode_queries(["..."])
```

For the supplied checkpoint, vectors are expected to have dimension 30,522.
