# Text-similarity v0

Mirrors `VisualModel/src`: an embedding builder and a cosine query script.
Input: trademark name; output: 312-d vector via `cointegrated/rubert-tiny2`

## Setup

```bash
pip install torch transformers numpy pandas tqdm
hf download cointegrated/rubert-tiny2 --local-dir models/rubert-tiny2
```

Place the data slice at `data/temp_trademark.csv`

## Build embeddings

```bash
python src/embedding.py \
  --csv data/temp_trademark.csv \
  --model-path models/rubert-tiny2 \
  --output-pt models/text_embedding.pt
```

Produces `models/text_embedding.pt` `(N, 312)` float32 L2-normalized,
plus `models/text_embedding.csv` with row-aligned metadata
(`row_idx, name_clean, name_display, mark_significant, certificate_link, classes_json`).

## Query

CLI:

```bash
python src/similarity.py "EUROPLEX" --top 10 --mktu 5,35
```

Library:

```python
from src.similarity import calculate_similarity
import numpy as np

sims, sidecar = calculate_similarity(
    "EUROPLEX",
    embeddings_path="models/text_embedding.pt",
    model_path="models/rubert-tiny2",
    mktu=[5, 35],   # optional
)
top = np.argsort(sims)[::-1][:10]
for idx in top:
    print(f"{sims[idx]*100:.1f}%  {sidecar.iloc[idx]['name_display']}")
```

`sims` is `(N,) float`, cosine similarity in `[-1, 1]`. МКТУ-filtered rows
are set to `-inf`

NOTE: results are not pre-ranked
