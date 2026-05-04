"""Query the cached embeddings for the top-N most similar trademarks

Takes a query string, encodes it with the same model used to build the cache,
computes cosine similarity, prints the top-N as `i. score% - name (cert_link)`

Optionally filters by МКТУ classes
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

# Make sibling modules importable both as a CLI (`python src/similarity.py ...`)
# and as a library (`from src.similarity import calculate_similarity`)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from encoder_semantic import SemanticEncoder


def load_embeddings(embeddings_path: str) -> tuple[torch.Tensor, pd.DataFrame]:
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    sidecar_path = os.path.splitext(embeddings_path)[0] + ".csv"
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(f"Sidecar CSV not found: {sidecar_path}")
    embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    sidecar = pd.read_csv(sidecar_path)
    if len(embeddings) != len(sidecar):
        raise ValueError(
            f"Row count mismatch: embeddings={len(embeddings)}, sidecar={len(sidecar)}"
        )
    return embeddings, sidecar


def cosine_topk(
    query_vec: torch.Tensor, embeddings: torch.Tensor, mask: np.ndarray | None, top_n: int
) -> tuple[np.ndarray, np.ndarray]:
    sims = (embeddings @ query_vec.squeeze(0)).numpy()
    if mask is not None:
        sims = np.where(mask, sims, -np.inf)
    k = min(top_n, int(np.isfinite(sims).sum()))
    if k <= 0:
        return np.array([]), np.array([])
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return sims[top_idx], top_idx


def class_mask(sidecar: pd.DataFrame, mktu: list[int]) -> np.ndarray | None:
    if not mktu:
        return None
    target = set(mktu)
    return sidecar["classes_json"].apply(lambda j: bool(set(json.loads(j)) & target)).to_numpy()


def calculate_similarity(
    query: str,
    embeddings_path: str = "models/text_embedding.pt",
    model_path: str = "models/rubert-tiny2",
    mktu: list[int] | None = None,
    device: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings, sidecar = load_embeddings(embeddings_path)
    encoder = SemanticEncoder(model_path, device=device)
    q = encoder.encode([query])
    sims = (embeddings @ q.squeeze(0)).numpy()
    if mktu:
        mask = class_mask(sidecar, mktu)
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
    return sims, sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description="Query cached trademark embeddings")
    parser.add_argument("query", help="trademark name to check")
    parser.add_argument("--embeddings-path", default="models/text_embedding.pt")
    parser.add_argument("--model-path", default="models/rubert-tiny2")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--mktu",
        default="",
        help="comma-separated МКТУ class filter, e.g. '5,35'; empty = all classes",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    mktu = [int(x) for x in args.mktu.split(",") if x.strip()]

    try:
        embeddings, sidecar = load_embeddings(args.embeddings_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Hint: run `python src/embedding.py` first", file=sys.stderr)
        return 1

    encoder = SemanticEncoder(args.model_path, device=args.device)
    q = encoder.encode([args.query])

    mask = class_mask(sidecar, mktu)
    sims, top_idx = cosine_topk(q, embeddings, mask, args.top)
    if len(top_idx) == 0:
        print("No matches (МКТУ filter eliminated all rows)")
        return 0

    for rank, (sim, idx) in enumerate(zip(sims, top_idx), start=1):
        row = sidecar.iloc[idx]
        name = row["name_display"] or row["name_clean"]
        link = row.get("certificate_link", "") or ""
        link_str = f"  {link}" if isinstance(link, str) and link.startswith("http") else ""
        print(f"{rank:>3}. {sim*100:6.2f}% - {name}{link_str}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
