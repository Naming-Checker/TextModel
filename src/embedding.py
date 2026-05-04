"""Build text embeddings for the trademark CSV slice.

Produces a (.pt) tensor of L2-normalized embeddings and
a (.csv) sidecar with metadata aligned by row order
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from encoder_semantic import SemanticEncoder


def pick_name(row: pd.Series) -> str:
    for col in ("mark_preprocessed", "mark_significant_normal", "mark_significant"):
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def parse_classes(val) -> list[int]:
    if pd.isna(val):
        return []
    try:
        return [int(x) for x in json.loads(val)]
    except Exception:
        try:
            import ast

            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, (list, tuple)):
                return [int(x) for x in parsed]
        except Exception:
            pass
    return []


def build_dataframe(csv_path: str, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if limit:
        df = df.head(limit)
    df["name_clean"] = df.apply(pick_name, axis=1)
    df = df[df["name_clean"] != ""].reset_index(drop=True)
    df["classes_json"] = df["class_number"].apply(lambda v: json.dumps(parse_classes(v)))
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode trademark names into a (.pt + .csv) pair")
    parser.add_argument("--csv", default="data/temp_trademark.csv", help="input trademark CSV")
    parser.add_argument("--model-path", default="models/rubert-tiny2", help="local HF snapshot dir")
    parser.add_argument("--output-pt", default="models/text_embedding.pt")
    parser.add_argument("--output-csv", default=None, help="defaults to <output-pt>.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="cap rows for a quick run")
    parser.add_argument("--device", default=None, help="cuda | cpu (auto)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1
    if not os.path.isdir(args.model_path):
        print(f"Model dir not found: {args.model_path}", file=sys.stderr)
        print("Hint: huggingface-cli download cointegrated/rubert-tiny2 --local-dir models/rubert-tiny2", file=sys.stderr)
        return 1

    output_csv = args.output_csv or os.path.splitext(args.output_pt)[0] + ".csv"
    os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)

    print(f"Loading {args.csv}...")
    df = build_dataframe(args.csv, args.limit)
    print(f"Rows with non-empty name_clean: {len(df)}")

    print(f"Loading encoder from {args.model_path}...")
    encoder = SemanticEncoder(args.model_path, device=args.device)
    print(f"Device: {encoder.device}, dim: {encoder.dim}")

    names = df["name_clean"].tolist()
    print(f"Encoding {len(names)} names (batch_size={args.batch_size})...")
    chunks = []
    for start in tqdm(range(0, len(names), args.batch_size)):
        chunk = encoder.encode(names[start : start + args.batch_size], batch_size=args.batch_size)
        chunks.append(chunk)
    embeddings = torch.cat(chunks, dim=0)
    print(f"Embeddings shape: {tuple(embeddings.shape)}")

    print(f"Saving embeddings to {args.output_pt}")
    torch.save(embeddings, args.output_pt)

    sidecar = pd.DataFrame(
        {
            "row_idx": range(len(df)),
            "name_clean": df["name_clean"].values,
            "name_display": df.get("mark_significant", df["name_clean"]).fillna("").astype(str).values,
            "mark_significant": df.get("mark_significant", "").fillna("").astype(str).values,
            "certificate_link": df.get("certificate_link", "").fillna("").astype(str).values,
            "classes_json": df["classes_json"].values,
        }
    )
    print(f"Saving sidecar to {output_csv}")
    sidecar.to_csv(output_csv, index=False)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
