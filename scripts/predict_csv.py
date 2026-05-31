from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core import predict_records


def main():
    parser = argparse.ArgumentParser(description = "Score a CSV with the combined review detectors.")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--text-col", default = "text")
    parser.add_argument("--store-col", default = "store")
    parser.add_argument("--bundle-id-col", default = "bundle_id")
    parser.add_argument("--review-id-col", default = "review_id")
    parser.add_argument("--lang-col", default = None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    required = [args.text_col, args.store_col, args.bundle_id_col, args.review_id_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    records = []
    for _, row in df.iterrows():
        record = {
            "store": row.get(args.store_col),
            "bundle_id": row.get(args.bundle_id_col),
            "review_id": row.get(args.review_id_col),
            "text": row.get(args.text_col),
        }
        if args.lang_col and args.lang_col in df.columns:
            record["lang"] = row.get(args.lang_col)
        records.append(record)

    scored = pd.DataFrame(predict_records(records))
    scored.to_csv(args.output_csv, index = False)

    print(json.dumps({
        "input_csv": str(Path(args.input_csv)),
        "output_csv": str(Path(args.output_csv)),
        "n_rows": len(scored),
        "columns": scored.columns.tolist(),
    }, indent = 2))


if __name__ == "__main__":
    main()
