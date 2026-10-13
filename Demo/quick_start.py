#!/usr/bin/env python3
"""Score bundled gene-component examples with an explicitly selected model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import DecoderTCR as dt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="Exact model ID")
    parser.add_argument("--device", required=True)
    parser.add_argument("--registry")
    parser.add_argument("--rows", type=int, default=4)
    args = parser.parse_args(argv)
    if args.rows < 1:
        parser.error("--rows must be positive")
    source = Path(__file__).parent / "sample_data/genes_pairs.csv"
    rows = pd.read_csv(source, keep_default_na=False).head(args.rows)
    loaded = dt.load(args.model_id, device=args.device, registry=args.registry)
    result = loaded.score(rows)
    print(result[["name", "peptide", "reconstruction_ok", "pll"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
