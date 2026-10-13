#!/usr/bin/env python3
"""Run an explicitly selected real-artifact scoring smoke check."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import DecoderTCR as dt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="Exact model ID")
    parser.add_argument("--device", required=True)
    parser.add_argument("--registry")
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parents[1] / "Demo/sample_data/genes_pairs.csv"
    rows = pd.read_csv(source, keep_default_na=False)
    result = dt.score_from_components(
        rows,
        model=args.model_id,
        device=args.device,
        registry=args.registry,
    )
    valid = result[result["reconstruction_ok"]]
    if valid.empty or not np.isfinite(valid["pll"].to_numpy()).all():
        raise RuntimeError("The smoke result has no finite reconstructed score")
    print(f"model_id={args.model_id} rows={len(result)} scored={len(valid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
