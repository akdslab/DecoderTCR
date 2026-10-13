#!/usr/bin/env python3
"""Download and verify explicitly selected exact model artifacts."""

from __future__ import annotations

import argparse
import json

import DecoderTCR as dt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", nargs="+", help="Exact model ID")
    parser.add_argument("--registry", help="Explicit local or Hub registry")
    args = parser.parse_args(argv)
    records = [dt.artifact(model_id, registry=args.registry) for model_id in args.model_id]
    print(
        json.dumps(
            [{**record, "path": str(record["path"])} for record in records],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
