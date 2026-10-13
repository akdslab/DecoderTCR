"""List models and run scoring, design, and registered reproduction."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .core import (
    ReleaseError,
    ValidationError,
    atomic_json,
    canonical_bytes,
    config_path,
    sha256_file,
    utc_now,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        help="Local registry or hf://OWNER/REPOSITORY/PREFIX",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init", aliases=["configure"], help="Save an explicitly selected registry"
    )
    init.add_argument(
        "--registry",
        required=True,
        help="Existing local registry or hf://OWNER/REPOSITORY/PREFIX",
    )
    models = commands.add_parser("models", help="List registered model releases")
    models.add_argument("--all", action="store_true", help="Include archived and blocked models")
    commands.add_parser("datasets", help="List registered raw gene-input datasets")
    dataset = commands.add_parser("dataset", help="Export one exact raw gene-input dataset")
    dataset.add_argument("dataset_id", help="Exact dataset ID")
    dataset.add_argument("--output", required=True, type=Path)
    score = commands.add_parser("score", help="Score TCR and pMHC rows")
    score.add_argument("--model", required=True, help="Exact model ID")
    score.add_argument("--input", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    score.add_argument("--device", default="cuda")
    score.add_argument("--species", choices=("human", "mouse"), default="human")
    profile = commands.add_parser("profile", help="Profile a masked sequence region")
    profile.add_argument("--model", required=True, help="Exact model ID")
    profile.add_argument("--input", required=True, type=Path)
    profile.add_argument("--output", required=True, type=Path)
    profile.add_argument("--device", default="cuda")
    profile.add_argument("--species", choices=("human", "mouse"), default="human")
    profile.add_argument(
        "--region",
        choices=("peptide", "cdr3a", "cdr3b", "cdr3"),
        default="peptide",
    )
    profile.add_argument("--length", type=int)
    profile.add_argument("--method", choices=("one_shot", "iegr"), default="one_shot")
    profile.add_argument("--temperature", type=float, default=0.1)
    profile.add_argument("--seed", type=int, default=42)
    design = commands.add_parser("design", help="Design peptide candidates")
    design.add_argument("--model", required=True, help="Exact model ID")
    design.add_argument("--input", required=True, type=Path)
    design.add_argument("--output", required=True, type=Path)
    design.add_argument("--device", default="cuda")
    design.add_argument("--species", choices=("human", "mouse"), default="human")
    design.add_argument("--length", required=True, type=int)
    design.add_argument("--num", type=int, default=10)
    design.add_argument("--method", choices=("one_shot", "iegr"), default="one_shot")
    design.add_argument("--profile-method", choices=("one_shot", "iegr"), default="one_shot")
    design.add_argument("--temperature", type=float)
    design.add_argument("--seed", type=int, default=42)
    design.add_argument("--cap", type=int, default=1_000_000)
    design.add_argument("--gibbs-k", type=int, default=0)
    design.add_argument("--gibbs-rounds", type=int, default=10)
    design.add_argument("--gibbs-subset-size", type=int, default=5)
    design.add_argument("--gibbs-temperature", type=float)
    design.add_argument("--gibbs-max-forwards", type=int)
    design.add_argument("--no-rescore", action="store_true")
    demo = commands.add_parser("demo", help="Run a registered demo result")
    demo.add_argument("--model", required=True, help="Exact model ID")
    demo.add_argument("--output", type=Path)
    reproduce = commands.add_parser("reproduce", help="Run a registered result recipe")
    reproduce.add_argument("result_id")
    reproduce.add_argument("--mode", choices=("metrics", "full"), default="metrics")
    reproduce.add_argument("--output", type=Path)
    reproduce.add_argument("--shard-index", type=int)
    reproduce.add_argument("--shard-count", type=int)
    reproduce.add_argument("--merge-shards", nargs="+", type=Path)
    reproduce.add_argument("--parent-output", type=Path)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command in ("init", "configure"):
            from .registry import Registry

            selected = Registry(args.registry)
            selected.catalog()
            atomic_json(config_path(), {"registry": selected.spec}, mode=0o600)
            print("Registry configured")
            return 0
        from .registry import Registry

        registry = Registry(args.registry)
        if args.command == "models":
            print(json.dumps(registry.models(include_unavailable=args.all), indent=2))
        elif args.command == "datasets":
            print(json.dumps(registry.datasets(), indent=2))
        elif args.command == "dataset":
            from ._datasets import resolve_artifact

            sidecar = args.output.with_suffix(args.output.suffix + ".json")
            if os.path.lexists(args.output) or os.path.lexists(sidecar):
                raise ValidationError("Output already exists. Choose a new output path")
            source, identity = resolve_artifact(args.dataset_id, registry=registry)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "schema_version": 1,
                "created_at": utc_now(),
                **identity,
                "output_sha256": identity["artifact_sha256"],
            }
            sidecar_created = False
            output_created = False
            try:
                with sidecar.open("xb") as metadata_stream:
                    sidecar_created = True
                    try:
                        output_stream = args.output.open("xb")
                        output_created = True
                        with source.open("rb") as source_stream, output_stream:
                            shutil.copyfileobj(source_stream, output_stream)
                        output_sha256 = sha256_file(args.output)
                        if output_sha256 != identity["artifact_sha256"]:
                            raise ValidationError(
                                "Exported dataset differs from the verified artifact"
                            )
                        metadata_stream.write(canonical_bytes(metadata) + b"\n")
                    except BaseException:
                        if output_created and (args.output.is_file() or args.output.is_symlink()):
                            args.output.unlink()
                        raise
            except FileExistsError as exc:
                if sidecar_created and (sidecar.is_file() or sidecar.is_symlink()):
                    sidecar.unlink()
                raise ValidationError("Output already exists. Choose a new output path") from exc
            except BaseException:
                if sidecar_created and (sidecar.is_file() or sidecar.is_symlink()):
                    sidecar.unlink()
                raise
            print(f"Exported {identity['n_rows']} rows")
        elif args.command == "score":
            import pandas as pd

            from .runtime import load

            if (
                args.output.exists()
                or args.output.with_suffix(args.output.suffix + ".json").exists()
            ):
                raise ValidationError("Output already exists. Choose a new output path")
            before = sha256_file(args.input)
            frame = (
                pd.read_parquet(args.input)
                if args.input.suffix == ".parquet"
                else pd.read_csv(args.input, keep_default_na=False)
            )
            model = load(args.model, device=args.device, registry=registry)
            result = model.score(
                frame,
                species=args.species,
            )
            if sha256_file(args.input) != before:
                raise ValidationError("Input changed while scoring")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.suffix == ".parquet":
                result.to_parquet(args.output, index=False)
            else:
                result.to_csv(args.output, index=False)
            metadata = {
                "schema_version": 1,
                "created_at": utc_now(),
                "model_id": args.model,
                "input_sha256": before,
                "output_sha256": sha256_file(args.output),
                "rows": len(result),
                "settings": {
                    "mask_region": "peptide",
                    "normalization": "full_vocabulary",
                    "input_format": "genes",
                    "species": args.species,
                },
                "artifact_sha256": result["artifact_sha256"].iloc[0]
                if len(result)
                else model.info.get("artifact_sha256"),
                "recipe": result.attrs,
            }
            atomic_json(args.output.with_suffix(args.output.suffix + ".json"), metadata)
            print(f"Scored {len(result)} rows")
        elif args.command in ("profile", "design"):
            import pandas as pd

            from .design import design_peptides, iegr_profile
            from .features import region_profile
            from .runtime import load

            if (
                args.output.exists()
                or args.output.with_suffix(args.output.suffix + ".json").exists()
            ):
                raise ValidationError("Output already exists. Choose a new output path")
            before = sha256_file(args.input)
            frame = (
                pd.read_parquet(args.input)
                if args.input.suffix in (".parquet", ".pq")
                else pd.read_csv(args.input, keep_default_na=False)
            )
            model = load(args.model, device=args.device, registry=registry)
            if args.command == "profile":
                if args.method == "iegr":
                    result = iegr_profile(
                        model,
                        frame,
                        region=args.region,
                        length=args.length,
                        temperature=args.temperature,
                        seed=args.seed,
                        species=args.species,
                    )
                else:
                    result = region_profile(
                        model,
                        frame,
                        region=args.region,
                        length=args.length,
                        species=args.species,
                    )
                result_attributes = dict(result.attrs)
                result = result.reset_index()
                result.attrs.update(result_attributes)
                settings = {
                    "region": args.region,
                    "length": args.length,
                    "method": args.method,
                    "temperature": args.temperature if args.method == "iegr" else None,
                    "seed": args.seed if args.method == "iegr" else None,
                    "input_format": "genes",
                    "species": args.species,
                }
            else:
                result = design_peptides(
                    model,
                    frame,
                    args.length,
                    n=args.num,
                    method=args.method,
                    profile_method=args.profile_method,
                    temperature=args.temperature,
                    seed=args.seed,
                    cap=args.cap,
                    gibbs_k=args.gibbs_k,
                    gibbs_rounds=args.gibbs_rounds,
                    gibbs_subset_size=args.gibbs_subset_size,
                    gibbs_temperature=args.gibbs_temperature,
                    gibbs_max_forwards=args.gibbs_max_forwards,
                    rescore=not args.no_rescore,
                    species=args.species,
                )
                settings = {
                    "length": args.length,
                    "n": args.num,
                    "method": args.method,
                    "profile_method": args.profile_method,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "cap": args.cap,
                    "gibbs_k": args.gibbs_k,
                    "gibbs_rounds": args.gibbs_rounds,
                    "gibbs_subset_size": args.gibbs_subset_size,
                    "gibbs_temperature": args.gibbs_temperature,
                    "gibbs_max_forwards": args.gibbs_max_forwards,
                    "rescore": not args.no_rescore,
                    "input_format": "genes",
                    "species": args.species,
                }
            if sha256_file(args.input) != before:
                raise ValidationError("Input changed during inference")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.suffix in (".parquet", ".pq"):
                result.to_parquet(args.output, index=False)
            else:
                result.to_csv(args.output, index=False)
            metadata = {
                "schema_version": 1,
                "created_at": utc_now(),
                "model_id": model.model_id,
                "artifact_sha256": model.artifact_sha256,
                "input_sha256": before,
                "output_sha256": sha256_file(args.output),
                "rows": len(result),
                "command": args.command,
                "settings": settings,
                "result_attributes": dict(result.attrs),
            }
            atomic_json(args.output.with_suffix(args.output.suffix + ".json"), metadata)
            print(f"Wrote {len(result)} {args.command} rows")
        elif args.command == "demo":
            from .reproduction import demo

            print(json.dumps(demo(args.model, output=args.output, registry=registry), indent=2))
        elif args.command == "reproduce":
            from .reproduction import merge_reproduction_shards, reproduce, reproduce_shard

            if (args.shard_index is None) != (args.shard_count is None):
                raise ValidationError("Shard index and count must be supplied together")
            if args.mode != "full" and (
                args.shard_index is not None
                or args.merge_shards is not None
                or args.parent_output is not None
            ):
                raise ValidationError("Shard and parent-output operations require full mode")
            if args.shard_index is not None and args.merge_shards is not None:
                raise ValidationError("Choose a shard worker or a shard merge")
            if args.parent_output is not None and (
                args.shard_index is not None or args.merge_shards is not None
            ):
                raise ValidationError("A parent output cannot be combined with shard operations")
            if args.shard_index is not None:
                value = reproduce_shard(
                    args.result_id,
                    args.shard_index,
                    args.shard_count,
                    args.output,
                    registry,
                )
            elif args.merge_shards is not None:
                value = merge_reproduction_shards(
                    args.result_id, args.merge_shards, args.output, registry
                )
            else:
                value = reproduce(
                    args.result_id,
                    mode=args.mode,
                    output=args.output,
                    registry=registry,
                    parent_output=args.parent_output,
                )
            print(json.dumps(value, indent=2))
        return 0
    except (ReleaseError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
