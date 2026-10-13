"""Freeze explicit benchmark populations and calculate their recorded metrics."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .core import (
    IntegrityError,
    ValidationError,
    artifact_descriptor,
    atomic_json,
    canonical_bytes,
    read_json,
    seal,
    secure_path,
    sha256_file,
    verify_document,
)

SEQUENCE_FIELDS = ("HLA_a", "HLA_b", "Peptide", "TCR_a", "TCR_b")
MODAL_THRESHOLDS = {"HLA_a": 287, "TCR_b": 294, "TCR_a": 258}
POPULATION_FIELDS = (
    "group_id",
    "group_key",
    "label",
    "offered_peptide",
    "cognate_epitope",
    "hla_allele",
    "split",
    "metadata_json",
)


def inventory() -> dict:
    return json.loads(files(__package__).joinpath("benchmark_inventory.json").read_text())[
        "benchmarks"
    ]


def release_inventory(specification: dict) -> dict:
    """Remove producer locations and lineage from a dataset inventory."""
    private = {
        "files",
        "data_dir",
        "results_dir",
        "derived_from",
        "derives",
        "not_comparable_with",
    }
    return {key: value for key, value in specification.items() if key not in private}


def _declaration(node):
    if isinstance(node, ast.Dict):
        return {_declaration(k): _declaration(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        if node.args or any(item.arg is None for item in node.keywords):
            raise ValidationError("Benchmark declarations must be literal dictionaries")
        return {item.arg: _declaration(item.value) for item in node.keywords}
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise ValidationError("Benchmark declarations contain executable expressions") from exc


def _producer_inventory(root: Path) -> dict:
    tree = ast.parse((root / "eval/scripts/benchmark_registry.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BENCHMARKS" for target in node.targets
        ):
            return _declaration(node.value)
    raise ValidationError("Producer has no explicit BENCHMARKS declaration")


def _entry_rows(raw: dict | list) -> list[tuple[str, dict]]:
    entries = list(raw.items()) if isinstance(raw, dict) else list(enumerate(raw))
    if not entries:
        raise ValidationError("Benchmark population is empty")
    if any(not isinstance(entry, dict) for _, entry in entries):
        raise ValidationError("Benchmark entries must be objects")
    return [(str(key), entry) for key, entry in entries]


def measure_convention(raw: dict | list) -> dict:
    entries = _entry_rows(raw)
    modes = {}
    for field in MODAL_THRESHOLDS:
        lengths = [
            len(entry.get("sequences", {}).get(field, ""))
            for _, entry in entries
            if entry.get("sequences", {}).get(field)
        ]
        if lengths:
            modes[field] = Counter(lengths).most_common(1)[0][0]
    votes = ["v1" if length >= MODAL_THRESHOLDS[field] else "v2" for field, length in modes.items()]
    tally = Counter(votes)
    convention, count = tally.most_common(1)[0] if tally else ("unknown", 0)
    if count < 2:
        convention = "unknown"
    if modes.get("HLA_a", 0) >= 400:
        convention = "pmhc_inhouse"
    return {"verdict": convention, "modal_lengths": modes}


def freeze_entries(raw: dict | list, name: str, convention: str) -> tuple[pd.DataFrame, list]:
    """Finalize the complete evaluation population before workers select rows."""
    definitions = inventory()
    if name not in definitions or convention not in ("v1", "v2"):
        raise ValidationError("Unknown benchmark or sequence convention")
    spec = definitions[name]
    if spec["files"][convention] is None:
        raise ValidationError(f"{name} has no {convention} build")
    entries = _entry_rows(raw)
    by_allele = defaultdict(set)
    if spec["needs_cross_pair"]:
        for _, entry in entries:
            meta, seq = entry.get("meta_data", {}), entry.get("sequences", {})
            allele, cognate = meta.get("hla_allele"), meta.get("epitope")
            if not allele or not cognate or cognate != seq.get("Peptide"):
                raise ValidationError("Cross-pairing requires a cognate peptide and HLA allele")
            by_allele[str(allele)].add(cognate)
    rows, excluded = [], []
    for key, entry in entries:
        seq, meta = entry.get("sequences", {}), entry.get("meta_data", {})
        if any(not isinstance(seq.get(field), str) for field in SEQUENCE_FIELDS):
            raise ValidationError(f"Entry {key} does not provide all five sequence strings")
        if not seq["Peptide"]:
            raise ValidationError(f"Entry {key} has an empty peptide")
        allele = str(meta.get("hla_allele") or meta.get("allele") or "")
        cognate = meta.get("true_epitope") or meta.get("cognate_epitope")
        if spec["needs_cross_pair"]:
            cognate = meta["epitope"]
        elif cognate is None and meta.get("label") == 1:
            cognate = seq["Peptide"]
        cognate = str(cognate) if cognate is not None else ""
        targets = [seq["Peptide"]]
        if spec["needs_cross_pair"]:
            targets = sorted(by_allele[allele])
            if len(targets) < 2:
                excluded.append(
                    {"source_row_id": key, "reason": "allele_has_fewer_than_two_epitopes"}
                )
                continue
        for target in targets:
            group_kind = spec["group_key"]
            if group_kind == "clone":
                group = meta.get("clone") or meta.get("epitope") or meta.get("clone_id")
            elif group_kind == "cognate_epitope":
                group = cognate
            else:
                group = meta.get("epitope") or target
            if group is None or str(group) == "":
                raise ValidationError(f"Entry {key} has no declared grouping value")
            label = int(target == cognate) if spec["needs_cross_pair"] else meta.get("label")
            if spec["metric_type"] == "ranking" and label not in (0, 1):
                raise ValidationError(f"Entry {key} lacks a binary label")
            normalized = {field: seq[field] for field in SEQUENCE_FIELDS}
            normalized["Peptide"] = target
            identity = {
                "benchmark": name,
                "convention": convention,
                "source_row_id": key,
                "sequences": normalized,
                "group_id": str(group),
                "label": label,
            }
            row = {
                **normalized,
                "row_id": hashlib.sha256(canonical_bytes(identity)).hexdigest(),
                "source_row_id": key,
                "group_id": str(group),
                "group_key": group_kind,
                "label": label,
                "offered_peptide": target,
                "cognate_epitope": cognate,
                "hla_allele": allele,
                "split": str(meta.get("split") or "benchmark"),
                "sequence_convention": convention,
                "metadata_json": json.dumps(meta, sort_keys=True),
                "pocket_idx_json": json.dumps(entry.get("pocket_idx", {}), sort_keys=True),
            }
            for field in ("kd_um", "log10_kd_um", "kd_nM", "kd_nm", "kd", "affinity"):
                if field in meta:
                    row[field] = meta[field]
            rows.append(row)
    if not rows:
        raise ValidationError("Preparation excluded every benchmark row")
    frame = pd.DataFrame(rows)
    if frame.row_id.duplicated().any():
        raise ValidationError("Prepared benchmark row identities are duplicated")
    return frame, excluded


def shard_rows(frame: pd.DataFrame, index: int, count: int) -> pd.DataFrame:
    if count < 1 or index < 0 or index >= count:
        raise ValidationError("Shard index must be in the range of the positive worker count")
    if "row_id" not in frame or frame.row_id.duplicated().any():
        raise ValidationError("Sharding requires a finalized table with unique row IDs")
    return frame.iloc[index::count].copy()


def population_digest(frame: pd.DataFrame) -> str:
    """Hash a convention-independent multiset of prepared biological rows."""
    missing = set(POPULATION_FIELDS) - set(frame.columns)
    if missing:
        raise ValidationError(f"Prepared benchmark lacks population fields: {sorted(missing)}")
    row_digests = []
    for values in frame.loc[:, POPULATION_FIELDS].itertuples(index=False, name=None):
        row = dict(zip(POPULATION_FIELDS, values))
        try:
            metadata = json.loads(row.pop("metadata_json"))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Prepared benchmark metadata is not valid JSON") from exc
        if not isinstance(metadata, dict):
            raise ValidationError("Prepared benchmark metadata must be a JSON object")
        subject = next(
            (
                str(metadata[field])
                for field in ("clonotype", "clone")
                if metadata.get(field) not in (None, "")
            ),
            None,
        )
        if subject is None:
            raise ValidationError("Prepared benchmark row has no stable biological subject")
        normalized = {
            "subject": subject,
            **{
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in row.items()
            },
        }
        row_digests.append(hashlib.sha256(canonical_bytes(normalized)).digest())
    digest = hashlib.sha256(b"decodertcr-population-v1\0")
    digest.update(len(row_digests).to_bytes(8, byteorder="big"))
    for row_digest in sorted(row_digests):
        digest.update(row_digest)
    return digest.hexdigest()


def _prepare_dataset_view(root, name, convention, definitions, producer):
    if name not in definitions or convention not in ("v1", "v2"):
        raise ValidationError("Unknown benchmark or convention")
    spec = definitions[name]
    if spec["files"][convention] is None:
        raise ValidationError(f"{name} has no {convention} build")
    required = ("files", "group_key", "ranking_metric", "needs_cross_pair", "derived_from")
    if producer is None or any(producer.get(key) != spec.get(key) for key in required):
        raise ValidationError("Producer benchmark declaration changed. Review the inventory first")
    subdirs = {
        "immrep23": "eval/immrep2023_benchmark/data",
        "immrep25": "eval/immrep2025_benchmark/data",
    }
    source = root / subdirs.get(spec.get("data_dir"), "eval/data") / spec["files"][convention]
    initial = source.stat()
    source_hash = sha256_file(source)
    raw = json.loads(source.read_text())
    measured = measure_convention(raw)
    if measured["verdict"] not in (convention, "pmhc_inhouse"):
        raise ValidationError(f"Benchmark measures as {measured['verdict']}, expected {convention}")
    if len(raw) != spec["n_entries"]:
        raise ValidationError("Declared benchmark row count changed. Review the inventory first")
    labels = [entry.get("meta_data", {}).get("label") for _, entry in _entry_rows(raw)]
    if (
        sum(label == 1 for label in labels) != spec["n_pos"]
        or sum(label == 0 for label in labels) != spec["n_neg"]
    ):
        raise ValidationError("Declared benchmark label counts changed. Review the inventory first")
    frame, excluded = freeze_entries(raw, name, convention)
    return {
        "convention": convention,
        "source": source,
        "source_signature": (initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns),
        "source_sha256": source_hash,
        "raw": raw,
        "measured_convention": measured,
        "frame": frame,
        "excluded": excluded,
    }


def _write_dataset_view(prepared, destination, bundle_root) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "raw").mkdir()
    copied = destination / "raw" / "input.json"
    shutil.copyfile(prepared["source"], copied)
    final = prepared["source"].stat()
    if (final.st_size, final.st_mtime_ns, final.st_ctime_ns) != prepared[
        "source_signature"
    ] or sha256_file(copied) != prepared["source_sha256"]:
        raise IntegrityError("Benchmark input changed during import")
    prepared["frame"].to_parquet(destination / "table.parquet", index=False)
    prepared["frame"].head(4).to_csv(destination / "example.csv", index=False)
    atomic_json(destination / "exclusions.json", {"rows": prepared["excluded"]})
    return {
        "sequence_convention": prepared["convention"],
        "table": artifact_descriptor(destination / "table.parquet", bundle_root),
        "example": artifact_descriptor(destination / "example.csv", bundle_root),
        "exclusions": artifact_descriptor(destination / "exclusions.json", bundle_root),
        "raw": [artifact_descriptor(copied, bundle_root)],
        "n_rows": len(prepared["frame"]),
        "n_source_rows": len(prepared["raw"]),
        "n_excluded_source_rows": len(prepared["excluded"]),
        "measured_convention": prepared["measured_convention"],
    }


def import_dataset(producer_root, name, convention, destination, version="1.0.0") -> dict:
    """Copy one benchmark build into a frozen schema version 1 bundle."""
    from .core import parse_release_id

    parse_release_id(f"dataset@{version}")
    root, dest = Path(producer_root).resolve(), Path(destination)
    if dest.exists():
        raise FileExistsError("Dataset destination already exists")
    definitions = inventory()
    if name not in definitions or convention not in ("v1", "v2"):
        raise ValidationError("Unknown benchmark or convention")
    producer = _producer_inventory(root).get(name)
    prepared = _prepare_dataset_view(root, name, convention, definitions, producer)
    spec = definitions[name]
    view = _write_dataset_view(prepared, dest, dest)
    manifest = seal(
        {
            "schema_version": 1,
            "kind": "dataset",
            "dataset_id": f"{name}-{convention}@{version}",
            "benchmark_name": name,
            **view,
            "group_key": spec["group_key"],
            "metric_type": spec["metric_type"],
            "inventory": release_inventory(spec),
            "preparation": "complete_population_before_sharding_v1",
        }
    )
    atomic_json(dest / "manifest.json", manifest)
    return manifest


def import_dataset_pair(producer_root, name, destination, version="1.0.0") -> dict:
    """Copy both preprocessed convention views into one versioned dataset bundle."""
    from .core import parse_release_id

    parse_release_id(f"dataset@{version}")
    root, dest = Path(producer_root).resolve(), Path(destination)
    if dest.exists():
        raise FileExistsError("Dataset destination already exists")
    definitions = inventory()
    if name not in definitions:
        raise ValidationError("Unknown benchmark")
    producer = _producer_inventory(root).get(name)
    prepared = {
        convention: _prepare_dataset_view(root, name, convention, definitions, producer)
        for convention in ("v1", "v2")
    }
    counts = {
        convention: (
            len(item["raw"]),
            len(item["frame"]),
            len(item["excluded"]),
        )
        for convention, item in prepared.items()
    }
    if counts["v1"] != counts["v2"]:
        raise ValidationError("Paired convention views must freeze the same population counts")
    population_digests = {
        convention: population_digest(item["frame"]) for convention, item in prepared.items()
    }
    if len(set(population_digests.values())) != 1:
        raise ValidationError("Paired convention views do not contain the same population")
    dest.mkdir(parents=True, exist_ok=False)
    views = {
        convention: seal(_write_dataset_view(item, dest / convention, dest))
        for convention, item in prepared.items()
    }
    spec = definitions[name]
    manifest = seal(
        {
            "schema_version": 2,
            "kind": "dataset",
            "dataset_id": f"{name}@{version}",
            "benchmark_name": name,
            "group_key": spec["group_key"],
            "metric_type": spec["metric_type"],
            "inventory": release_inventory(spec),
            "preparation": "complete_population_before_sharding_v1",
            "population_sha256": population_digests["v1"],
            "views": views,
        }
    )
    atomic_json(dest / "manifest.json", manifest)
    return manifest


def _verify_legacy_dataset_view(bundle, convention):
    manifest = verify_document(read_json(bundle / "manifest.json"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "dataset"
        or manifest.get("sequence_convention") != convention
    ):
        raise ValidationError(f"Expected a schema version 1 {convention} dataset bundle")
    benchmark = manifest.get("benchmark_name")
    if not isinstance(benchmark, str) or not manifest.get("dataset_id", "").startswith(
        f"{benchmark}-{convention}@"
    ):
        raise ValidationError("Legacy dataset identity and convention disagree")
    required = {
        "table",
        "example",
        "exclusions",
        "raw",
        "n_rows",
        "n_source_rows",
        "n_excluded_source_rows",
        "group_key",
        "metric_type",
        "inventory",
        "measured_convention",
        "preparation",
    }
    if (
        not required <= manifest.keys()
        or not isinstance(manifest["raw"], list)
        or not manifest["raw"]
    ):
        raise ValidationError("Legacy dataset manifest is incomplete")
    if any(
        type(manifest[key]) is not int
        or manifest[key] < (1 if key != "n_excluded_source_rows" else 0)
        for key in ("n_rows", "n_source_rows", "n_excluded_source_rows")
    ):
        raise ValidationError("Legacy dataset population counts are invalid")

    def verified(descriptor):
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise IntegrityError("Legacy dataset artifact descriptor is invalid")
        if (
            not isinstance(descriptor["path"], str)
            or not isinstance(descriptor["sha256"], str)
            or len(descriptor["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in descriptor["sha256"])
            or type(descriptor["size_bytes"]) is not int
            or descriptor["size_bytes"] < 0
        ):
            raise IntegrityError("Legacy dataset artifact descriptor is invalid")
        unresolved = bundle / descriptor["path"]
        if unresolved.is_symlink():
            raise IntegrityError("Legacy dataset artifacts cannot be symlinks")
        path = secure_path(bundle, descriptor["path"])
        if (
            not path.is_file()
            or path.stat().st_size != descriptor["size_bytes"]
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise IntegrityError("Legacy dataset artifact changed")
        return path

    artifacts = {key: verified(manifest[key]) for key in ("table", "example", "exclusions")}
    artifacts["raw"] = [verified(descriptor) for descriptor in manifest["raw"]]
    if (
        artifacts["table"].suffix != ".parquet"
        or artifacts["example"].suffix != ".csv"
        or artifacts["exclusions"].suffix != ".json"
    ):
        raise ValidationError("Legacy dataset uses unsupported prepared artifact formats")
    return manifest, artifacts


def _copy_legacy_dataset_view(manifest, artifacts, convention, destination, bundle_root):
    destination.mkdir(parents=True, exist_ok=False)
    copied = {}
    for key, filename in (
        ("table", "table.parquet"),
        ("example", "example.csv"),
        ("exclusions", "exclusions.json"),
    ):
        target = destination / filename
        shutil.copyfile(artifacts[key], target)
        copied[key] = artifact_descriptor(target, bundle_root)
        if (
            copied[key]["sha256"] != manifest[key]["sha256"]
            or copied[key]["size_bytes"] != manifest[key]["size_bytes"]
        ):
            raise IntegrityError("Dataset artifact changed while combining convention views")
    raw_dir = destination / "raw"
    raw_dir.mkdir()
    raw = []
    for index, source in enumerate(artifacts["raw"]):
        suffix = source.suffix or ".bin"
        filename = f"input{suffix}" if len(artifacts["raw"]) == 1 else f"input-{index:03d}{suffix}"
        target = raw_dir / filename
        shutil.copyfile(source, target)
        descriptor = artifact_descriptor(target, bundle_root)
        original = manifest["raw"][index]
        if (
            descriptor["sha256"] != original["sha256"]
            or descriptor["size_bytes"] != original["size_bytes"]
        ):
            raise IntegrityError("Dataset raw artifact changed while combining convention views")
        raw.append(descriptor)
    return seal(
        {
            "sequence_convention": convention,
            **copied,
            "raw": raw,
            "n_rows": manifest["n_rows"],
            "n_source_rows": manifest["n_source_rows"],
            "n_excluded_source_rows": manifest["n_excluded_source_rows"],
            "measured_convention": manifest["measured_convention"],
        }
    )


def combine_dataset_views(v1_bundle, v2_bundle, destination, version="1.0.0") -> dict:
    """Combine two verified legacy snapshots without preprocessing their frozen rows."""
    from .core import parse_release_id

    parse_release_id(f"dataset@{version}")
    sources = {"v1": Path(v1_bundle).resolve(), "v2": Path(v2_bundle).resolve()}
    dest = Path(destination)
    if dest.exists():
        raise FileExistsError("Dataset destination already exists")
    verified = {
        convention: _verify_legacy_dataset_view(bundle, convention)
        for convention, bundle in sources.items()
    }
    v1, v2 = (verified[convention][0] for convention in ("v1", "v2"))
    common = ("benchmark_name", "group_key", "metric_type", "inventory", "preparation")
    if any(v1.get(key) != v2.get(key) for key in common):
        raise ValidationError("Convention views disagree on their benchmark contract")
    counts = ("n_rows", "n_source_rows", "n_excluded_source_rows")
    if any(v1.get(key) != v2.get(key) for key in counts):
        raise ValidationError("Paired convention views must freeze the same population counts")
    population_digests = {
        convention: population_digest(
            pd.read_parquet(artifacts["table"], columns=list(POPULATION_FIELDS))
        )
        for convention, (_, artifacts) in verified.items()
    }
    if len(set(population_digests.values())) != 1:
        raise ValidationError("Paired convention views do not contain the same population")
    dest.mkdir(parents=True, exist_ok=False)
    views = {
        convention: _copy_legacy_dataset_view(
            manifest,
            artifacts,
            convention,
            dest / convention,
            dest,
        )
        for convention, (manifest, artifacts) in verified.items()
    }
    manifest = seal(
        {
            "schema_version": 2,
            "kind": "dataset",
            "dataset_id": f"{v1['benchmark_name']}@{version}",
            "benchmark_name": v1["benchmark_name"],
            "group_key": v1["group_key"],
            "metric_type": v1["metric_type"],
            "inventory": release_inventory(v1["inventory"]),
            "preparation": v1["preparation"],
            "population_sha256": population_digests["v1"],
            "views": views,
        }
    )
    atomic_json(dest / "manifest.json", manifest)
    return manifest


def recall_at_k(labels: np.ndarray, scores: np.ndarray, ks=(10, 100, 1000)) -> dict:
    """Exclude a tied block only when it straddles the requested cutoff."""
    order = np.argsort(-scores, kind="stable")
    ranked, ordered = labels[order], scores[order]
    n_pos = int(labels.sum())
    out = {}
    for requested in ks:
        if int(requested) != requested or requested <= 0:
            raise ValidationError("Recall cutoffs must be positive integers")
        cut = min(int(requested), len(ranked))
        if cut == 0 or n_pos == 0:
            out[f"recall@{requested}"] = None
            continue
        if cut < len(ordered) and ordered[cut] == ordered[cut - 1]:
            boundary = ordered[cut - 1]
            while cut and ordered[cut - 1] == boundary:
                cut -= 1
        out[f"recall@{requested}"] = float(ranked[:cut].sum()) / n_pos
    return out


def _macro_auc(frame, scores, group_column):
    values = []
    for _, group in frame.assign(_scores=np.asarray(scores)).groupby(group_column, sort=True):
        if group.label.nunique() == 2:
            values.append(float(roc_auc_score(group.label, group._scores)))
    return float(np.mean(values)) if values else None


def _marginals(frame, score, group_column):
    d = frame.copy()
    d["_tcr"] = d.TCR_a.astype(str) + "|" + d.TCR_b.astype(str)
    d["_pep"] = d.Peptide.astype(str)
    out = {}
    for name, over, keep in (("tcr_blind", "_tcr", "_pep"), ("pep_blind", "_pep", "_tcr")):
        crossed = float((d.groupby(keep)[over].transform("nunique") >= 2).mean())
        percentile = d.groupby(over)[score].rank(pct=True)
        marginal = percentile.groupby(d[keep]).transform("mean")
        out[f"{name}_crossed_frac"] = crossed
        out[name] = _macro_auc(d, marginal, group_column)
    return out


def _controls(frame, score, group_column, group_key, full):
    mg = _marginals(frame, score, group_column)
    primary = "tcr_blind" if group_key == "clone" else "pep_blind"
    other = "pep_blind" if primary == "tcr_blind" else "tcr_blind"

    def why(value):
        if value is None:
            return "undefined"
        if abs(value - 0.5) < 1e-9:
            return "exactly_chance"
        if full is not None and abs(value - full) < 1e-9:
            return "equals_full"
        return ""

    valid = mg[f"{primary}_crossed_frac"] >= 0.9
    control = mg[primary] if valid else None
    result = {
        **mg,
        "control_used": primary,
        "control_value": control,
        "gain": full - control if full is not None and control is not None else None,
        "control_crossed_frac": mg[f"{primary}_crossed_frac"],
        "control_invalid": "" if valid else "design_not_crossed",
        "control_degenerate": why(mg[primary]),
        "other_marginal": other,
        "other_marginal_value": mg[other],
        "other_marginal_degenerate": why(mg[other]),
    }
    if not valid:
        d = frame.copy()
        d["_tcr"] = d.TCR_a.astype(str) + "|" + d.TCR_b.astype(str)
        d["_pep"] = d.Peptide.astype(str)
        over, keep = ("_tcr", "_pep") if primary == "tcr_blind" else ("_pep", "_tcr")
        sub = d[d.groupby(keep)[over].transform("nunique") >= 2]
        subfull = _macro_auc(sub, sub[score], group_column)
        subcontrol = _marginals(sub, score, group_column)[primary] if len(sub) else None
        n_groups = sum(g.label.nunique() == 2 for _, g in sub.groupby(group_column))
        result.update(
            control_subset_rows=len(sub),
            control_subset_full=subfull,
            control_subset_value=subcontrol,
            control_subset_groups=n_groups,
            control_subset_pos=int(sub.label.sum()),
            control_subset_neg=int((sub.label == 0).sum()),
            control_subset_gain=(
                subfull - subcontrol if subfull is not None and subcontrol is not None else None
            ),
            control_subset_note="" if n_groups >= 5 else "fewer_than_five_both_class_groups",
        )
    return result


def compute_metrics(frame: pd.DataFrame, recipe: dict) -> dict:
    """Calculate metrics from a table and an explicit recipe."""
    metric_type = recipe.get("metric_type", "ranking")
    score, group_column = recipe.get("score_column", "pll"), recipe.get("group_column", "group_id")
    if score not in frame:
        raise ValidationError(f"Missing score column {score}")
    numeric = pd.to_numeric(frame[score], errors="raise")
    if np.isinf(numeric).any():
        raise ValidationError("Infinite scores are invalid")
    valid = numeric.notna()
    data = frame.loc[valid].copy()
    data[score] = numeric.loc[valid]
    result = {
        "metric_type": metric_type,
        "n_entries": len(frame),
        "n_scored": len(data),
        "n_missing_scores": int((~valid).sum()),
        "per_group": {},
    }
    if metric_type == "context":
        result.update(mean_pll=float(data[score].mean()) if len(data) else None)
        return result
    if metric_type == "affinity":
        column = recipe.get("affinity_column", "log10_kd_um")
        if column not in data:
            raise ValidationError(f"Affinity recipe requires {column}")
        affinity = pd.to_numeric(data[column], errors="coerce")
        keep = np.isfinite(affinity)
        x, y = data.loc[keep, score].to_numpy(), affinity[keep].to_numpy()
        if recipe.get("affinity_transform", "identity") == "negative_log10":
            if (y <= 0).any():
                raise ValidationError("Logarithmic affinity values must be positive")
            y = -np.log10(y)
        elif recipe.get("affinity_transform", "identity") != "identity":
            raise ValidationError("Unknown affinity transformation")
        nonconstant = len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0
        result.update(
            n_affinity=len(x),
            pearson=float(pearsonr(x, y).statistic) if nonconstant else None,
            spearman=float(spearmanr(x, y).statistic) if nonconstant else None,
        )
        return result
    if metric_type != "ranking":
        raise ValidationError("Unknown metric type")
    if "label" not in data or group_column not in data:
        raise ValidationError("Ranking metrics require labels and declared groups")
    if data[group_column].isna().any() or not data.label.isin([0, 1]).all():
        raise ValidationError("Ranking labels must be binary and groups must be present")
    group_key = recipe.get("group_key", "peptide")
    ks = tuple(recipe.get("recall_ks", (10, 100, 1000)))
    aucs, aps, prevalences = [], [], []
    recalls = defaultdict(list)
    sizes = []
    for group_name, group in data.groupby(group_column, sort=True):
        labels, scores = group.label.to_numpy(int), group[score].to_numpy(float)
        n_pos = int(labels.sum())
        record = {
            "n": len(group),
            "n_pos": n_pos,
            "n_neg": len(group) - n_pos,
            "auroc": None,
            "auprc": None,
            "prevalence": n_pos / len(group),
        }
        if np.unique(labels).size == 2:
            record.update(
                auroc=float(roc_auc_score(labels, scores)),
                auprc=float(average_precision_score(labels, scores)),
            )
            aucs.append(record["auroc"])
            aps.append(record["auprc"])
            prevalences.append(record["prevalence"])
        if n_pos:
            record.update(recall_at_k(labels, scores, ks))
            for key, value in record.items():
                if key.startswith("recall@"):
                    recalls[key].append(value)
        sizes.append(len(group))
        result["per_group"][str(group_name)] = record
    full = float(np.mean(aucs)) if aucs else None
    result.update(
        group_key=group_key,
        macro_auroc=full,
        macro_auprc=float(np.mean(aps)) if aps else None,
        macro_prevalence=float(np.mean(prevalences)) if prevalences else None,
        pooled_prevalence=float(data.label.mean()) if len(data) else None,
        n_pos=int(frame.label.sum()),
        n_groups=len(aucs),
        n_groups_total=len(sizes),
    )
    for key, values in recalls.items():
        result[f"macro_{key}"] = float(np.mean(values))
        if max(sizes) <= int(key.split("@")[1]):
            result[f"{key}_degenerate"] = f"K>={max(sizes)}=max group size"
    if recipe.get("controls", True) and len(data):
        if not {"TCR_a", "TCR_b", "Peptide"}.issubset(data.columns):
            raise ValidationError("Marginal controls require TCR and peptide sequences")
        result.update(_controls(data, score, group_column, group_key, full))
    return result
