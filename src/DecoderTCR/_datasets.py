"""Read verified public gene-input datasets from the selected registry."""

from __future__ import annotations

from collections import Counter

from .core import IntegrityError
from .registry import Registry


def _selected(registry=None) -> Registry:
    return registry if isinstance(registry, Registry) else Registry(registry)


def _identity(resolved) -> dict:
    manifest = resolved.manifest
    return {
        "dataset_id": resolved.dataset_id,
        "input_format": manifest["input_format"],
        "species": manifest["species"],
        "n_rows": manifest["n_rows"],
        "artifact_sha256": manifest["artifact"]["sha256"],
        "manifest_sha256": manifest["sha256"],
        "catalog_sha256": resolved.catalog_sha256,
        "generation": resolved.generation,
        "source": dict(manifest["source"]),
    }


def resolve_artifact(dataset_id: str, *, registry=None):
    """Return a verified raw CSV path and its immutable dataset identity."""
    selected = _selected(registry)
    resolved = selected.resolve_dataset(dataset_id)
    path = selected.verify_artifact(resolved.bundle, resolved.manifest["artifact"])
    return path, _identity(resolved)


def load(dataset_id: str, *, registry=None):
    """Return one verified raw gene-input dataset as a DataFrame."""
    import pandas as pd

    selected = _selected(registry)
    resolved = selected.resolve_dataset(dataset_id)
    manifest = resolved.manifest
    path = selected.verify_artifact(resolved.bundle, manifest["artifact"])
    frame = pd.read_csv(path, keep_default_na=False)
    if list(frame.columns) != manifest["columns"] or len(frame) != manifest["n_rows"]:
        raise IntegrityError("Parsed dataset shape differs from its manifest")
    labels = Counter(str(value) for value in frame[manifest["label_column"]])
    if dict(labels) != manifest["label_counts"]:
        raise IntegrityError("Parsed dataset labels differ from its manifest")
    usage_column = manifest["usage_column"]
    usage = Counter(str(value) for value in frame[usage_column]) if usage_column else Counter()
    if dict(usage) != manifest["usage_counts"]:
        raise IntegrityError("Parsed dataset usage groups differ from its manifest")
    frame.attrs["dataset"] = _identity(resolved)
    return frame


__all__ = ["load", "resolve_artifact"]
