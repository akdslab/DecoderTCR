"""Load and verify bundled reconstruction resources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from DecoderTCR.core import IntegrityError, ValidationError

_DATA_ROOT = Path(__file__).resolve().parent / "data"
_MANIFEST = _DATA_ROOT / "manifest.json"
_STITCHR_SPECIES = {"human": "HUMAN", "mouse": "MOUSE"}


def validate_species(value: str) -> str:
    """Return one supported biological species identifier."""
    if not isinstance(value, str):
        raise ValidationError("species must be 'human' or 'mouse'")
    species = value.strip().lower()
    if species not in _STITCHR_SPECIES:
        raise ValidationError("species must be 'human' or 'mouse'")
    return species


def stitchr_species(value: str) -> str:
    """Return the pinned Stitchr species name."""
    return _STITCHR_SPECIES[validate_species(value)]


def data_path(relative: str) -> Path:
    """Return one path below the bundled reconstruction data directory."""
    path = (_DATA_ROOT / relative).resolve()
    try:
        path.relative_to(_DATA_ROOT.resolve())
    except ValueError as error:
        raise ValidationError(f"Invalid reconstruction resource path: {relative!r}") from error
    return path


def verify_resources(relative_paths: tuple[str, ...] | list[str]) -> None:
    """Verify selected resources against the bundled SHA-256 manifest."""
    try:
        manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
        expected = manifest["files"]
    except (OSError, KeyError, json.JSONDecodeError, TypeError) as error:
        raise IntegrityError(
            "The reconstruction resource manifest is missing or malformed"
        ) from error
    if not isinstance(expected, dict):
        raise IntegrityError("The reconstruction resource manifest file map is malformed")

    for relative in relative_paths:
        digest = expected.get(relative)
        if not isinstance(digest, str) or len(digest) != 64:
            raise IntegrityError(f"Resource is absent from the hash manifest: {relative}")
        path = data_path(relative)
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise IntegrityError(f"Reconstruction resource is unavailable: {relative}") from error
        if actual != digest:
            raise IntegrityError(f"Reconstruction resource failed hash verification: {relative}")


def resource_identity(relative_paths: tuple[str, ...] | list[str]) -> dict:
    """Return JSON-safe identities after verifying selected bundled resources."""
    verify_resources(relative_paths)
    manifest_bytes = _MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    return {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "resources": {relative: manifest["files"][relative] for relative in relative_paths},
    }


def _stitchr_resources(species: str) -> tuple[str, ...]:
    stitchr_name = stitchr_species(species)
    return (
        f"stitchr/{stitchr_name}/TRA.fasta",
        f"stitchr/{stitchr_name}/TRB.fasta",
        f"stitchr/{stitchr_name}/TRD.fasta",
        f"stitchr/{stitchr_name}/TRG.fasta",
        f"stitchr/{stitchr_name}/J-region-motifs.tsv",
        f"stitchr/{stitchr_name}/C-region-motifs.tsv",
        f"stitchr/{stitchr_name}/data-production-date.tsv",
        f"stitchr/{stitchr_name}/IMGTgeneDLwarnings.txt",
        f"stitchr/kazusa/{stitchr_name}.txt",
        "stitchr/linkers.tsv",
    )


STITCHR_RESOURCES_BY_SPECIES = {
    species: _stitchr_resources(species) for species in _STITCHR_SPECIES
}
STITCHR_RESOURCES = STITCHR_RESOURCES_BY_SPECIES["human"]
