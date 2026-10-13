"""Resolve public DecoderTCR artifacts through a live catalog."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download

from . import __version__
from .contracts import validate_bundle_schema
from .core import (
    IntegrityError,
    ReleaseError,
    UnavailableError,
    ValidationError,
    atomic_json,
    config_path,
    parse_model_id,
    parse_release_id,
    read_json,
    secure_path,
    sha256_file,
    verify_document,
)

DEFAULT_REPOSITORY = "Decoder-CHI/DecoderTCR-staging"
DEFAULT_PREFIX = "registry"
_KINDS = ("releases", "datasets", "results", "environments")
_STATES = {"available", "archived", "blocked"}
_PUBLIC_BENCHMARKS = {"tcrvdb", "immrep23", "prp", "viral-public-grid"}
_RAW_GENE_DATASETS = {
    "demo-gene-pairs@1.0.0": {
        "n_rows": 30,
        "columns": (
            "name",
            "trav",
            "traj",
            "cdr3a",
            "trbv",
            "trbj",
            "cdr3b",
            "hla",
            "peptide",
            "label",
        ),
        "label_column": "label",
        "label_counts": {"0": 15, "1": 15},
        "usage_column": None,
        "usage_counts": {},
        "public_baseline_path": "Demo/sample_data/genes_pairs.csv",
    },
    "immrep23-public-inputs@1.0.0": {
        "n_rows": 3484,
        "columns": (
            "ID",
            "Peptide",
            "HLA",
            "Va",
            "Ja",
            "TCRa",
            "CDR1a",
            "CDR2a",
            "CDR3a",
            "CDR3a_extended",
            "Vb",
            "Jb",
            "TCRb",
            "CDR1b",
            "CDR2b",
            "CDR3b",
            "CDR3b_extended",
            "Label",
            "Usage",
        ),
        "label_column": "Label",
        "label_counts": {"0": 2886, "1": 598},
        "usage_column": "Usage",
        "usage_counts": {"Private": 3217, "Public": 267},
        "public_baseline_path": "results/immrep23/immrep23_test.csv",
    },
}
_FAMILIES = {
    "decodertcr": {"300M", "600M", "6B"},
    "decodertcr-esm2": {"650M", "3B"},
}
_ARCHITECTURES = {
    "decodertcr": {size: f"DecoderTCRC_{size}" for size in _FAMILIES["decodertcr"]},
    "decodertcr-esm2": {size: f"ESM2_{size}" for size in _FAMILIES["decodertcr-esm2"]},
}


def _version(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not re.fullmatch(
        r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value
    ):
        raise ValidationError("Versions must have three numeric components")
    return tuple(int(item) for item in value.split("."))


def _compatible_member(family: str, member_name: str, member: dict) -> bool:
    backbones = {"decodertcr": "esmc", "decodertcr-esm2": "esm2"}
    return (
        family in _ARCHITECTURES
        and member_name in _ARCHITECTURES[family]
        and member.get("backbone") == backbones[family]
        and member.get("arch") == _ARCHITECTURES[family][member_name]
        and member.get("sequence_convention") in {"v1", "v2"}
    )


def _release_identity(release_id: str, manifest: dict) -> tuple[str, str]:
    family, version = parse_release_id(release_id)
    if (
        manifest.get("release_id") != release_id
        or manifest.get("family") != family
        or manifest.get("version") != version
    ):
        raise IntegrityError("Release identity disagrees with the live catalog")
    members = manifest.get("members", {})
    if not isinstance(members, dict) or set(members) != _FAMILIES.get(family):
        raise IntegrityError("Release family membership is incomplete or unexpected")
    for member_name, member in members.items():
        if not _compatible_member(family, member_name, member):
            raise IntegrityError("Release member identity and architecture disagree")
    return family, version


def _safe_repo_path(value: str) -> str:
    if not isinstance(value, str):
        raise ValidationError("Registry paths must be strings")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise ValidationError("Registry paths must be relative and cannot traverse parents")
    return path.as_posix()


def _descriptors(value):
    if isinstance(value, dict):
        if {"path", "sha256", "size_bytes"} <= value.keys():
            yield value
        for nested in value.values():
            yield from _descriptors(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _descriptors(nested)


def _validate_public_bundle(manifest: dict) -> None:
    validate_bundle_schema(manifest)
    if manifest.get("kind") != "dataset":
        return
    if manifest.get("schema_version") == 3:
        expected = _RAW_GENE_DATASETS.get(manifest.get("dataset_id"))
        observed = {
            "n_rows": manifest.get("n_rows"),
            "columns": tuple(manifest.get("columns", ())),
            "label_column": manifest.get("label_column"),
            "label_counts": manifest.get("label_counts"),
            "usage_column": manifest.get("usage_column"),
            "usage_counts": manifest.get("usage_counts"),
            "public_baseline_path": manifest.get("source", {}).get("public_baseline_path"),
        }
        if expected is None or observed != expected:
            raise ValidationError("Raw public dataset contract differs from the released input")
        if _safe_repo_path(manifest["artifact"]["path"]) != "data.csv":
            raise ValidationError("Raw public dataset artifact must be data.csv")
        return
    if manifest.get("schema_version") != 2:
        return
    benchmark = manifest.get("benchmark_name")
    if benchmark not in _PUBLIC_BENCHMARKS:
        raise ValidationError("Dataset is outside the public benchmark inventory")
    if manifest.get("dataset_id") != f"{benchmark}@1.0.0":
        raise ValidationError("Dataset identity and benchmark name disagree")
    population = manifest.get("population_sha256")
    if not isinstance(population, str) or not re.fullmatch(r"[0-9a-f]{64}", population):
        raise ValidationError("Paired datasets require a population identity")
    views = manifest.get("views")
    if not isinstance(views, dict) or set(views) != {"v1", "v2"}:
        raise ValidationError("Paired datasets require exactly v1 and v2 views")
    for convention, view in views.items():
        verify_document(view)
        if view.get("sequence_convention") != convention:
            raise ValidationError("Paired dataset view and sequence convention disagree")
        for descriptor in _descriptors(view):
            parts = PurePosixPath(_safe_repo_path(descriptor["path"])).parts
            if not parts or parts[0] != convention:
                raise ValidationError(
                    "Paired dataset artifacts must be stored beneath their convention"
                )


@dataclass(frozen=True)
class BundleLocation:
    """Locate one bundle in a local mirror or a pinned Hub revision."""

    relative: str
    local_root: Path | None
    repository: str | None
    prefix: str
    revision: str | None


@dataclass(frozen=True)
class ResolvedModel:
    bundle: BundleLocation
    manifest: dict
    member: dict
    state: str
    catalog_sha256: str
    model_id: str


@dataclass(frozen=True)
class ResolvedDataset:
    bundle: BundleLocation
    manifest: dict
    catalog_sha256: str
    generation: int
    dataset_id: str


class Registry:
    """Read a selected local mirror or live Hub catalog."""

    def __init__(self, source: Path | str | None = None):
        configured = source
        if configured is None:
            configured = os.environ.get("DECODERTCR_PUBLIC_REGISTRY")
        if configured is None and config_path().is_file():
            configured = read_json(config_path()).get("registry")
        if configured is None:
            self.local_root = None
            self.repository = DEFAULT_REPOSITORY
            self.prefix = DEFAULT_PREFIX
            self.spec = f"hf://{self.repository}/{self.prefix}"
        else:
            if not isinstance(configured, (str, Path)) or not str(configured).strip():
                raise ValidationError("Registry must be a nonempty path or Hub specification")
            text = str(configured).strip()
            if text.startswith("hf://"):
                body = text[5:].strip("/")
                parts = body.split("/")
                if len(parts) < 2:
                    raise ValidationError("Hub registries use hf://OWNER/REPOSITORY/PREFIX")
                self.local_root = None
                self.repository = "/".join(parts[:2])
                self.prefix = "/".join(parts[2:]) or DEFAULT_PREFIX
                _safe_repo_path(self.prefix)
                self.spec = f"hf://{self.repository}/{self.prefix}"
            else:
                self.local_root = Path(configured).expanduser().resolve()
                self.repository = None
                self.prefix = ""
                self.spec = str(self.local_root)

    @property
    def root(self) -> Path:
        if self.local_root is None:
            raise UnavailableError("A Hub registry has no local filesystem root")
        return self.local_root

    def _live_revision(self) -> str | None:
        if self.local_root is not None:
            return None
        try:
            info = HfApi().repo_info(
                repo_id=self.repository,
                repo_type="model",
                revision="main",
            )
        except Exception as exc:
            raise UnavailableError(
                "Cannot obtain the live Hub registry revision. "
                "Use an explicit local mirror for offline work"
            ) from exc
        revision = getattr(info, "sha", None)
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
            raise IntegrityError("The Hub did not return a commit-pinned registry revision")
        return revision

    def _path(self, relative: str, revision: str | None) -> Path:
        relative = _safe_repo_path(relative)
        if self.local_root is not None:
            unresolved = self.local_root / relative
            current = self.local_root
            linked = False
            for part in PurePosixPath(relative).parts:
                current = current / part
                linked = linked or current.is_symlink()
            path = secure_path(self.local_root, relative)
            if linked or unresolved.is_symlink():
                raise IntegrityError("Local registry files cannot be symbolic links")
            return path
        if revision is None:
            raise IntegrityError("Hub registry reads require a pinned revision")
        filename = f"{self.prefix}/{relative}" if self.prefix else relative
        try:
            value = hf_hub_download(
                repo_id=self.repository,
                filename=filename,
                repo_type="model",
                revision=revision,
                local_files_only=False,
            )
        except Exception as exc:
            raise UnavailableError(f"Cannot download registry file: {relative}") from exc
        path = Path(value)
        if not path.is_file():
            raise IntegrityError(f"Hub registry file is not regular: {relative}")
        return path

    def _document(self, relative: str, revision: str | None) -> dict:
        return verify_document(read_json(self._path(relative, revision)))

    def catalog(self) -> dict:
        """Read the current local catalog or obtain the live Hub head first."""
        revision = self._live_revision()
        catalog = self._document("catalog.json", revision)
        if catalog.get("schema_version") not in (1, 2):
            raise ValidationError("Upgrade DecoderTCR to read this catalog format")
        if not isinstance(catalog.get("generation"), int) or catalog["generation"] < 0:
            raise IntegrityError("Invalid catalog generation")
        for kind in _KINDS:
            if not isinstance(catalog.get(kind, {}), dict):
                raise IntegrityError(f"Invalid catalog section: {kind}")
            catalog.setdefault(kind, {})
            for identity, entry in catalog[kind].items():
                if not isinstance(entry, dict) or not {"path", "sha256"} <= entry.keys():
                    raise IntegrityError("Invalid catalog artifact reference")
                _safe_repo_path(entry["path"])
                if kind == "releases":
                    parse_release_id(identity)
                    if entry.get("state") not in _STATES:
                        raise IntegrityError("Invalid model availability state")
        catalog["_hub_revision"] = revision
        return catalog

    def _bundle(self, manifest_path: str, revision: str | None) -> BundleLocation:
        parent = PurePosixPath(_safe_repo_path(manifest_path)).parent.as_posix()
        return BundleLocation(parent, self.local_root, self.repository, self.prefix, revision)

    def _bundle_file(self, bundle: BundleLocation, relative: str) -> Path:
        relative = _safe_repo_path(relative)
        joined = PurePosixPath(bundle.relative, relative).as_posix()
        return self._path(joined, bundle.revision)

    def _manifest(self, entry: dict, revision: str | None) -> tuple[BundleLocation, dict]:
        path = _safe_repo_path(entry["path"])
        manifest = self._document(path, revision)
        if manifest["sha256"] != entry["sha256"]:
            raise IntegrityError("Manifest differs from the live catalog reference")
        if manifest.get("schema_version") not in (1, 2, 3):
            raise ValidationError("Upgrade DecoderTCR to read this manifest format")
        _validate_public_bundle(manifest)
        bundle = self._bundle(path, revision)
        if manifest.get("kind") == "release":
            validation = verify_document(read_json(self._bundle_file(bundle, "validation.json")))
            if validation["sha256"] != entry.get("validation_sha256"):
                raise IntegrityError("Validation differs from the live catalog reference")
            if validation.get("manifest_sha256") != manifest["sha256"]:
                raise IntegrityError("Validation refers to a different release manifest")
            evidence = validation.get("members")
            if validation.get("passed") is not True or not isinstance(evidence, dict):
                raise IntegrityError("Release validation does not record a passing result")
            if set(evidence) != set(manifest.get("members", {})):
                raise IntegrityError("Release validation does not cover every member")
            for member_name, item in evidence.items():
                if (
                    item.get("passed") is not True
                    or item.get("artifact_sha256")
                    != manifest["members"][member_name]["weights"]["sha256"]
                ):
                    raise IntegrityError("Release member validation is incomplete or mismatched")
        return bundle, manifest

    def models(self, include_unavailable: bool = False) -> list[dict]:
        catalog = self.catalog()
        revision = catalog.pop("_hub_revision")
        values = []
        for release_id, entry in sorted(catalog["releases"].items()):
            if not include_unavailable and entry["state"] != "available":
                continue
            _, manifest = self._manifest(entry, revision)
            if manifest.get("kind") != "release":
                raise IntegrityError("Release identity disagrees with the live catalog")
            family, _ = _release_identity(release_id, manifest)
            for member_name, member in manifest["members"].items():
                values.append(
                    {
                        "model_id": f"{release_id}:{member_name}",
                        "release_id": release_id,
                        "member": member_name,
                        "state": entry["state"],
                        "reason": entry.get("reason", ""),
                        "backbone": member["backbone"],
                        "arch": member["arch"],
                        "sequence_convention": member["sequence_convention"],
                        "minimum_package_version": manifest.get("minimum_package_version", "0.1.0"),
                        "compatible": _compatible_member(family, member_name, member)
                        and _version(manifest.get("minimum_package_version", "0.1.0"))
                        <= _version(__version__),
                    }
                )
        return values

    def datasets(self) -> list[dict]:
        """List registered raw gene-input datasets without downloading their CSVs."""
        catalog = self.catalog()
        revision = catalog.pop("_hub_revision")
        values = []
        for dataset_id, entry in sorted(catalog["datasets"].items()):
            _, manifest = self._manifest(entry, revision)
            if manifest.get("kind") != "dataset" or manifest.get("dataset_id") != dataset_id:
                raise IntegrityError("Dataset identity disagrees with the live catalog")
            if manifest.get("schema_version") != 3:
                continue
            values.append(
                {
                    "dataset_id": dataset_id,
                    "schema_version": 3,
                    "input_format": manifest["input_format"],
                    "species": manifest["species"],
                    "n_rows": manifest["n_rows"],
                    "columns": list(manifest["columns"]),
                    "artifact_sha256": manifest["artifact"]["sha256"],
                    "manifest_sha256": manifest["sha256"],
                    "catalog_sha256": catalog["sha256"],
                }
            )
        return values

    def dataset_info(self, dataset_id: str) -> dict:
        """Return raw dataset metadata without downloading its CSV."""
        resolved = self.resolve_dataset(dataset_id)
        return {
            **resolved.manifest,
            "catalog_sha256": resolved.catalog_sha256,
            "generation": resolved.generation,
        }

    def resolve_dataset(self, dataset_id: str) -> ResolvedDataset:
        """Resolve one raw gene-input dataset at a single catalog revision."""
        catalog = self.catalog()
        revision = catalog.pop("_hub_revision")
        entry = catalog["datasets"].get(dataset_id)
        if entry is None:
            raise UnavailableError(f"No registered dataset: {dataset_id}")
        bundle, manifest = self._manifest(entry, revision)
        if manifest.get("kind") != "dataset" or manifest.get("dataset_id") != dataset_id:
            raise IntegrityError("Dataset identity disagrees with the live catalog")
        if manifest.get("schema_version") != 3:
            raise ValidationError(
                "This dataset is a prepared reproduction bundle, not raw gene input"
            )
        return ResolvedDataset(
            bundle,
            manifest,
            catalog["sha256"],
            catalog["generation"],
            dataset_id,
        )

    def info(self, identity: str) -> dict:
        if ":" in identity:
            family, version, member_name = parse_model_id(identity)
            release_id = f"{family}@{version}"
        else:
            parse_release_id(identity)
            release_id, member_name = identity, None
        catalog = self.catalog()
        revision = catalog.pop("_hub_revision")
        entry = catalog["releases"].get(release_id)
        if entry is None:
            raise UnavailableError(f"No registered release: {release_id}")
        _, manifest = self._manifest(entry, revision)
        if manifest.get("kind") != "release":
            raise IntegrityError("Release identity disagrees with the live catalog")
        _release_identity(release_id, manifest)
        if member_name is not None and member_name not in manifest["members"]:
            raise UnavailableError(f"No registered model: {identity}")
        return {
            "release_id": release_id,
            "model_id": identity if member_name else None,
            "state": entry["state"],
            "reason": entry.get("reason", ""),
            "manifest": manifest,
            "member": manifest["members"][member_name] if member_name else None,
            "catalog_sha256": catalog["sha256"],
            "generation": catalog["generation"],
            "results": [],
        }

    def resolve(self, exact_id: str) -> ResolvedModel:
        family, version, member_name = parse_model_id(exact_id)
        release_id = f"{family}@{version}"
        catalog = self.catalog()
        revision = catalog.pop("_hub_revision")
        entry = catalog["releases"].get(release_id)
        if entry is None:
            raise UnavailableError(f"No registered release: {release_id}")
        if entry["state"] == "blocked":
            raise UnavailableError(f"{release_id} is blocked: {entry.get('reason', '')}")
        bundle, manifest = self._manifest(entry, revision)
        if manifest.get("kind") != "release":
            raise IntegrityError("Release identity disagrees with the live catalog")
        family, _ = _release_identity(release_id, manifest)
        required = manifest.get("minimum_package_version", "0.1.0")
        if _version(required) > _version(__version__):
            raise ValidationError(f"Upgrade DecoderTCR to >= {required} to load {release_id}")
        if member_name not in manifest.get("members", {}):
            raise UnavailableError(f"No registered model: {exact_id}")
        member = manifest["members"][member_name]
        if not _compatible_member(family, member_name, member):
            raise ValidationError("Upgrade DecoderTCR to load this model contract")
        if entry["state"] == "archived":
            warnings.warn(
                f"{release_id} is archived: {entry.get('reason', '')}",
                stacklevel=2,
            )
        return ResolvedModel(bundle, manifest, member, entry["state"], catalog["sha256"], exact_id)

    def resolve_bundle(self, kind: str, identity: str) -> tuple[dict, BundleLocation]:
        aliases = {"dataset": "datasets", "result": "results", "environment": "environments"}
        kind = aliases.get(kind, kind)
        if kind not in _KINDS[1:]:
            raise ValidationError("Resolve models through an exact model ID")
        catalog = self.catalog()
        revision = catalog.pop("_hub_revision")
        entry = catalog[kind].get(identity)
        if entry is None:
            raise UnavailableError(f"No registered {kind[:-1]}: {identity}")
        bundle, manifest = self._manifest(entry, revision)
        singular = kind[:-1]
        declared = manifest.get("id", manifest.get(f"{singular}_id"))
        if manifest.get("kind") != singular or declared != identity:
            raise IntegrityError("Bundle identity disagrees with the live catalog")
        return manifest, bundle

    def verify_artifact(self, bundle: BundleLocation, descriptor: dict, force=False) -> Path:
        if (
            not isinstance(descriptor, dict)
            or not {
                "path",
                "sha256",
                "size_bytes",
            }
            <= descriptor.keys()
        ):
            raise ValidationError("Artifact descriptor requires path, sha256, and size_bytes")
        digest = descriptor["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValidationError("Artifact requires a full SHA-256 digest")
        path = self._bundle_file(bundle, descriptor["path"])
        try:
            before = path.stat()
        except OSError as exc:
            raise IntegrityError(f"Artifact is unavailable: {descriptor['path']}") from exc
        if not stat.S_ISREG(before.st_mode) or before.st_size != descriptor["size_bytes"]:
            raise IntegrityError(f"Artifact size or type mismatch: {descriptor['path']}")
        signature = {
            "device": before.st_dev,
            "inode": before.st_ino,
            "size": before.st_size,
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
        }
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        key = hashlib.sha256(f"{path}\0{digest}".encode()).hexdigest()
        receipt = cache_root / "decodertcr-public" / "verified" / f"{key}.json"
        if not force and receipt.is_file():
            try:
                recorded = verify_document(read_json(receipt))
                if recorded.get("artifact_sha256") == digest and recorded.get("stat") == signature:
                    return path
            except ReleaseError:
                pass
        if sha256_file(path, progress=True) != digest:
            raise IntegrityError(f"Artifact checksum mismatch: {descriptor['path']}")
        after = path.stat()
        after_signature = {
            "device": after.st_dev,
            "inode": after.st_ino,
            "size": after.st_size,
            "mtime_ns": after.st_mtime_ns,
            "ctime_ns": after.st_ctime_ns,
        }
        if after_signature != signature:
            raise IntegrityError("Artifact changed while being verified")
        try:
            from .core import seal

            atomic_json(receipt, seal({"artifact_sha256": digest, "stat": signature}), mode=0o600)
        except OSError:
            warnings.warn("Verified artifact but could not save its receipt", stacklevel=2)
        return path


__all__ = ["BundleLocation", "Registry", "ResolvedDataset", "ResolvedModel"]
