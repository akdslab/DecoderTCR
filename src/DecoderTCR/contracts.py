"""Validate immutable bundle contracts with the shipped canonical JSON Schema."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import best_match

from .core import ValidationError

_FORMAT_CHECKER = FormatChecker()


@_FORMAT_CHECKER.checks("date-time", raises=ValueError)
def _date_time(value) -> bool:
    # Validate RFC 3339 date-time values.
    if not isinstance(value, str):
        return True
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?"
        r"(?:[Zz]|[+-](?:[01]\d|2[0-3]):[0-5]\d)",
        value,
    ):
        return False
    return datetime.fromisoformat(value.upper()).utcoffset() is not None


@lru_cache(maxsize=1)
def _bundle_schema() -> dict:
    resource = files("DecoderTCR").joinpath("schemas/bundle.schema.json")
    if resource.is_file():
        return json.loads(resource.read_text(encoding="utf-8"))
    # Read the same canonical file when running an editable source checkout.
    source = Path(__file__).resolve()
    if source.parent.parent.name == "src":
        path = source.parents[2] / "schemas" / "bundle.schema.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise ValidationError("Bundle schema is missing. Reinstall DecoderTCR")


@lru_cache(maxsize=8)
def _validator(definition: str, allow_future_model_contract: bool = False) -> Draft202012Validator:
    document = _bundle_schema()
    if allow_future_model_contract:
        document = deepcopy(document)
        # Keep artifact and family structure strict while recognizing newer model identifiers.
        member = document["$defs"]["member"]
        for field in ("backbone", "arch", "sequence_convention"):
            member["properties"][field] = {"type": "string", "minLength": 1}
        for definition_name in ("release", "releaseV2"):
            for branch in document["$defs"][definition_name]["allOf"]:
                for member in branch["then"]["properties"]["members"]["properties"].values():
                    member["allOf"] = member["allOf"][:1]
    return Draft202012Validator(
        {"$ref": f"#/$defs/{definition}", "$defs": document["$defs"]},
        format_checker=_FORMAT_CHECKER,
    )


def validate_bundle_schema(manifest: dict, *, allow_future_model_contract: bool = False) -> None:
    """Reject incomplete or unsupported structures before their files are consumed."""
    kind = manifest.get("kind")
    if not isinstance(kind, str) or kind not in {"release", "dataset", "result", "environment"}:
        raise ValidationError("Bundle manifest has an unsupported kind")
    definition = kind
    if kind == "release":
        definition = "releaseV2" if manifest.get("schema_version") == 2 else "release"
    if kind == "dataset":
        definition = {
            1: "datasetLegacy",
            2: "datasetV2",
            3: "datasetRawGene",
        }.get(manifest.get("schema_version"), "dataset")
    error = best_match(_validator(definition, allow_future_model_contract).iter_errors(manifest))
    if error is not None:
        location = ".".join(str(part) for part in error.absolute_path) or "manifest"
        raise ValidationError(
            f"Invalid {kind} bundle contract at {location}: {error.message}. "
            "Check the bundle or upgrade DecoderTCR if it uses a newer contract"
        )
    if kind == "release" and manifest.get("schema_version") == 2:
        minimum_version = tuple(
            int(part) for part in manifest["minimum_package_version"].split(".")
        )
        if minimum_version < (0, 4, 0):
            raise ValidationError(
                "Schema version 2 releases require package version 0.4.0 or newer"
            )
        if set(manifest["declared_members"]) != set(manifest["members"]):
            raise ValidationError("Release declared_members must exactly match members")
