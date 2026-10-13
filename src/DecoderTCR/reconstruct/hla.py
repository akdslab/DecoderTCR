"""HLA sequence lookup for DecoderTCR sequence conventions."""

from __future__ import annotations

import json
import re

from DecoderTCR.core import IntegrityError, ValidationError

from ._resources import data_path, validate_species, verify_resources

_REFERENCE_FILES = {"v1": "hla_v1.json", "v2": "hla_v2.json"}
_SPECIES_V2_FILE = "hla_v2_species.json"
_CLASS_I = {"A", "B", "C"}
_CLASS_II_PAIRS = {
    frozenset(("DRA", "DRB1")),
    frozenset(("DRA", "DRB3")),
    frozenset(("DRA", "DRB4")),
    frozenset(("DRA", "DRB5")),
    frozenset(("DQA1", "DQB1")),
    frozenset(("DPA1", "DPB1")),
}
_ALPHA_LOCI = {"DRA", "DQA1", "DPA1"}


def _convention(value: str) -> str:
    convention = str(value).strip()
    if convention not in _REFERENCE_FILES:
        raise ValidationError("sequence_convention must be 'v1' or 'v2'")
    return convention


def _human_component(allele: str) -> tuple[str, str]:
    value = allele.strip().upper()
    value = re.sub(r"^HLA[-_]?", "", value)
    value = value.replace("*", "").replace(":", "").replace("-", "")
    match = re.fullmatch(
        r"(DRA|DRB1|DRB3|DRB4|DRB5|DQA1|DQB1|DPA1|DPB1|A|B|C)(\d{4,})",
        value,
    )
    if match is None:
        raise ValidationError(f"Invalid HLA allele: {allele!r}")
    return match.group(1), value


def _mouse_component(allele: str) -> tuple[str, str]:
    value = allele.strip().upper().replace("*", "").replace(":", "").replace("-", "")
    match = re.fullmatch(r"(H2(?:AA|AB|EA|EB|K|D|L))([A-Z0-9]+)", value)
    if match is None:
        raise ValidationError(f"Invalid mouse MHC allele: {allele!r}")
    locus = match.group(1)
    role = "alpha" if locus in {"H2AA", "H2EA"} else "beta"
    if locus in {"H2K", "H2D", "H2L"}:
        role = "class_i"
    return role, value


def normalize_allele_key(allele: str, *, species: str = "human") -> str:
    """Normalize class I or one explicit class II alpha and beta pair."""
    if not isinstance(allele, str):
        raise ValidationError("MHC allele must be a string")
    species = validate_species(species)
    value = allele.strip()
    parts = value.split("/") if "/" in value else value.split("__")
    if len(parts) == 2 and parts[1].strip().upper() in {"B2M", "B2M_HUMAN", "B2M_MOUSE"}:
        explicit_b2m = parts[1].strip().upper()
        if explicit_b2m == "B2M_HUMAN" and species != "human":
            raise ValidationError("Human B2M cannot be selected for mouse MHC")
        if explicit_b2m == "B2M_MOUSE" and species != "mouse":
            raise ValidationError("Mouse B2M cannot be selected for human MHC")
        parts = parts[:1]
    if len(parts) == 1:
        if species == "human":
            locus, normalized = _human_component(parts[0])
            class_i = locus in _CLASS_I
        else:
            role, normalized = _mouse_component(parts[0])
            class_i = role == "class_i"
        if not class_i:
            raise ValidationError("Class II lookup requires an explicit alpha and beta pair")
        return f"{normalized}__B2M"
    if len(parts) != 2:
        raise ValidationError("MHC lookup accepts one allele or one explicit pair")
    if species == "human":
        components = [_human_component(part) for part in parts]
        loci = frozenset(locus for locus, _ in components)
        if loci not in _CLASS_II_PAIRS:
            raise ValidationError("Unsupported class II alpha and beta pairing")
        components.sort(key=lambda item: item[0] not in _ALPHA_LOCI)
    else:
        components = [_mouse_component(part) for part in parts]
        if {role for role, _ in components} != {"alpha", "beta"}:
            raise ValidationError("Unsupported mouse class II alpha and beta pairing")
        components.sort(key=lambda item: item[0] != "alpha")
    return "__".join(normalized for _, normalized in components)


def _read_reference(relative: str) -> dict[str, dict[str, str]]:
    verify_resources([relative])
    try:
        value = json.loads(data_path(relative).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"The MHC reference {relative} is malformed") from error
    if not isinstance(value, dict):
        raise IntegrityError(f"The MHC reference {relative} is malformed")
    return value


def _reference(sequence_convention: str, species: str) -> dict[str, dict[str, str]]:
    convention = _convention(sequence_convention)
    species = validate_species(species)
    if convention == "v1":
        if species != "human":
            raise ValidationError("The selected model does not support mouse reconstruction")
        return _read_reference(_REFERENCE_FILES[convention])
    additions = _read_reference(_SPECIES_V2_FILE)
    try:
        selected = additions[species]
    except (KeyError, TypeError) as error:
        raise IntegrityError("The species MHC reference is malformed") from error
    if not isinstance(selected, dict):
        raise IntegrityError("The species MHC reference is malformed")
    if species == "mouse":
        return selected
    reference = _read_reference(_REFERENCE_FILES[convention])
    overlap = set(reference).intersection(selected)
    if overlap:
        raise IntegrityError(f"Human V2 MHC reference keys overlap: {sorted(overlap)}")
    return {**reference, **selected}


def lookup_hla(allele: str, sequence_convention: str, *, species: str = "human") -> tuple[str, str]:
    """Return the exact stored class I or explicit class II chain pair."""
    species = validate_species(species)
    key = normalize_allele_key(allele, species=species)
    reference = _reference(sequence_convention, species)
    try:
        record = reference[key]
        hla_a = record["HLA_a"]
        hla_b = record["HLA_b"]
    except (KeyError, TypeError) as error:
        near = sorted(
            candidate for candidate in reference if candidate.split("__", 1)[0][:1] == key[:1]
        )[:12]
        raise ValidationError(
            f"HLA allele {allele!r} normalizes to {key!r} and is not available. "
            f"Same-locus alleles include {near}"
        ) from error
    if not isinstance(hla_a, str) or not hla_a or not isinstance(hla_b, str) or not hla_b:
        raise IntegrityError(f"The HLA reference entry is malformed: {key}")
    return hla_a, hla_b


def list_alleles(sequence_convention: str, *, species: str = "human") -> list[str]:
    """List exact MHC lookup keys available for a convention and species."""
    species = validate_species(species)
    available = []
    for key in _reference(sequence_convention, species):
        try:
            normalize_allele_key(key, species=species)
        except ValidationError:
            continue
        available.append(key)
    return sorted(available)
