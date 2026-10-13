"""Prepare full sequence entries from TCR genes and an HLA allele."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from DecoderTCR.core import IntegrityError, ValidationError

from ._resources import (
    STITCHR_RESOURCES_BY_SPECIES,
    data_path,
    resource_identity,
    validate_species,
    verify_resources,
)
from .hla import lookup_hla
from .tcr import stitch_tcrs

_ALIASES = {
    "trav": ("trav", "tcra_v", "va", "v_alpha"),
    "traj": ("traj", "tcra_j", "ja", "j_alpha"),
    "cdr3a": ("cdr3a_extended", "cdr3a", "tcra_cdr3", "cdr3_a", "cdr3_alpha"),
    "trbv": ("trbv", "tcrb_v", "vb", "v_beta"),
    "trbj": ("trbj", "tcrb_j", "jb", "j_beta"),
    "cdr3b": ("cdr3b_extended", "cdr3b", "tcrb_cdr3", "cdr3_b", "cdr3_beta"),
    "hla": ("hla", "allele", "hla_allele", "mhc"),
    "peptide": ("peptide", "epitope", "pep"),
    "name": ("name", "id", "tcr_name"),
    "label": ("label", "binder", "y"),
    "species": ("species", "organism"),
}
_REQUIRED = ("trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b", "hla", "peptide")
_TCR_FIELDS = ("trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b")
_TCR_RESULT_FIELDS = (
    "normalized_TRAV",
    "normalized_TRAJ",
    "normalized_TRBV",
    "normalized_TRBJ",
    "TRAV",
    "TRAJ",
    "TRBV",
    "TRBJ",
    "TRAC",
    "TRBC",
    "TRA_leader",
    "TRB_leader",
    "warnings",
)
_V2_LEADERS = "tcr_leaders_v2.txt"
_MOUSE_V2_LEADERS = "tcr_leaders_mouse_v2.json"
_SPECIES_V2_HLA = "hla_v2_species.json"


def _sequence_convention(value: str) -> str:
    convention = str(value).strip()
    if convention not in {"v1", "v2"}:
        raise ValidationError("sequence_convention must be 'v1' or 'v2'")
    return convention


def _load_frame(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy(deep=True)
    if isinstance(data, Mapping):
        return pd.DataFrame([dict(data)])
    if isinstance(data, list):
        if not all(isinstance(row, Mapping) for row in data):
            raise ValidationError("list inputs must contain only dictionaries")
        return pd.DataFrame([dict(row) for row in data])
    if isinstance(data, (str, Path)):
        path = Path(data)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        raise ValidationError("Gene input paths must end in .csv, .parquet, or .pq")
    raise ValidationError("Data must be a CSV or Parquet path, DataFrame, dictionary, or list")


def _resolve_columns(frame: pd.DataFrame) -> dict[str, object]:
    lowered: dict[str, object] = {}
    for column in frame.columns:
        key = str(column).casefold()
        if key in lowered:
            raise ValidationError(f"Duplicate case-insensitive input header: {column!r}")
        lowered[key] = column

    resolved: dict[str, object] = {}
    for canonical, aliases in _ALIASES.items():
        matches = [(alias, lowered[alias]) for alias in aliases if alias in lowered]
        if not matches:
            continue
        if canonical in {"cdr3a", "cdr3b"} and matches[0][0].endswith("_extended"):
            resolved[canonical] = matches[0][1]
            continue
        if len(matches) != 1:
            headers = [str(column) for _, column in matches]
            raise ValidationError(f"Ambiguous aliases for {canonical}: {headers}")
        resolved[canonical] = matches[0][1]

    missing = [field for field in _REQUIRED if field not in resolved]
    if missing:
        raise ValidationError(f"Missing required gene input columns: {missing}")
    return resolved


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _row_tcr_failure(row: dict[str, str]) -> str:
    missing = [field for field in _TCR_FIELDS if not row[field]]
    if missing:
        return f"blank required TCR fields: {missing}"
    for field in _TCR_FIELDS:
        value = row[field]
        if any(character in value for character in ("\t", "\r", "\n")):
            return f"{field} contains a TSV control character"
        if "," in value or "%" in value:
            return f"{field} requests multiple thimble outputs"
    return ""


def _leaders() -> tuple[str, ...]:
    verify_resources([_V2_LEADERS])
    values = [
        line.strip()
        for line in data_path(_V2_LEADERS).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise IntegrityError("The V2 TCR leader reference is empty")
    return tuple(sorted(set(values), key=len, reverse=True))


def _mouse_leaders() -> dict[str, str]:
    verify_resources([_MOUSE_V2_LEADERS])
    try:
        document = json.loads(data_path(_MOUSE_V2_LEADERS).read_text(encoding="utf-8"))
        values = document["leaders"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise IntegrityError("The mouse V2 TCR leader reference is malformed") from error
    if not isinstance(values, dict) or not values:
        raise IntegrityError("The mouse V2 TCR leader reference is malformed")
    if not all(
        isinstance(gene, str)
        and gene
        and isinstance(peptide, str)
        and peptide
        and peptide.startswith("M")
        for gene, peptide in values.items()
    ):
        raise IntegrityError("The mouse V2 TCR leader reference is malformed")
    return values


def _strip_leader(sequence: str, leaders: tuple[str, ...]) -> str:
    for leader in leaders:
        if sequence.startswith(leader):
            return sequence[len(leader) :]
    raise ValidationError("No IMGT leader is a prefix of the reconstructed chain")


def _strip_mouse_leader(sequence: str, selected_gene: str, leaders: dict[str, str]) -> str:
    reference_gene = selected_gene.removesuffix("(L)")
    try:
        leader = leaders[reference_gene]
    except KeyError as error:
        raise ValidationError(
            f"No verified mouse leader peptide for reported allele {selected_gene!r}"
        ) from error
    if not sequence.startswith(leader):
        raise ValidationError(
            f"Verified mouse leader for {selected_gene!r} is not a reconstructed-chain prefix"
        )
    return sequence[len(leader) :]


def _metadata(sequence_convention: str, species: str) -> dict:
    species = validate_species(species)
    if sequence_convention == "v1" and species != "human":
        raise ValidationError("The selected model does not support mouse reconstruction")
    convention_files = ["hla_v1.json"]
    if sequence_convention == "v2":
        convention_files = [_SPECIES_V2_HLA, _MOUSE_V2_LEADERS]
        if species == "human":
            convention_files = ["hla_v2.json", _SPECIES_V2_HLA, _V2_LEADERS]
    selected = [*STITCHR_RESOURCES_BY_SPECIES[species], *convention_files]
    identity = resource_identity(selected)
    return {
        "sequence_convention": sequence_convention,
        "species": species,
        "engine": {"name": "Stitchr", "version": "1.3.1", "interface": "thimble"},
        "resource_bundle_sha256": identity["manifest_sha256"],
        "resources": identity["resources"],
    }


def reference_identity(sequence_convention: str, *, species: str = "human") -> dict:
    """Return verified identities for one reconstruction convention."""
    species = validate_species(species)
    metadata = _metadata(_sequence_convention(sequence_convention), species)
    return {
        "species": species,
        "resource_bundle_sha256": metadata["resource_bundle_sha256"],
        "resources": metadata["resources"],
    }


def _output_name(frame: pd.DataFrame, preferred: str) -> str:
    lower = {str(column).casefold() for column in frame.columns}
    if preferred.casefold() not in lower:
        return preferred
    prefixed = f"reconstruction_{preferred}"
    if prefixed.casefold() in lower:
        raise ValidationError(
            f"input contains both {preferred!r} and reserved output column {prefixed!r}"
        )
    return prefixed


def _row_species_failure(row: dict[str, str], species: str) -> str:
    declared = row.get("species", "")
    if not declared:
        return ""
    try:
        row_species = validate_species(declared)
    except ValidationError:
        return f"unsupported row species: {declared!r}"
    if row_species != species:
        return f"row species {row_species!r} does not match requested species {species!r}"
    return ""


def _build(
    data, sequence_convention: str, *, species: str = "human"
) -> tuple[pd.DataFrame, list[dict | None]]:
    convention = _sequence_convention(sequence_convention)
    species = validate_species(species)
    if convention == "v1" and species != "human":
        raise ValidationError("The selected model does not support mouse reconstruction")
    frame = _load_frame(data)
    columns = _resolve_columns(frame)
    metadata = _metadata(convention, species)

    canonical_rows: list[dict[str, str]] = []
    pre_failures: list[str] = []
    stitch_input: list[dict] = []
    stitch_positions: list[int] = []
    for position in range(len(frame)):
        row = {field: _text(frame.iloc[position][column]) for field, column in columns.items()}
        canonical_rows.append(row)
        failure = _row_species_failure(row, species) or _row_tcr_failure(row)
        pre_failures.append(failure)
        if not failure:
            stitch_positions.append(position)
            stitch_input.append({field: row[field] for field in _TCR_FIELDS})

    stitched: list[dict | None] = [None] * len(frame)
    if stitch_input:
        values = stitch_tcrs(stitch_input, species=species)
        if len(values) != len(stitch_positions):
            raise IntegrityError("Thimble result count does not match submitted row count")
        for position, value in zip(stitch_positions, values, strict=True):
            stitched[position] = value

    leaders = _leaders() if convention == "v2" and species == "human" else ()
    mouse_leaders = _mouse_leaders() if convention == "v2" and species == "mouse" else {}
    records: list[dict] = []
    entries: list[dict | None] = []
    for position, row in enumerate(canonical_rows):
        tcr = stitched[position]
        tcr_ok = tcr is not None and bool(tcr["ok"])
        tcr_reason = pre_failures[position] or (tcr["reason"] if tcr is not None else "")
        tcr_a = tcr["TCR_a"] if tcr is not None else ""
        tcr_b = tcr["TCR_b"] if tcr is not None else ""
        if tcr_ok and convention == "v2":
            try:
                if species == "human":
                    tcr_a = _strip_leader(tcr_a, leaders)
                    tcr_b = _strip_leader(tcr_b, leaders)
                else:
                    tcr_a = _strip_mouse_leader(tcr_a, tcr["TRA_leader"], mouse_leaders)
                    tcr_b = _strip_mouse_leader(tcr_b, tcr["TRB_leader"], mouse_leaders)
            except ValidationError as error:
                tcr_ok = False
                tcr_reason = str(error)

        hla_ok = bool(row["hla"])
        hla_reason = "" if hla_ok else "blank required HLA field"
        hla_a = ""
        hla_b = ""
        if hla_ok:
            try:
                hla_a, hla_b = lookup_hla(row["hla"], convention, species=species)
            except ValidationError as error:
                hla_ok = False
                hla_reason = str(error)

        peptide_ok = bool(row["peptide"])
        peptide_reason = "" if peptide_ok else "blank required peptide field"
        ok = bool(tcr_ok and hla_ok and peptide_ok)
        reasons = [value for value in (tcr_reason, hla_reason, peptide_reason) if value]
        record = {
            "TCR_a": tcr_a,
            "TCR_b": tcr_b,
            "HLA_a": hla_a,
            "HLA_b": hla_b,
            "Peptide": row["peptide"],
            "reconstruction_ok": ok,
            "reconstruction_reason": " | ".join(reasons),
            "tcr_ok": bool(tcr_ok),
            "tcr_reason": tcr_reason,
            "hla_ok": bool(hla_ok),
            "hla_reason": hla_reason,
        }
        for field in _TCR_RESULT_FIELDS:
            record[field] = tcr.get(field, "") if tcr is not None else ""
        records.append(record)

        if not ok:
            entries.append(None)
            continue
        cdr3a_start = tcr_a.find(row["cdr3a"])
        cdr3b_start = tcr_b.find(row["cdr3b"])
        if cdr3a_start < 0 or cdr3b_start < 0:
            record["reconstruction_ok"] = False
            record["reconstruction_reason"] = "CDR3 is absent from a reconstructed chain"
            entries.append(None)
            continue
        label = frame.iloc[position][columns["label"]] if "label" in columns else 0
        entries.append(
            {
                "sequences": {
                    "HLA_a": hla_a,
                    "HLA_b": hla_b,
                    "Peptide": row["peptide"],
                    "TCR_a": tcr_a,
                    "TCR_b": tcr_b,
                },
                "pocket_idx": {
                    "HLA_a": [],
                    "HLA_b": [],
                    "CDRa": [[cdr3a_start, cdr3a_start + len(row["cdr3a"]) - 1]],
                    "CDRb": [[cdr3b_start, cdr3b_start + len(row["cdr3b"]) - 1]],
                },
                "annotation_contract": "inclusive-end",
                "meta_data": {
                    "label": label,
                    "epitope": row["peptide"],
                    "hla_allele": row["hla"],
                    "species": species,
                },
                "reconstruction": metadata,
            }
        )

    output_fields = [
        "TCR_a",
        "TCR_b",
        "HLA_a",
        "HLA_b",
        "Peptide",
        "reconstruction_ok",
        "reconstruction_reason",
        "tcr_ok",
        "tcr_reason",
        "hla_ok",
        "hla_reason",
        *_TCR_RESULT_FIELDS,
    ]
    names = {field: _output_name(frame, field) for field in output_fields}
    for field in output_fields:
        frame[names[field]] = [record[field] for record in records]
    frame.attrs["reconstruction"] = metadata
    frame.attrs["reconstruction_columns"] = names
    return frame, entries


def reconstruct_components(
    data, sequence_convention: str, *, species: str = "human"
) -> pd.DataFrame:
    """Reconstruct sequence columns while preserving all original rows and fields."""
    frame, _ = _build(data, sequence_convention, species=species)
    return frame


def prepare_gene_inputs(
    data, sequence_convention: str, *, species: str = "human"
) -> tuple[pd.DataFrame, list[dict | None]]:
    """Return the reconstruction frame and aligned canonical scoring entries."""
    return _build(data, sequence_convention, species=species)
