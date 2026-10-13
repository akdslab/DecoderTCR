"""Reconstruct full TCR alpha and beta chains with pinned thimble behavior."""

from __future__ import annotations

import csv
import importlib.metadata
import re
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from DecoderTCR.core import IntegrityError, ReleaseError, ValidationError

from ._resources import (
    STITCHR_RESOURCES_BY_SPECIES,
    data_path,
    resource_identity,
    stitchr_species,
    validate_species,
    verify_resources,
)

_HEADERS = [
    "TCR_name",
    "TRAV",
    "TRAJ",
    "TRA_CDR3",
    "TRBV",
    "TRBJ",
    "TRB_CDR3",
    "TRAC",
    "TRBC",
    "TRA_leader",
    "TRB_leader",
    "Linker",
    "Link_order",
    "TRA_5_prime_seq",
    "TRA_3_prime_seq",
    "TRB_5_prime_seq",
    "TRB_3_prime_seq",
]
_REQUIRED_OUTPUT = {
    "TCR_name",
    "TRA_aa",
    "TRB_aa",
    "TRAV",
    "TRAJ",
    "TRBV",
    "TRBJ",
    "TRAC",
    "TRBC",
    "TRA_leader",
    "TRB_leader",
    "Warnings/Errors",
}
_GENE_FIELDS = ("trav", "traj", "trbv", "trbj")
_INPUT_FIELDS = ("trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b")


@lru_cache(maxsize=2)
def _parsed_imgt_gene_sets(
    resource_paths: tuple[str, ...], resource_hashes: tuple[str, ...]
) -> tuple[frozenset[str], frozenset[str]]:
    genes: set[str] = set()
    star01_function: dict[str, str] = {}
    any_functional: set[str] = set()
    for relative in resource_paths:
        with data_path(relative).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith(">"):
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                gene_allele = parts[1]
                gene_name, _, allele = gene_allele.partition("*")
                genes.add(gene_name)
                function = parts[3].strip() if len(parts) >= 4 else ""
                if allele == "01":
                    star01_function[gene_name] = function
                if function == "F":
                    any_functional.add(gene_name)
    functional = {
        gene
        for gene in genes
        if star01_function.get(gene) == "F"
        or (gene not in star01_function and gene in any_functional)
    }
    return frozenset(genes), frozenset(functional)


def _imgt_gene_sets(species: str) -> tuple[frozenset[str], frozenset[str]]:
    species = validate_species(species)
    selected = list(STITCHR_RESOURCES_BY_SPECIES[species][:4])
    identity = resource_identity(selected)
    hashes = tuple(identity["resources"][relative] for relative in selected)
    return _parsed_imgt_gene_sets(tuple(selected), hashes)


def _resolve_family(gene_only: str, gene_sets: tuple[frozenset[str], frozenset[str]]) -> str:
    genes, functional = gene_sets
    subgenes = sorted(
        gene for gene in genes if gene.startswith(gene_only + "-") or gene == gene_only
    )
    functional_subgenes = [gene for gene in subgenes if gene in functional]
    chosen = functional_subgenes or subgenes
    return chosen[0] if chosen else gene_only


def _immrep_to_imgt(gene: str, gene_sets: tuple[frozenset[str], frozenset[str]]) -> str:
    genes, _ = gene_sets
    value = re.sub(r"^TCR([AB])([VJ])", r"TR\1\2", gene)
    if value.endswith("-X"):
        match = re.match(r"(TR[AB][VJ])(\d+)-X", value)
        gene_only = f"{match.group(1)}{int(match.group(2))}" if match else value[:-2]
        return _resolve_family(gene_only, gene_sets)
    match = re.match(r"(TR[AB][VJ])(\d+)-(\d+)", value)
    if not match:
        return value
    prefix, number, suffix = match.group(1), str(int(match.group(2))), str(int(match.group(3)))
    if prefix == "TRAJ":
        return f"{prefix}{number}"
    if prefix == "TRBJ":
        return f"{prefix}{number}-{suffix}"
    subgene = f"{prefix}{number}-{suffix}"
    gene_only = f"{prefix}{number}"
    if subgene in genes:
        return subgene
    if gene_only in genes:
        return gene_only
    for known in genes:
        if known.startswith(f"{prefix}{number}/") or known.startswith(
            f"{prefix}{number}-{suffix}/"
        ):
            return known
    return subgene


def _normalize_gene(gene: str, gene_sets: tuple[frozenset[str], frozenset[str]]) -> str:
    if not isinstance(gene, str):
        raise ValidationError("gene must be a string")
    value = gene.strip()
    if not value:
        return value
    if value.upper().startswith("TCR"):
        return _immrep_to_imgt(value, gene_sets)

    genes, _ = gene_sets
    allele = ""
    core = value
    if "*" in value:
        core, allele = value.split("*", 1)
    match = re.match(r"(TR[AB][VJ])0*(\d+)(?:-0*(\d+))?(.*)$", core, re.I)
    if match:
        prefix = match.group(1).upper()
        number = match.group(2)
        subgene = match.group(3)
        tail = match.group(4) or ""
        core = f"{prefix}{number}" + (f"-{subgene}" if subgene else "") + tail
    candidate = f"{core}*{allele}" if allele else core
    if core in genes:
        return candidate if allele else core
    family = re.match(r"(TR[AB][VJ])(\d+)", core)
    if family:
        resolved = _resolve_family(f"{family.group(1)}{family.group(2)}", gene_sets)
        if resolved in genes:
            return resolved
    return core


def normalize_gene(gene: str, *, species: str = "human") -> str:
    """Normalize public and IMMREP V or J gene notation for thimble."""
    species = validate_species(species)
    return _normalize_gene(gene, _imgt_gene_sets(species))


def _require_stitchr() -> None:
    try:
        version = importlib.metadata.version("stitchr")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValidationError("TCR reconstruction requires Stitchr 1.3.1") from error
    if version != "1.3.1":
        raise ValidationError(f"TCR reconstruction requires Stitchr 1.3.1, found {version}")


def _validate_scalar(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string")
    if any(character in value for character in ("\t", "\r", "\n")):
        raise ValidationError(f"{field} contains a TSV control character")
    if "," in value or "%" in value:
        raise ValidationError(f"{field} requests multiple thimble outputs")
    return value.strip()


def stitch_tcrs(rows: list[dict], *, species: str = "human") -> list[dict]:
    """Return untrimmed full receptor chains for receptor embedding."""
    species = validate_species(species)
    if not isinstance(rows, list):
        raise ValidationError("rows must be a list of dictionaries")
    if not rows:
        return []
    resources = STITCHR_RESOURCES_BY_SPECIES[species]
    gene_sets = _imgt_gene_sets(species)
    verify_resources(list(resources[4:]))
    _require_stitchr()

    prepared: list[dict[str, str]] = []
    for index, source in enumerate(rows):
        if not isinstance(source, dict):
            raise ValidationError(f"row {index} is not a dictionary")
        values = {field: _validate_scalar(source.get(field, ""), field) for field in _INPUT_FIELDS}
        normalized = {field: _normalize_gene(values[field], gene_sets) for field in _GENE_FIELDS}
        prepared.append({**values, **normalized, "_id": f"row_{index:09d}"})

    worker_error = None
    cleanup_error = None
    try:
        with tempfile.TemporaryDirectory(prefix="decodertcr-thimble-") as directory:
            input_path = Path(directory) / "input.tsv"
            output_path = Path(directory) / "output.tsv"
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(_HEADERS)
                for row in prepared:
                    writer.writerow(
                        [
                            row["_id"],
                            row["trav"],
                            row["traj"],
                            row["cdr3a"],
                            row["trbv"],
                            row["trbj"],
                            row["cdr3b"],
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                        ]
                    )
            command = [
                sys.executable,
                "-m",
                "DecoderTCR.reconstruct._thimble_worker",
                "-in",
                str(input_path),
                "-o",
                str(output_path),
                "-s",
                stitchr_species(species),
                "-sc",
            ]
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            if process.returncode != 0:
                detail = (process.stderr or process.stdout).strip()[:1500]
                worker_error = ReleaseError(
                    f"Thimble failed with exit {process.returncode}: {detail}"
                )
            else:
                try:
                    with output_path.open(encoding="utf-8", newline="") as handle:
                        reader = csv.DictReader(handle, delimiter="\t")
                        if (
                            reader.fieldnames is None
                            or len(reader.fieldnames) != len(set(reader.fieldnames))
                            or not _REQUIRED_OUTPUT.issubset(reader.fieldnames)
                        ):
                            raise IntegrityError("Thimble output has an unexpected header")
                        output_rows = list(reader)
                except (OSError, csv.Error) as error:
                    raise IntegrityError("Thimble output is missing or malformed") from error
    except OSError as error:
        if worker_error is None:
            raise
        cleanup_error = error
    if worker_error is not None:
        if cleanup_error is not None:
            raise worker_error from cleanup_error
        raise worker_error

    expected_ids = {row["_id"] for row in prepared}
    by_id: dict[str, dict[str, str]] = {}
    for row in output_rows:
        if None in row or any(value is None for value in row.values()):
            raise IntegrityError("Thimble output has a malformed row")
        identifier = row.get("TCR_name", "")
        if identifier not in expected_ids:
            raise IntegrityError(f"Thimble returned an unexpected row identifier: {identifier!r}")
        if identifier in by_id:
            raise IntegrityError(f"Thimble returned duplicate row identifier: {identifier!r}")
        by_id[identifier] = row
    if set(by_id) != expected_ids:
        missing = sorted(expected_ids.difference(by_id))
        raise IntegrityError(f"Thimble omitted expected rows: {missing}")

    results: list[dict] = []
    for source, prepared_row in zip(rows, prepared, strict=True):
        row = by_id[prepared_row["_id"]]
        tcr_a = row.get("TRA_aa", "") or ""
        tcr_b = row.get("TRB_aa", "") or ""
        warning = row.get("Warnings/Errors", "") or ""
        reason = ""
        if not tcr_a or not tcr_b:
            reason = "empty or unusable chain"
        elif "*" in tcr_a or "*" in tcr_b:
            reason = "premature stop in reconstructed chain"
        results.append(
            {
                "name": source.get("name", ""),
                "TCR_a": tcr_a,
                "TCR_b": tcr_b,
                "ok": not reason,
                "reason": reason,
                "warnings": warning,
                "normalized_TRAV": prepared_row["trav"],
                "normalized_TRAJ": prepared_row["traj"],
                "normalized_TRBV": prepared_row["trbv"],
                "normalized_TRBJ": prepared_row["trbj"],
                "TRAV": row.get("TRAV", "") or "",
                "TRAJ": row.get("TRAJ", "") or "",
                "TRBV": row.get("TRBV", "") or "",
                "TRBJ": row.get("TRBJ", "") or "",
                "TRAC": row.get("TRAC", "") or "",
                "TRBC": row.get("TRBC", "") or "",
                "TRA_leader": row.get("TRA_leader", "") or "",
                "TRB_leader": row.get("TRB_leader", "") or "",
            }
        )
    return results
