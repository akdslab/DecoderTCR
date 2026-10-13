"""Feature extraction and conditional profiles for resolved model handles."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import (
    AA20,
    AA20_IDS,
    MASK_IDX,
    SEQUENCE_ORDER,
)
from .core import ValidationError
from .tokenizer import TCRpMHCTokenizer

PLACEHOLDER_AA = "A"
REGIONS = ("peptide", "cdr3a", "cdr3b", "cdr3")
_INTERNAL_REGIONS = (*REGIONS, "cdr")

_ALIASES = {
    "HLA_a": ("hla_a", "hla", "hla_chain", "hla_seq", "mhc", "mhc_a"),
    "HLA_b": ("hla_b", "b2m", "beta2m", "mhc_b"),
    "Peptide": ("peptide", "epitope", "pep"),
    "TCR_a": ("tcr_a", "tcra", "tcr_alpha"),
    "TCR_b": ("tcr_b", "tcrb", "tcr_beta"),
}
_CDR_KEY = {"TCR_a": "CDRa", "TCR_b": "CDRb"}


def _model_attrs(loaded, operation: str) -> dict:
    identity = dict(loaded.info)
    identity["operation"] = operation
    return identity


def _attach_model_attrs(
    frame: pd.DataFrame, loaded, operation: str, entry: dict | None = None
) -> pd.DataFrame:
    frame.attrs["reproduction"] = _model_attrs(loaded, operation)
    reconstruction = (entry or {}).get("reconstruction")
    if reconstruction is not None:
        if not isinstance(reconstruction, Mapping):
            raise ValidationError("reconstruction metadata must be a mapping")
        frame.attrs["reconstruction"] = dict(reconstruction)
    return frame


def _scalar_sequence(value, name: str) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return ""
    if not isinstance(value, str):
        raise ValidationError(f"{name} must contain an amino-acid sequence string")
    return value.strip()


def _validate_unique_input_labels(labels) -> None:
    lowered = set()
    for label in labels:
        key = str(label).casefold()
        if key in lowered:
            raise ValidationError(f"Duplicate case-insensitive input header: {label!r}")
        lowered.add(key)


def _as_single_mapping(data) -> dict:
    if isinstance(data, pd.DataFrame):
        if len(data) != 1:
            raise ValidationError(f"Expected one complex, got {len(data)} rows")
        _validate_unique_input_labels(data.columns)
        return data.iloc[0].to_dict()
    if isinstance(data, pd.Series):
        _validate_unique_input_labels(data.index)
        return data.to_dict()
    if not isinstance(data, Mapping):
        raise ValidationError(
            f"Expected a mapping, Series, or one-row DataFrame, got {type(data).__name__}"
        )
    return dict(data)


def _flat_sequences(row: Mapping, *, require_peptide: bool) -> dict[str, str]:
    lowered_names = [str(key).lower() for key in row]
    if len(lowered_names) != len(set(lowered_names)):
        raise ValidationError("Input fields must be unique ignoring capitalization")
    lower = {str(key).lower(): value for key, value in row.items()}
    sequences = {}
    for canonical, aliases in _ALIASES.items():
        matches = [alias for alias in aliases if alias in lower]
        if len(matches) > 1:
            raise ValidationError(f"Provide only one sequence field for {canonical}")
        value = lower[matches[0]] if matches else ""
        sequences[canonical] = _scalar_sequence(value, canonical)
    if require_peptide and not sequences["Peptide"]:
        raise ValidationError("The entry has no peptide")
    return sequences


def _prepared_single(data, sequence_convention: str, *, require_peptide: bool = True) -> dict:
    row = _as_single_mapping(data)
    if "sequences" in row:
        sequences = row["sequences"]
        if not isinstance(sequences, Mapping):
            raise ValidationError("sequences must be a mapping")
        metadata = row.get("meta_data", {})
        if not isinstance(metadata, Mapping):
            raise ValidationError("meta_data must be a mapping")
        normalized = {}
        for key in SEQUENCE_ORDER:
            normalized[key] = _scalar_sequence(sequences.get(key, ""), key)
        if require_peptide and not normalized["Peptide"]:
            raise ValidationError("The entry has no peptide")
        pocket_idx = row.get("pocket_idx", {})
        contracts = [
            value
            for value in (row.get("annotation_contract"), metadata.get("annotation_contract"))
            if value is not None
        ]
        declarations = [
            value
            for value in (row.get("sequence_convention"), metadata.get("sequence_convention"))
            if value is not None
        ]
        if len(contracts) > 1 and contracts[0] != contracts[1]:
            raise ValidationError("Conflicting annotation_contract declarations")
        if len(declarations) > 1 and declarations[0] != declarations[1]:
            raise ValidationError("Conflicting sequence_convention declarations")
        contract = contracts[0] if contracts else None
        declared = declarations[0] if declarations else None
    else:
        normalized = _flat_sequences(row, require_peptide=require_peptide)
        pocket_idx = row.get("pocket_idx", {})
        contract = row.get("annotation_contract")
        declared = row.get("sequence_convention")
    if declared is not None and declared != sequence_convention:
        raise ValidationError("Input sequence convention does not match the selected model")
    if pocket_idx is None or (isinstance(pocket_idx, float) and np.isnan(pocket_idx)):
        pocket_idx = {}
    if not isinstance(pocket_idx, Mapping):
        raise ValidationError("pocket_idx must be a mapping")
    if contract is not None and contract not in ("half-open", "inclusive-end"):
        raise ValidationError("annotation_contract must be 'half-open' or 'inclusive-end'")
    entry = {
        "sequences": normalized,
        "pocket_idx": dict(pocket_idx),
        "annotation_contract": contract,
    }
    reconstruction = row.get("reconstruction")
    if reconstruction is not None:
        if not isinstance(reconstruction, Mapping):
            raise ValidationError("reconstruction metadata must be a mapping")
        entry["reconstruction"] = dict(reconstruction)
    return entry


def _gene_single(
    loaded,
    data,
    *,
    require_peptide: bool = True,
    species: str = "human",
) -> dict:
    from .reconstruct.components import prepare_gene_inputs

    row = _as_single_mapping(data)
    _, entries = prepare_gene_inputs(row, loaded.sequence_convention, species=species)
    if len(entries) != 1:
        raise ValidationError(f"Expected one complex, got {len(entries)} rows")
    if entries[0] is None:
        raise ValidationError("Could not reconstruct the requested complex")
    entry = dict(entries[0])
    entry["annotation_contract"] = "inclusive-end"
    if require_peptide and not entry["sequences"].get("Peptide"):
        raise ValidationError("The reconstructed entry has no peptide")
    return entry


def _single_entry(
    loaded,
    data,
    *,
    require_peptide: bool = True,
    species: str = "human",
    prepared: bool = False,
) -> dict:
    if prepared:
        return _prepared_single(data, loaded.sequence_convention, require_peptide=require_peptide)
    return _gene_single(
        loaded,
        data,
        require_peptide=require_peptide,
        species=species,
    )


def _entries(loaded, data, *, species: str = "human", prepared: bool = False) -> list[dict]:
    from .reconstruct._resources import validate_species

    species = validate_species(species)
    if prepared:
        if isinstance(data, pd.DataFrame):
            records = data.to_dict("records")
        elif isinstance(data, pd.Series):
            records = [data.to_dict()]
        elif isinstance(data, Mapping):
            records = [data]
        else:
            try:
                records = list(data)
            except TypeError as exc:
                raise ValidationError("Prepared inputs must be records or a dataframe") from exc
        if not all(isinstance(record, Mapping) for record in records):
            raise ValidationError("Prepared inputs must be records or a dataframe")
        structured = ["sequences" in record for record in records]
        if any(structured) and not all(structured):
            raise ValidationError("Do not mix flat and structured sequence records")
        return [_prepared_single(record, loaded.sequence_convention) for record in records]
    from .reconstruct.components import prepare_gene_inputs

    _, entries = prepare_gene_inputs(data, loaded.sequence_convention, species=species)
    failed = [index for index, entry in enumerate(entries) if entry is None]
    if failed:
        raise ValidationError(f"Could not reconstruct input rows: {failed}")
    result = []
    for entry in entries:
        copied = dict(entry)
        copied["annotation_contract"] = "inclusive-end"
        result.append(copied)
    return result


def _masked_entry(
    loaded,
    data,
    region: str,
    length: int | None,
    *,
    species: str = "human",
    prepared: bool = False,
) -> dict:
    from .reconstruct._resources import validate_species

    species = validate_species(species)
    if region not in _INTERNAL_REGIONS:
        raise ValidationError(f"Unknown region {region!r}. Choose from {list(REGIONS)}")
    if region == "peptide":
        if not isinstance(length, int) or isinstance(length, bool) or length < 1:
            raise ValidationError("Peptide profiling needs a positive integer length")
        if not prepared:
            row = _as_single_mapping(data)
            peptide_aliases = set(_ALIASES["Peptide"])
            row = {
                key: value for key, value in row.items() if str(key).lower() not in peptide_aliases
            }
            row["peptide"] = PLACEHOLDER_AA * length
            entry = _gene_single(loaded, row, species=species)
        else:
            entry = _prepared_single(data, loaded.sequence_convention, require_peptide=False)
            entry["sequences"]["Peptide"] = PLACEHOLDER_AA * length
    else:
        if length is not None:
            raise ValidationError("length only applies to region='peptide'")
        entry = _single_entry(loaded, data, species=species, prepared=prepared)
    if not (
        entry["sequences"]["HLA_a"] or entry["sequences"]["TCR_a"] or entry["sequences"]["TCR_b"]
    ):
        raise ValidationError("Nothing to condition on. Supply an HLA_a or a TCR chain")
    return entry


def _validated_range(limits, chain: str, chain_length: int, inclusive: bool) -> tuple[int, int]:
    if (
        not isinstance(limits, (list, tuple))
        or len(limits) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in limits)
    ):
        raise ValidationError(f"Invalid region annotation for {chain}")
    start, end = limits
    valid = 0 <= start <= end < chain_length if inclusive else 0 <= start <= end <= chain_length
    if not valid:
        raise ValidationError(f"Invalid region annotation for {chain}")
    return start, end


def region_positions(tok: TCRpMHCTokenizer, region: str, entry: dict) -> np.ndarray:
    """Return sequence positions using the operation-specific endpoint contract."""
    if region not in _INTERNAL_REGIONS:
        raise ValidationError(f"Unknown region {region!r}. Choose from {list(REGIONS)}")
    if region == "peptide":
        positions = np.asarray(tok.peptide_indices, dtype=np.intp)
    else:
        required = "half-open" if region == "cdr" else "inclusive-end"
        contract = entry.get("annotation_contract")
        if contract != required:
            raise ValidationError(f"Region {region!r} requires annotation_contract={required!r}")
        chains = (
            ("TCR_a", "TCR_b")
            if region in ("cdr", "cdr3")
            else (("TCR_a",) if region == "cdr3a" else ("TCR_b",))
        )
        selected = []
        annotation = entry.get("pocket_idx") or {}
        for chain in chains:
            span = tok.region_slices.get(chain)
            ranges = annotation.get(_CDR_KEY[chain], [])
            if span is None or not ranges:
                continue
            if not isinstance(ranges, (list, tuple)):
                raise ValidationError(f"Invalid region annotation for {chain}")
            chosen = ranges if region == "cdr" else [ranges[2] if len(ranges) >= 3 else ranges[-1]]
            for limits in chosen:
                start, end = _validated_range(
                    limits, chain, span.stop - span.start, inclusive=region != "cdr"
                )
                stop = end + 1 if region != "cdr" else end
                selected.extend(range(span.start + start, span.start + stop))
        positions = np.asarray(selected, dtype=np.intp)
    if positions.size == 0:
        raise ValidationError(f"Region {region!r} is empty for this complex")
    return positions


@torch.inference_mode()
def _embed(
    loaded,
    data,
    *,
    pool="mean",
    layer=None,
    species="human",
    prepared=False,
):
    """Return final or selected-layer residue embeddings from a resolved handle."""
    if pool not in ("mean", "regions", None):
        raise ValidationError("pool must be 'mean', 'regions', or None")
    entries = _entries(loaded, data, species=species, prepared=prepared)
    requested = loaded.num_layers if layer in (None, -1) else layer
    if loaded.backbone == "esm2":
        if not isinstance(requested, int) or not 0 <= requested <= loaded.num_layers:
            raise ValidationError(
                f"layer must be in [0, {loaded.num_layers}] for the ESM2 backbone"
            )
    elif layer not in (None, -1, loaded.num_layers):
        warnings.warn(
            "The ESMC backbone exposes only its final layer. layer is ignored.",
            stacklevel=2,
        )

    results = []
    for entry in entries:
        tok = TCRpMHCTokenizer(entry, mask_region=None)
        output = loaded.module(
            tok.original_ids.unsqueeze(0).to(loaded.device),
            repr_layers=[requested],
            return_contacts=False,
        )
        representations = output["representations"]
        representation = (
            representations[requested] if loaded.backbone == "esm2" else representations[-1]
        )
        values = representation[0].float().cpu().numpy()
        residues = values[1 : 1 + len(tok.full_seq)]
        if pool == "mean":
            results.append(residues.mean(axis=0))
        elif pool is None:
            results.append(residues)
        else:
            results.append(
                {
                    name: values[span.start + 1 : span.stop + 1].mean(axis=0)
                    for name, span in tok.region_slices.items()
                }
            )
    if pool == "mean":
        return np.stack(results) if results else np.empty((0, 0), dtype=np.float32)
    return results


def embed(
    loaded,
    data,
    *,
    pool="mean",
    layer=None,
    species="human",
):
    """Return residue embeddings for gene-component inputs."""
    return _embed(loaded, data, pool=pool, layer=layer, species=species)


@torch.inference_mode()
def masked_logits(loaded, tok: TCRpMHCTokenizer, positions: np.ndarray) -> torch.Tensor:
    """Mask sequence positions and return their model logits."""
    ids = tok.original_ids.clone()
    token_indices = torch.as_tensor(positions, dtype=torch.long) + 1
    ids[token_indices] = MASK_IDX
    output = loaded.module(
        ids.unsqueeze(0).to(loaded.device),
        repr_layers=[loaded.num_layers],
        return_contacts=False,
    )
    return output["logits"][0, token_indices.to(loaded.device)].float().cpu()


def profile_from_logits(logits: torch.Tensor) -> pd.DataFrame:
    """Convert logits into float64 probabilities over the standard amino acids."""
    probabilities = torch.softmax(logits[:, AA20_IDS].double(), dim=-1).numpy()
    profile = pd.DataFrame(probabilities, columns=list(AA20))
    profile.index = pd.RangeIndex(1, len(profile) + 1, name="position")
    profile["entropy"] = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
    return profile


def region_profile(
    loaded,
    data,
    region="peptide",
    length=None,
    *,
    species="human",
) -> pd.DataFrame:
    """Return the conditional amino-acid profile for one annotated region."""
    if region not in REGIONS:
        raise ValidationError(f"Unknown region {region!r}. Choose from {list(REGIONS)}")
    return _region_profile(
        loaded,
        data,
        region,
        length,
        species=species,
        prepared=False,
    )


def _region_profile(
    loaded,
    data,
    region="peptide",
    length=None,
    *,
    species="human",
    prepared=False,
) -> pd.DataFrame:
    entry = _masked_entry(
        loaded,
        data,
        region,
        length,
        species=species,
        prepared=prepared,
    )
    tok = TCRpMHCTokenizer(entry, mask_region=None)
    positions = region_positions(tok, region, entry)
    profile = profile_from_logits(masked_logits(loaded, tok, positions))
    return _attach_model_attrs(profile, loaded, "region_profile", entry)


def peptide_profile(
    loaded,
    data,
    length,
    *,
    species="human",
) -> pd.DataFrame:
    """Return a peptide profile conditioned on the supplied complex."""
    return region_profile(
        loaded,
        data,
        region="peptide",
        length=length,
        species=species,
    )


def consensus(profile: pd.DataFrame) -> str:
    """Return the highest-probability residue at each profile position."""
    missing = [amino_acid for amino_acid in AA20 if amino_acid not in profile]
    if missing:
        raise ValidationError(f"Profile is missing amino-acid columns: {missing}")
    return "".join(profile[list(AA20)].idxmax(axis=1))


def sequence_logo(
    profile: pd.DataFrame,
    units="bits",
    ax=None,
    save: str | Path | None = None,
    title: str | None = None,
):
    """Draw a probability or information-content sequence logo."""
    import logomaker
    import matplotlib.pyplot as plt

    if units not in ("bits", "probability"):
        raise ValidationError("units must be 'bits' or 'probability'")
    matrix = profile[list(AA20)].copy()
    matrix.index = range(len(matrix))
    if units == "bits":
        matrix = logomaker.transform_matrix(matrix, from_type="probability", to_type="information")
    if ax is None:
        _, ax = plt.subplots(figsize=(0.6 * len(matrix) + 1, 2.4))
    logomaker.Logo(matrix, ax=ax)
    ax.set_xticks(range(len(matrix)))
    ax.set_xticklabels(range(1, len(matrix) + 1))
    ax.set_xlabel("position")
    ax.set_ylabel("bits" if units == "bits" else "probability")
    if units == "bits":
        ax.set_ylim(0, np.log2(len(AA20)))
    if title:
        ax.set_title(title)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.tight_layout()
        ax.figure.savefig(save, dpi=200)
    return ax


__all__ = [
    "consensus",
    "embed",
    "peptide_profile",
    "region_profile",
    "sequence_logo",
]
