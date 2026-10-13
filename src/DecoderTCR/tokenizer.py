"""Tokenize prepared sequences and apply deterministic region masking."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from .constants import ALPHABET, BATCH_CONVERTER, MASK_IDX, SEQUENCE_ORDER
from .core import ValidationError

_RESIDUES = frozenset(token for token in ALPHABET.all_toks if len(token) == 1)


def build_sequence(sequences: Mapping[str, str]) -> tuple[str, dict[str, slice]]:
    """Concatenate supplied components without separators or sequence conversion."""
    parts = []
    slices = {}
    offset = 0
    for key in SEQUENCE_ORDER:
        sequence = sequences.get(key, "")
        if not isinstance(sequence, str):
            raise ValidationError(f"{key} must contain an amino-acid sequence string")
        invalid = set(sequence) - _RESIDUES
        if invalid:
            raise ValidationError(
                f"{key} contains unsupported residue characters: {sorted(invalid)}"
            )
        if sequence:
            slices[key] = slice(offset, offset + len(sequence))
            parts.append(sequence)
            offset += len(sequence)
    return "".join(parts), slices


class TCRpMHCTokenizer:
    """Keep region indices in sequence space and token indices offset by CLS."""

    def __init__(self, entry: dict, mask_region: str | None = "peptide"):
        sequences = entry.get("sequences")
        if not isinstance(sequences, Mapping):
            raise ValidationError("Each structured input needs a sequences mapping")
        contract = entry.get("annotation_contract")
        if contract is not None and contract not in ("half-open", "inclusive-end"):
            raise ValidationError("annotation_contract must be 'half-open' or 'inclusive-end'")
        if contract == "inclusive-end" and mask_region in ("cdr", "cdr3", "bg"):
            raise ValidationError(
                f"mask_region={mask_region!r} requires half-open CDR annotations. "
                "Gene reconstruction provides inclusive-end CDR3 annotations"
            )
        self.full_seq, self.region_slices = build_sequence(sequences)
        if not self.full_seq:
            raise ValidationError("Cannot tokenize an empty sequence")
        peptide = self.region_slices.get("Peptide", slice(0, 0))
        self.peptide_indices = np.arange(peptide.start, peptide.stop, dtype=np.intp)
        annotation = entry.get("pocket_idx") or {}
        self.pocket_indices = self._annotated_indices(annotation, False)
        self.cdr_indices = self._annotated_indices(annotation, True)
        self.cdr3_indices = self._annotated_indices(annotation, True, cdr3=True)
        _, _, original = BATCH_CONVERTER([("input", self.full_seq)])
        self.original_ids = original[0]
        self.input_ids = self.original_ids.clone()
        self.target_indices = self.region_indices(mask_region)
        self.token_indices = torch.as_tensor(self.target_indices + 1, dtype=torch.long)
        self.input_ids[self.token_indices] = MASK_IDX
        self.labels = torch.full_like(self.original_ids, -100)
        self.labels[self.token_indices] = self.original_ids[self.token_indices]

    def _annotated_indices(self, annotation, cdr, cdr3=False) -> np.ndarray:
        positions = []
        keys = {"TCR_a": "CDRa", "TCR_b": "CDRb"} if cdr else {"HLA_a": "HLA_a", "HLA_b": "HLA_b"}
        for sequence_key, annotation_key in keys.items():
            if sequence_key not in self.region_slices:
                continue
            span = self.region_slices[sequence_key]
            ranges = annotation.get(annotation_key, [])
            if not ranges:
                continue
            if not cdr:
                ranges = [ranges]
            elif cdr3:
                ranges = ranges[2:3]
            for limits in ranges:
                if (
                    len(limits) != 2
                    or not all(isinstance(x, int) for x in limits)
                    or not 0 <= limits[0] <= limits[1] <= span.stop - span.start
                ):
                    raise ValidationError(f"Invalid region annotation for {sequence_key}")
                positions.extend(range(span.start + limits[0], span.start + limits[1]))
        return np.asarray(sorted(set(positions)), dtype=np.intp)

    def region_indices(self, region: str | None) -> np.ndarray:
        if region is None:
            return np.asarray([], dtype=np.intp)
        indices = {
            "peptide": self.peptide_indices,
            "pocket": self.pocket_indices,
            "cdr": self.cdr_indices,
            "cdr3": self.cdr3_indices,
        }
        if region == "bg":
            special = set(self.peptide_indices) | set(self.pocket_indices) | set(self.cdr_indices)
            return np.asarray(
                [i for i in range(len(self.full_seq)) if i not in special], dtype=np.intp
            )
        if region not in indices:
            raise ValidationError("mask_region must be peptide, pocket, cdr, cdr3, or bg")
        return indices[region]
