"""Score simultaneously masked peptide residues with full-vocabulary PLL."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import SEQUENCE_CONVENTIONS
from .core import ValidationError
from .tokenizer import TCRpMHCTokenizer

_ALIASES = {
    "HLA_a": ("hla_a", "hla", "hla_chain", "hla_seq"),
    "HLA_b": ("hla_b", "b2m", "beta2m"),
    "Peptide": ("peptide",),
    "TCR_a": ("tcr_a", "tcra"),
    "TCR_b": ("tcr_b", "tcrb"),
}


def _sequence(value, key):
    if value is None or (not isinstance(value, (dict, list, tuple)) and pd.isna(value)):
        return ""
    if not isinstance(value, str):
        raise ValidationError(f"{key} must contain sequence strings")
    return value


def _sequence_columns(names):
    columns = {str(column).lower(): column for column in names}
    if len(columns) != len(names):
        raise ValidationError("Input columns must be unique ignoring capitalization")
    selected = {}
    missing = []
    for key, aliases in _ALIASES.items():
        matches = [columns[alias] for alias in aliases if alias in columns]
        if len(matches) > 1:
            raise ValidationError(f"Provide only one sequence column for {key}")
        if matches:
            selected[key] = matches[0]
        else:
            missing.append(key)
    if missing:
        raise ValidationError(
            f"Missing required sequence columns: {', '.join(missing)}. "
            "Provide all five sequence columns, using an explicit empty value for absent chains"
        )
    return selected


def prepare_inputs(data, sequence_convention: str) -> tuple[pd.DataFrame, list[dict]]:
    """Preserve input fields while resolving supported sequence column names."""
    if sequence_convention not in SEQUENCE_CONVENTIONS:
        raise ValidationError("Unsupported sequence convention. Upgrade DecoderTCR")
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, keep_default_na=False)
        elif path.suffix.lower() in (".parquet", ".pq"):
            frame = pd.read_parquet(path)
        else:
            raise ValidationError("Scoring inputs must be CSV or Parquet")
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        records = [data] if isinstance(data, Mapping) else list(data)
        if records and not all(isinstance(row, Mapping) for row in records):
            raise ValidationError("Scoring inputs must be records or a dataframe")
        structured = ["sequences" in row for row in records]
        if any(structured) and not all(structured):
            raise ValidationError("Do not mix flat and structured sequence records")
        if any(structured):
            flattened = []
            for row in records:
                sequences = row["sequences"]
                metadata = row.get("meta_data", {})
                if not isinstance(sequences, Mapping) or not isinstance(metadata, Mapping):
                    raise ValidationError("sequences and meta_data must be mappings")
                _sequence_columns(sequences)
                declarations = {}
                for key in ("sequence_convention", "annotation_contract"):
                    values = [
                        value for value in (row.get(key), metadata.get(key)) if value is not None
                    ]
                    if len(values) > 1 and values[0] != values[1]:
                        raise ValidationError(f"Conflicting {key} declarations")
                    if values:
                        declarations[key] = values[0]
                flattened.append(
                    {
                        **metadata,
                        **sequences,
                        **declarations,
                        "pocket_idx": row.get("pocket_idx", {}),
                    }
                )
            records = flattened
        for row in records:
            _sequence_columns(row)
        frame = pd.DataFrame(records)
    if not frame.columns.is_unique:
        raise ValidationError("Duplicate input columns are not supported")
    selected = _sequence_columns(frame.columns)
    if frame.empty:
        return frame, []
    if "sequence_convention" in frame:
        conventions = frame["sequence_convention"].dropna().unique().tolist()
        if conventions != [sequence_convention]:
            raise ValidationError("Input sequence convention does not match the selected model")
    entries = []
    for row in frame.to_dict("records"):
        sequences = {key: _sequence(row[column], key) for key, column in selected.items()}
        if not sequences["Peptide"]:
            raise ValidationError("Peptide sequences cannot be empty")
        annotation = row.get("pocket_idx")
        if isinstance(annotation, str):
            try:
                annotation = json.loads(annotation) if annotation else {}
            except ValueError as exc:
                raise ValidationError("pocket_idx must be a JSON object") from exc
        if annotation is None or (isinstance(annotation, float) and np.isnan(annotation)):
            annotation = {}
        if not isinstance(annotation, dict):
            raise ValidationError("pocket_idx must be a mapping")
        contract = row.get("annotation_contract")
        if contract is not None and not (isinstance(contract, float) and np.isnan(contract)):
            if contract not in ("half-open", "inclusive-end"):
                raise ValidationError("annotation_contract must be 'half-open' or 'inclusive-end'")
        else:
            contract = None
        entry = {"sequences": sequences, "pocket_idx": annotation}
        if contract is not None:
            entry["annotation_contract"] = contract
        entries.append(entry)
    return frame, entries


def compute_pll(true_token_ids: torch.Tensor, logits: torch.Tensor) -> float:
    """Normalize across every output channel before selecting true residues."""
    if true_token_ids.ndim != 1 or logits.ndim != 2 or len(true_token_ids) != len(logits):
        raise ValidationError("PLL expects one logit row for each target token")
    if not len(true_token_ids):
        return float("nan")
    if not torch.isfinite(logits).all():
        raise ValidationError("Model produced nonfinite logits")
    log_probs = torch.log_softmax(logits, dim=-1)
    true_token_ids = true_token_ids.to(logits.device)
    selected = log_probs[torch.arange(len(true_token_ids), device=logits.device), true_token_ids]
    return selected.sum().item() / len(true_token_ids)


@torch.inference_mode()
def run_pll_benchmark(
    model,
    num_layers: int,
    entries: list[dict],
    mask_region: str = "peptide",
    device="cpu",
    cache_size: int = 256,
):
    """Reuse bounded target logits for identical fully masked contexts."""
    if not isinstance(cache_size, int) or cache_size < 0:
        raise ValidationError("cache_size must be a nonnegative integer")
    model.eval()
    cache = OrderedDict()
    scores = []
    for entry in entries:
        tok = TCRpMHCTokenizer(entry, mask_region=mask_region)
        if not len(tok.target_indices):
            scores.append(float("nan"))
            continue
        key = tuple(tok.input_ids.tolist()) if mask_region == "peptide" else None
        if key is not None and key in cache:
            logits = cache.pop(key)
            cache[key] = logits
        else:
            output = model(tok.input_ids.unsqueeze(0).to(device))
            logits = output["logits"][0, tok.token_indices.to(device)].detach().cpu()
            if key is not None and cache_size:
                cache[key] = logits
                if len(cache) > cache_size:
                    cache.popitem(last=False)
        scores.append(compute_pll(tok.original_ids[tok.token_indices], logits))
    return np.asarray(scores, dtype=np.float64)
