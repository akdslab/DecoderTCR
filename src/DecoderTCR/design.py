"""Sequence design with a resolved DecoderTCR model handle."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch

from .constants import AA20, AA20_IDS, MASK_IDX
from .core import ValidationError
from .features import (
    REGIONS,
    _attach_model_attrs,
    _masked_entry,
    _region_profile,
    consensus,
    region_positions,
)
from .tokenizer import TCRpMHCTokenizer

METHODS = ("one_shot", "iegr")
DEFAULT_TEMPERATURE = {"one_shot": 1.0, "iegr": 0.1}
DRAW_CAP = 1_000_000
BATCH = 250_000
CLIP_P = 1e-12
MAX_CODEABLE_LENGTH = 14
FORWARD_BUDGET_PER_DESIGN = 20

_AA_IDS = torch.tensor(AA20_IDS, dtype=torch.long)
_ID_TO_AA = dict(zip(AA20_IDS, AA20))


def _positive_integer(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValidationError(f"{name} must be at least 1")
    return value


def _finite_number(value, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
        raise ValidationError(f"{name} must be a finite number")
    return float(value)


def _encode(values: np.ndarray) -> np.ndarray:
    """Pack residue indices into uint64 base-20 codes."""
    code = np.zeros(len(values), dtype=np.uint64)
    for column in range(values.shape[1]):
        code = code * np.uint64(20) + values[:, column].astype(np.uint64)
    return code


def sample_from_profile(
    profile: pd.DataFrame,
    n: int = 1000,
    temperature: float = 1.0,
    seed: int = 42,
    cap: int = DRAW_CAP,
) -> tuple[list[str], dict]:
    """Draw distinct sequences from independent profile marginals."""
    n = _positive_integer(n, "n")
    cap = _positive_integer(cap, "cap")
    temperature = _finite_number(temperature, "temperature")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError("seed must be an integer")
    missing = [amino_acid for amino_acid in AA20 if amino_acid not in profile]
    if missing:
        raise ValidationError(f"Profile is missing amino-acid columns: {missing}")
    probabilities = profile[list(AA20)].to_numpy(dtype=np.float64, copy=True)
    if probabilities.ndim != 2 or not len(probabilities):
        raise ValidationError("Profile must contain at least one position")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValidationError("Profile probabilities must be finite and nonnegative")
    if (probabilities.sum(axis=1) <= 0).any():
        raise ValidationError("Every profile position must have positive probability mass")
    length = probabilities.shape[0]
    if length > MAX_CODEABLE_LENGTH:
        raise ValidationError(f"Sequences longer than {MAX_CODEABLE_LENGTH} are not supported")

    if temperature <= 0:
        sequence = consensus(profile)
        return [sequence], {
            "n_requested": n,
            "n_returned": 1,
            "n_draws_used": 0,
            "saturated": n > 1,
            "cap_reached": False,
            "support_exhausted": n > 1,
        }

    if temperature != 1.0:
        log_probabilities = np.log(np.clip(probabilities, CLIP_P, None)) / temperature
        probabilities = np.exp(log_probabilities - log_probabilities.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    letters = np.asarray(list(AA20))
    seen = np.zeros(0, dtype=np.uint64)
    kept: list[np.uint64] = []
    used = 0
    last_novel = None
    while len(kept) < n and used < cap:
        wanted = n - len(kept)
        size = int(min(BATCH, cap - used, max(1024, 4 * wanted)))
        draw = np.stack(
            [rng.choice(20, size=size, p=probabilities[position]) for position in range(length)],
            axis=1,
        ).astype(np.uint8)
        used += size
        codes = _encode(draw)
        unique, first = np.unique(codes, return_index=True)
        novel = ~np.isin(unique, seen)
        last_novel = int(novel.sum())
        if novel.any():
            fresh = unique[novel]
            take = fresh[np.argsort(first[novel])][:wanted]
            kept.extend(take.tolist())
            seen = np.union1d(seen, fresh)

    sequences = []
    for code in kept[:n]:
        indices = np.empty(length, dtype=np.int64)
        value = int(code)
        for position in range(length - 1, -1, -1):
            indices[position] = value % 20
            value //= 20
        sequences.append("".join(letters[indices]))
    short = len(sequences) < n
    return sequences, {
        "n_requested": n,
        "n_returned": len(sequences),
        "n_draws_used": used,
        "saturated": short,
        "cap_reached": used >= cap,
        "support_exhausted": bool(short and last_novel == 0),
    }


def _warn_esm2(loaded) -> None:
    if loaded.backbone == "esm2":
        warnings.warn(
            "Iterative masking with ESM2 changes token-dropout mask-ratio rescaling.",
            UserWarning,
            stacklevel=3,
        )


def _sample(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> int:
    amino_acid_logits = logits[_AA_IDS].float()
    if temperature <= 0:
        return int(_AA_IDS[int(amino_acid_logits.argmax())])
    probabilities = torch.softmax(amino_acid_logits / temperature, dim=-1)
    return int(_AA_IDS[int(torch.multinomial(probabilities, 1, generator=generator))])


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits[:, _AA_IDS].float(), dim=-1)
    return -(probabilities * torch.log(probabilities + 1e-12)).sum(dim=-1)


@torch.inference_mode()
def _forward(loaded, ids: torch.Tensor) -> torch.Tensor:
    output = loaded.module(
        ids.unsqueeze(0).to(loaded.device),
        repr_layers=[loaded.num_layers],
        return_contacts=False,
    )
    return output["logits"][0].float().cpu()


def _rebuild(base: torch.Tensor, chosen: dict[int, int], masked: list[int]) -> torch.Tensor:
    ids = base.clone()
    for token_index, value in chosen.items():
        ids[token_index] = value
    for token_index in masked:
        ids[token_index] = MASK_IDX
    return ids


def _phase1(loaded, base_ids, token_indices, temperature, generator):
    logits = _forward(loaded, _rebuild(base_ids, {}, token_indices))
    chosen = {
        token_index: _sample(logits[token_index], temperature, generator)
        for token_index in token_indices
    }
    remaining = list(token_indices)
    while len(remaining) > 1:
        rows = logits[torch.tensor(remaining, dtype=torch.long)]
        remaining.pop(int(_entropy(rows).argmin()))
        logits = _forward(loaded, _rebuild(base_ids, chosen, remaining))
        for token_index in remaining:
            chosen[token_index] = _sample(logits[token_index], temperature, generator)
    ids = _rebuild(base_ids, chosen, [])
    sequence = "".join(_ID_TO_AA[int(ids[index])] for index in token_indices)
    return ids, sequence


def _phase2(
    loaded,
    ids,
    token_indices,
    temperature,
    rounds,
    subset_size,
    generator,
    rng,
):
    length = len(token_indices)
    subset_size = min(subset_size, length)
    interval = max(length - 1, 1)
    index_array = np.asarray(token_indices)
    current = ids.clone()
    output = []
    for step in range(rounds * length):
        subset = [int(index) for index in rng.choice(index_array, size=subset_size, replace=False)]
        logits = _forward(loaded, _rebuild(current, {}, subset))
        for token_index in subset:
            current[token_index] = _sample(logits[token_index], temperature, generator)
        if (step + 1) % interval == 0:
            sequence = "".join(_ID_TO_AA[int(current[index])] for index in token_indices)
            output.append((current.clone(), sequence))
    return output


def _profile_row(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits[_AA_IDS].double(), dim=-1).numpy()


def _iegr_profile(
    loaded,
    data,
    region="peptide",
    length=None,
    temperature=0.1,
    seed=42,
    *,
    species="human",
    prepared=False,
) -> pd.DataFrame:
    """Profile a region while committing positions in ascending entropy order."""
    temperature = _finite_number(temperature, "temperature")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError("seed must be an integer")
    _warn_esm2(loaded)
    entry = _masked_entry(
        loaded,
        data,
        region,
        length,
        species=species,
        prepared=prepared,
    )
    tok = TCRpMHCTokenizer(entry, mask_region=None)
    token_indices = [int(position) + 1 for position in region_positions(tok, region, entry)]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = {}
    order = []
    chosen = {}
    remaining = list(token_indices)
    while remaining:
        logits = _forward(loaded, _rebuild(tok.original_ids, chosen, remaining))
        selected = torch.tensor(remaining, dtype=torch.long)
        token_index = remaining[int(_entropy(logits[selected]).argmin())]
        rows[token_index] = _profile_row(logits[token_index])
        order.append(token_index)
        chosen[token_index] = _sample(logits[token_index], temperature, generator)
        remaining.remove(token_index)
    probabilities = np.stack([rows[token_index] for token_index in token_indices])
    profile = pd.DataFrame(probabilities, columns=list(AA20))
    profile.index = pd.RangeIndex(1, len(profile) + 1, name="position")
    profile["entropy"] = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
    profile["commit_order"] = [order.index(token_index) + 1 for token_index in token_indices]
    return _attach_model_attrs(profile, loaded, "iegr_profile", entry)


def _block_gibbs(
    loaded,
    data,
    peptides,
    region="peptide",
    k=5,
    rounds=1,
    n=None,
    temperature=1.0,
    seed=42,
    max_forwards=None,
    *,
    species="human",
    prepared=False,
) -> tuple[list[str], dict]:
    """Refine a sequence library by repeatedly resampling random blocks."""
    peptides = list(peptides)
    if not peptides:
        return [], {
            "n_input": 0,
            "n_returned": 0,
            "n_forwards": 0,
            "n_changed": 0,
            "budget_exhausted": False,
        }
    k = _positive_integer(k, "k")
    rounds = _positive_integer(rounds, "rounds")
    if n is not None:
        n = _positive_integer(n, "n")
    if max_forwards is not None:
        max_forwards = _positive_integer(max_forwards, "max_forwards")
    temperature = _finite_number(temperature, "temperature")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError("seed must be an integer")
    if not all(isinstance(peptide, str) for peptide in peptides):
        raise ValidationError("Every sequence must be a string")
    length = len(peptides[0])
    if length < 1 or any(len(peptide) != length for peptide in peptides):
        raise ValidationError("Every sequence must have the same positive length")
    invalid = sorted(set("".join(peptides)) - set(AA20))
    if invalid:
        raise ValidationError(f"Sequences contain unsupported residues: {invalid}")
    _warn_esm2(loaded)

    entry = _masked_entry(
        loaded,
        data,
        region,
        length if region == "peptide" else None,
        species=species,
        prepared=prepared,
    )
    tok = TCRpMHCTokenizer(entry, mask_region=None)
    token_indices = [int(position) + 1 for position in region_positions(tok, region, entry)]
    if len(token_indices) != length:
        raise ValidationError(
            f"Sequences are length {length} but the region holds {len(token_indices)}"
        )
    amino_acid_to_id = dict(zip(AA20, AA20_IDS))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    block = min(k, length)
    index_array = np.asarray(token_indices)
    budget = (
        max_forwards
        if max_forwards is not None
        else rounds * len(peptides)
        if n is None
        else FORWARD_BUDGET_PER_DESIGN * n
    )
    chains = []
    for peptide in peptides:
        ids = tok.original_ids.clone()
        for token_index, amino_acid in zip(token_indices, peptide):
            ids[token_index] = amino_acid_to_id[amino_acid]
        chains.append(ids)

    def step(ids):
        subset = [int(index) for index in rng.choice(index_array, size=block, replace=False)]
        logits = _forward(loaded, _rebuild(ids, {}, subset))
        for token_index in subset:
            ids[token_index] = _sample(logits[token_index], temperature, generator)
        return "".join(_ID_TO_AA[int(ids[index])] for index in token_indices)

    seen = set()
    output = []
    forwards = 0
    changed = 0
    if n is None:
        for ids, peptide in zip(chains, peptides):
            refined = peptide
            for _ in range(rounds):
                if forwards >= budget:
                    break
                refined = step(ids)
                forwards += 1
            changed += refined != peptide
            if refined not in seen:
                seen.add(refined)
                output.append(refined)
    else:
        chain_index = 0
        while len(output) < n and forwards < budget:
            origin = peptides[chain_index % len(chains)]
            refined = step(chains[chain_index % len(chains)])
            forwards += 1
            changed += refined != origin
            if refined not in seen:
                seen.add(refined)
                output.append(refined)
            chain_index += 1
    return output, {
        "n_input": len(peptides),
        "n_returned": len(output),
        "n_forwards": forwards,
        "n_changed": changed,
        "budget_exhausted": bool(n is not None and len(output) < n),
    }


def _iegr(
    loaded,
    data,
    region="peptide",
    length=None,
    n=10,
    temperature=0.1,
    gibbs_rounds=10,
    gibbs_subset_size=5,
    seed=42,
    *,
    species="human",
    prepared=False,
) -> pd.DataFrame:
    """Design one region using iterative entropy-guided refinement."""
    n = _positive_integer(n, "n")
    temperature = _finite_number(temperature, "temperature")
    if not isinstance(gibbs_rounds, int) or isinstance(gibbs_rounds, bool) or gibbs_rounds < 0:
        raise ValidationError("gibbs_rounds must be a nonnegative integer")
    gibbs_subset_size = _positive_integer(gibbs_subset_size, "gibbs_subset_size")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError("seed must be an integer")
    _warn_esm2(loaded)
    entry = _masked_entry(
        loaded,
        data,
        region,
        length,
        species=species,
        prepared=prepared,
    )
    tok = TCRpMHCTokenizer(entry, mask_region=None)
    token_indices = [int(position) + 1 for position in region_positions(tok, region, entry)]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    ids, sequence = _phase1(
        loaded,
        tok.original_ids,
        token_indices,
        temperature,
        generator,
    )
    rows = [{"sequence": sequence, "phase": 1, "step": 0}]
    if gibbs_rounds > 0:
        checkpoints = _phase2(
            loaded,
            ids,
            token_indices,
            temperature,
            gibbs_rounds,
            gibbs_subset_size,
            generator,
            rng,
        )
        rows.extend(
            {"sequence": candidate, "phase": 2, "step": step}
            for step, (_, candidate) in enumerate(checkpoints, start=1)
        )
    output = pd.DataFrame(rows).drop_duplicates(subset="sequence", ignore_index=True).head(n)
    return _attach_model_attrs(output, loaded, "iegr", entry)


def _rescore(loaded, base, sequences):
    entries = []
    for sequence in sequences:
        components = dict(base["sequences"])
        components["Peptide"] = sequence
        entries.append(
            {
                "sequences": components,
                "pocket_idx": base.get("pocket_idx", {}),
                "annotation_contract": base.get("annotation_contract"),
                "reconstruction": base.get("reconstruction"),
            }
        )
    return loaded._score_prepared(entries, mask_region="peptide")["pll"].to_numpy()


def iegr_profile(
    loaded,
    data,
    region="peptide",
    length=None,
    temperature=0.1,
    seed=42,
    *,
    species="human",
) -> pd.DataFrame:
    """Profile a gene-defined region in ascending entropy order."""
    if region not in REGIONS:
        raise ValidationError(f"Unknown region {region!r}. Choose from {list(REGIONS)}")
    return _iegr_profile(
        loaded,
        data,
        region=region,
        length=length,
        temperature=temperature,
        seed=seed,
        species=species,
    )


def block_gibbs(
    loaded,
    data,
    peptides,
    region="peptide",
    k=5,
    rounds=1,
    n=None,
    temperature=1.0,
    seed=42,
    max_forwards=None,
    *,
    species="human",
) -> tuple[list[str], dict]:
    """Refine a sequence library against one gene-defined context."""
    if region not in REGIONS:
        raise ValidationError(f"Unknown region {region!r}. Choose from {list(REGIONS)}")
    return _block_gibbs(
        loaded,
        data,
        peptides,
        region=region,
        k=k,
        rounds=rounds,
        n=n,
        temperature=temperature,
        seed=seed,
        max_forwards=max_forwards,
        species=species,
    )


def iegr(
    loaded,
    data,
    region="peptide",
    length=None,
    n=10,
    temperature=0.1,
    gibbs_rounds=10,
    gibbs_subset_size=5,
    seed=42,
    *,
    species="human",
) -> pd.DataFrame:
    """Design one gene-defined region using entropy-guided refinement."""
    if region not in REGIONS:
        raise ValidationError(f"Unknown region {region!r}. Choose from {list(REGIONS)}")
    return _iegr(
        loaded,
        data,
        region=region,
        length=length,
        n=n,
        temperature=temperature,
        gibbs_rounds=gibbs_rounds,
        gibbs_subset_size=gibbs_subset_size,
        seed=seed,
        species=species,
    )


def design_peptides(
    loaded,
    data,
    length,
    n=10,
    method="one_shot",
    temperature=None,
    seed=42,
    rescore=True,
    gibbs_rounds=10,
    gibbs_subset_size=5,
    *,
    profile_method="one_shot",
    gibbs_k=0,
    gibbs_temperature=None,
    gibbs_max_forwards=None,
    cap=DRAW_CAP,
    species="human",
) -> pd.DataFrame:
    """Generate candidate peptides and optionally score them with the same handle."""
    length = _positive_integer(length, "length")
    n = _positive_integer(n, "n")
    if method not in METHODS:
        raise ValidationError(f"method must be one of {METHODS}")
    if profile_method not in METHODS:
        raise ValidationError(f"profile_method must be one of {METHODS}")
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE[method]
    temperature = _finite_number(temperature, "temperature")
    if gibbs_temperature is None:
        gibbs_temperature = temperature
    gibbs_temperature = _finite_number(gibbs_temperature, "gibbs_temperature")
    if not isinstance(gibbs_k, int) or isinstance(gibbs_k, bool) or gibbs_k < 0:
        raise ValidationError("gibbs_k must be a nonnegative integer")

    working_data = _masked_entry(
        loaded,
        data,
        "peptide",
        length,
        species=species,
    )

    stats = {}
    reconstruction = None
    if method == "iegr":
        output = _iegr(
            loaded,
            working_data,
            region="peptide",
            length=length,
            n=n,
            temperature=temperature,
            gibbs_rounds=gibbs_rounds,
            gibbs_subset_size=gibbs_subset_size,
            seed=seed,
            species=species,
            prepared=True,
        )
        reconstruction = output.attrs.get("reconstruction")
        stats = {
            "n_requested": n,
            "n_returned": len(output),
            "n_draws_used": 0,
            "saturated": len(output) < n,
            "cap_reached": False,
            "support_exhausted": False,
        }
    else:
        if profile_method == "iegr":
            profile = _iegr_profile(
                loaded,
                working_data,
                region="peptide",
                length=length,
                temperature=temperature,
                seed=seed,
                species=species,
                prepared=True,
            )
        else:
            profile = _region_profile(
                loaded,
                working_data,
                region="peptide",
                length=length,
                species=species,
                prepared=True,
            )
        reconstruction = profile.attrs.get("reconstruction")
        sequences, stats = sample_from_profile(
            profile, n=n, temperature=temperature, seed=seed, cap=cap
        )
        stats["profile_method"] = profile_method
        if gibbs_k > 0:
            sequences, gibbs_stats = _block_gibbs(
                loaded,
                working_data,
                sequences,
                region="peptide",
                k=gibbs_k,
                n=n,
                max_forwards=gibbs_max_forwards,
                temperature=gibbs_temperature,
                seed=seed,
                species=species,
                prepared=True,
            )
            stats.update(
                {f"gibbs_{key}": value for key, value in gibbs_stats.items()},
                gibbs_k=gibbs_k,
                gibbs_temperature=gibbs_temperature,
            )
            stats["n_returned"] = len(sequences)
            stats["saturated"] = len(sequences) < n
        output = pd.DataFrame({"sequence": sequences})
        output["phase"] = 2 if gibbs_k > 0 else 0
        output["step"] = range(len(output))
    output.insert(1, "method", method)
    if rescore and len(output):
        output["pll"] = _rescore(
            loaded,
            working_data,
            output["sequence"].tolist(),
        )
        output = output.sort_values("pll", ascending=False, ignore_index=True)
    output.attrs.update(stats)
    identity = _attach_model_attrs(pd.DataFrame(), loaded, "design_peptides").attrs["reproduction"]
    output.attrs["reproduction"] = identity
    if reconstruction is not None:
        output.attrs["reconstruction"] = dict(reconstruction)
    return output


__all__ = [
    "block_gibbs",
    "design_peptides",
    "iegr",
    "iegr_profile",
    "sample_from_profile",
]
