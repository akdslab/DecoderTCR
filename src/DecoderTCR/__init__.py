"""Discover and use exact public DecoderTCR model releases."""

from .core import IntegrityError, ReleaseError, UnavailableError, ValidationError

__version__ = "1.5.0"


def implementation_info():
    """Return one content identity for DecoderTCR and both vendored backbones."""
    import hashlib
    from pathlib import Path

    source_root = Path(__file__).resolve().parent.parent
    package_names = ("DecoderTCR", "esm", "esmc")
    selected = []
    for package_name in package_names:
        package_root = source_root / package_name
        if not package_root.is_dir():
            raise IntegrityError(f"Installed package is incomplete: {package_name}")
        selected.extend(
            path
            for path in package_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    digest = hashlib.sha256()
    for path in sorted(selected, key=lambda value: value.relative_to(source_root).as_posix()):
        relative = path.relative_to(source_root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "package_version": __version__,
        "implementation_sha256": digest.hexdigest(),
        "file_count": len(selected),
        "packages": list(package_names),
    }


def models(include_unavailable=False, *, registry=None):
    from .registry import Registry

    selected = registry if isinstance(registry, Registry) else Registry(registry)
    return selected.models(include_unavailable=include_unavailable)


def datasets(*, registry=None):
    """List raw public gene-input datasets without downloading their CSVs."""
    from .registry import Registry

    selected = registry if isinstance(registry, Registry) else Registry(registry)
    return selected.datasets()


def dataset_info(dataset_id, *, registry=None):
    """Return raw public dataset metadata without downloading its CSV."""
    from .registry import Registry

    selected = registry if isinstance(registry, Registry) else Registry(registry)
    return selected.dataset_info(dataset_id)


def dataset(dataset_id, *, registry=None):
    """Return one verified raw gene-input dataset as a DataFrame."""
    from ._datasets import load as load_dataset

    return load_dataset(dataset_id, registry=registry)


def info(model_id, *, registry=None):
    from .registry import Registry

    selected = registry if isinstance(registry, Registry) else Registry(registry)
    return selected.info(model_id)


def load(model_id, device="cuda", *, registry=None):
    from .runtime import load as load_model

    return load_model(model_id, device=device, registry=registry)


def _resolved_artifact(model_id, registry):
    from .core import ValidationError
    from .registry import Registry

    if not isinstance(model_id, str):
        raise ValidationError("model_id must be an exact model ID")
    selected = registry if isinstance(registry, Registry) else Registry(registry)
    return selected, selected.resolve(model_id)


def _artifact_metadata(model_id, resolved):
    descriptor = resolved.member["weights"]
    return {
        "model_id": model_id,
        "sha256": descriptor["sha256"],
        "size_bytes": descriptor["size_bytes"],
        "backbone": resolved.member["backbone"],
        "architecture": resolved.member["arch"],
        "sequence_convention": resolved.member["sequence_convention"],
        "manifest_sha256": resolved.manifest["sha256"],
        "catalog_sha256": resolved.catalog_sha256,
    }


def artifact_info(model_id, *, registry=None):
    """Return exact artifact metadata without transferring its weights."""
    _, resolved = _resolved_artifact(model_id, registry)
    return _artifact_metadata(model_id, resolved)


def artifact(model_id, *, registry=None):
    """Return verified artifact facts without constructing a model."""
    selected, resolved = _resolved_artifact(model_id, registry)
    descriptor = resolved.member["weights"]
    return {
        **_artifact_metadata(model_id, resolved),
        "path": selected.verify_artifact(resolved.bundle, descriptor),
    }


def score(
    data,
    *,
    model,
    device="cuda",
    registry=None,
    species="human",
    mask_region="peptide",
    cache_size=256,
):
    return _resolve_loaded(model, device, registry).score(
        data,
        species=species,
        mask_region=mask_region,
        cache_size=cache_size,
    )


def score_from_components(
    data,
    *,
    model,
    device="cuda",
    registry=None,
    species="human",
    mask_region="peptide",
    cache_size=256,
):
    """Reconstruct gene components and score them with an exact model release."""
    return score(
        data,
        model=model,
        device=device,
        registry=registry,
        species=species,
        mask_region=mask_region,
        cache_size=cache_size,
    )


def embed(
    data,
    *,
    model,
    device="cuda",
    registry=None,
    species="human",
    pool="mean",
    layer=None,
):
    """Return backbone embeddings from an exact model release."""
    from .features import embed as embed_complexes

    return embed_complexes(
        _resolve_loaded(model, device, registry),
        data,
        species=species,
        pool=pool,
        layer=layer,
    )


def reconstruct_components(data, *, model, registry=None, species="human"):
    """Reconstruct gene components using the selected model's preparation."""
    from .reconstruct.components import reconstruct_components as reconstruct

    return reconstruct(
        data,
        _model_sequence_convention(model, registry),
        species=species,
    )


def reference_info(*, model, registry=None, species="human"):
    """Return verified reconstruction reference identities for a model."""
    from .reconstruct.components import reference_identity
    from .runtime import LoadedModel

    convention = _model_sequence_convention(model, registry)
    model_id = model.model_id if isinstance(model, LoadedModel) else model
    return {
        "model_id": model_id,
        "sequence_convention": convention,
        **reference_identity(convention, species=species),
    }


def lookup_hla(allele, *, model, registry=None, species="human"):
    """Return the allele sequences for the selected model's convention."""
    from .reconstruct.hla import lookup_hla as lookup

    return lookup(
        allele,
        _model_sequence_convention(model, registry),
        species=species,
    )


def list_alleles(*, model, registry=None, species="human"):
    """List alleles available for the selected model's convention."""
    from .reconstruct.hla import list_alleles as available

    return available(_model_sequence_convention(model, registry), species=species)


def normalize_allele_key(allele, *, species="human"):
    from .reconstruct.hla import normalize_allele_key as normalize

    return normalize(allele, species=species)


def normalize_gene(gene, *, species="human"):
    from .reconstruct.tcr import normalize_gene as normalize

    return normalize(gene, species=species)


def stitch_tcrs(rows, *, species="human"):
    from .reconstruct.tcr import stitch_tcrs as stitch

    return stitch(rows, species=species)


def region_profile(
    data,
    region="peptide",
    length=None,
    *,
    model,
    device="cuda",
    registry=None,
    species="human",
):
    from .features import region_profile as profile

    return profile(
        _resolve_loaded(model, device, registry),
        data,
        region=region,
        length=length,
        species=species,
    )


def peptide_profile(
    data,
    length,
    *,
    model,
    device="cuda",
    registry=None,
    species="human",
):
    from .features import peptide_profile as profile

    return profile(
        _resolve_loaded(model, device, registry),
        data,
        length,
        species=species,
    )


def iegr_profile(
    data,
    region="peptide",
    length=None,
    *,
    model,
    device="cuda",
    registry=None,
    species="human",
    temperature=0.1,
    seed=42,
):
    from .design import iegr_profile as profile

    return profile(
        _resolve_loaded(model, device, registry),
        data,
        region=region,
        length=length,
        species=species,
        temperature=temperature,
        seed=seed,
    )


def sample_from_profile(profile, **kwargs):
    from .design import sample_from_profile as sample

    return sample(profile, **kwargs)


def consensus(profile):
    from .features import consensus as profile_consensus

    return profile_consensus(profile)


def sequence_logo(profile, **kwargs):
    from .features import sequence_logo as draw_logo

    return draw_logo(profile, **kwargs)


def block_gibbs(
    data,
    peptides,
    *,
    model,
    device="cuda",
    registry=None,
    species="human",
    region="peptide",
    k=5,
    rounds=1,
    n=None,
    temperature=1.0,
    seed=42,
    max_forwards=None,
):
    from .design import block_gibbs as refine

    return refine(
        _resolve_loaded(model, device, registry),
        data,
        peptides,
        species=species,
        region=region,
        k=k,
        rounds=rounds,
        n=n,
        temperature=temperature,
        seed=seed,
        max_forwards=max_forwards,
    )


def iegr(
    data,
    *,
    model,
    device="cuda",
    registry=None,
    species="human",
    region="peptide",
    length=None,
    n=10,
    temperature=0.1,
    gibbs_rounds=10,
    gibbs_subset_size=5,
    seed=42,
):
    from .design import iegr as design_region

    return design_region(
        _resolve_loaded(model, device, registry),
        data,
        species=species,
        region=region,
        length=length,
        n=n,
        temperature=temperature,
        gibbs_rounds=gibbs_rounds,
        gibbs_subset_size=gibbs_subset_size,
        seed=seed,
    )


def design_peptides(
    data,
    length,
    *,
    model,
    device="cuda",
    registry=None,
    species="human",
    n=10,
    method="one_shot",
    temperature=None,
    seed=42,
    rescore=True,
    gibbs_rounds=10,
    gibbs_subset_size=5,
    profile_method="one_shot",
    gibbs_k=0,
    gibbs_temperature=None,
    gibbs_max_forwards=None,
    cap=1_000_000,
):
    from .design import design_peptides as design

    return design(
        _resolve_loaded(model, device, registry),
        data,
        length,
        species=species,
        n=n,
        method=method,
        temperature=temperature,
        seed=seed,
        rescore=rescore,
        gibbs_rounds=gibbs_rounds,
        gibbs_subset_size=gibbs_subset_size,
        profile_method=profile_method,
        gibbs_k=gibbs_k,
        gibbs_temperature=gibbs_temperature,
        gibbs_max_forwards=gibbs_max_forwards,
        cap=cap,
    )


def _resolve_loaded(model, device, registry):
    from .core import ValidationError
    from .runtime import LoadedModel

    if isinstance(model, LoadedModel):
        return model
    if not isinstance(model, str):
        raise ValidationError("model must be an exact model ID or LoadedModel")
    return load(model, device=device, registry=registry)


def _model_sequence_convention(model, registry):
    from .core import ValidationError
    from .registry import Registry
    from .runtime import LoadedModel

    if isinstance(model, LoadedModel):
        return model.sequence_convention
    if not isinstance(model, str):
        raise ValidationError("model must be an exact model ID or LoadedModel")
    selected = registry if isinstance(registry, Registry) else Registry(registry)
    return selected.resolve(model).member["sequence_convention"]


def reproduce(result_id, mode="metrics", output=None, *, registry=None, parent_output=None):
    from .reproduction import reproduce as reproduce_result

    return reproduce_result(
        result_id,
        mode=mode,
        output=output,
        registry=registry,
        parent_output=parent_output,
    )


def reproduce_shard(result_id, shard_index, shard_count, output, *, registry=None):
    from .reproduction import reproduce_shard as score_shard

    return score_shard(result_id, shard_index, shard_count, output, registry=registry)


def merge_reproduction_shards(result_id, shard_outputs, output, *, registry=None):
    from .reproduction import merge_reproduction_shards as merge_shards

    return merge_shards(result_id, shard_outputs, output, registry=registry)


__all__ = [
    "__version__",
    "IntegrityError",
    "ReleaseError",
    "UnavailableError",
    "ValidationError",
    "artifact",
    "artifact_info",
    "block_gibbs",
    "consensus",
    "dataset",
    "dataset_info",
    "datasets",
    "design_peptides",
    "embed",
    "info",
    "iegr",
    "iegr_profile",
    "implementation_info",
    "list_alleles",
    "load",
    "lookup_hla",
    "merge_reproduction_shards",
    "models",
    "normalize_allele_key",
    "normalize_gene",
    "peptide_profile",
    "reconstruct_components",
    "reference_info",
    "region_profile",
    "reproduce",
    "reproduce_shard",
    "sample_from_profile",
    "score",
    "score_from_components",
    "sequence_logo",
    "stitch_tcrs",
]
