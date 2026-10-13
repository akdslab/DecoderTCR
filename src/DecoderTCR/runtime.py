"""Load exact released artifacts and expose their scoring configuration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
import torch
from safetensors.torch import load_file

from .constants import ALPHABET, ESM2_ARCH, ESMC_ARCH, PAD_IDX, SCORING_RECIPE
from .core import IntegrityError, ValidationError
from .scoring import prepare_inputs, run_pll_benchmark


def model_spec(backbone: str, arch: str) -> dict:
    """Return constructor arguments for a supported architecture."""
    architectures = {"esmc": ESMC_ARCH, "esm2": ESM2_ARCH}
    if backbone not in architectures or arch not in architectures[backbone]:
        raise ValidationError(f"Unsupported architecture {backbone}/{arch}. Upgrade DecoderTCR")
    return dict(architectures[backbone][arch])


def build_model(backbone: str, arch: str, device="cpu", architecture_config=None):
    """Construct an inference module from declared or supplied dimensions."""
    spec = model_spec(backbone, arch) if architecture_config is None else dict(architecture_config)
    if backbone == "esm2":
        from esm.model.esm2 import ESM2

        with torch.device(device):
            model = ESM2(alphabet=ALPHABET, token_dropout=True, **spec)
    elif backbone == "esmc":
        from esmc.models.esmc import ESMC

        class InferenceESMC(ESMC):
            def forward(self, tokens, repr_layers=None, return_contacts=False):
                output = super().forward(sequence_tokens=tokens)
                return {
                    "logits": output.sequence_logits,
                    "representations": {-1: output.embeddings},
                }

        with torch.device(device):
            model = InferenceESMC(tokenizer=SimpleNamespace(pad_token_id=PAD_IDX), **spec)
    else:
        raise ValidationError(f"Unsupported backbone {backbone}. Upgrade DecoderTCR")
    return model.to(device).eval()


def normalize_state_dict(state: Mapping, backbone: str) -> dict[str, torch.Tensor]:
    """Resolve a supported checkpoint container and remove uniform wrapper prefixes."""
    if backbone not in ("esmc", "esm2"):
        raise ValidationError("Checkpoint backbone must be esmc or esm2")
    if not isinstance(state, Mapping):
        raise ValidationError("Checkpoint must contain a state dictionary")
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], Mapping):
        state = state["model"]
    if not isinstance(state, Mapping) or not state:
        raise ValidationError("Checkpoint state dictionary is empty")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValidationError("Every model state entry must be a named tensor")
    state = dict(state)
    for prefix in ("module.", "_forward_module."):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}
    for prefix in ("model.model.", "model."):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}
            break
    if backbone == "esm2":
        pattern = re.compile(r"^(encoder\.sentence_encoder\.|encoder\.)")
        normalized = {pattern.sub("", key): value for key, value in state.items()}
        if len(normalized) != len(state):
            raise ValidationError("Checkpoint contains colliding normalized state keys")
        state = normalized
    return state


def inspect_state_dict(state: Mapping, backbone: str) -> str:
    """Identify a supported architecture using embedding and transformer shapes."""
    state = normalize_state_dict(state, backbone)
    key = "embed.weight" if backbone == "esmc" else "embed_tokens.weight"
    if key not in state or state[key].ndim != 2:
        raise ValidationError("Checkpoint lacks a recognized token embedding")
    vocab, width = state[key].shape
    if vocab != (64 if backbone == "esmc" else len(ALPHABET)):
        raise ValidationError("Checkpoint token vocabulary does not match its backbone")
    layer_pattern = re.compile(
        r"^transformer\.blocks\.(\d+)\." if backbone == "esmc" else r"^layers\.(\d+)\."
    )
    layers = {int(match.group(1)) for name in state if (match := layer_pattern.match(name))}
    if not layers or layers != set(range(max(layers) + 1)):
        raise ValidationError("Checkpoint transformer layers are missing or discontinuous")
    architectures = ESMC_ARCH if backbone == "esmc" else ESM2_ARCH
    width_key, layer_key = (
        ("d_model", "n_layers") if backbone == "esmc" else ("embed_dim", "num_layers")
    )
    matches = [
        name
        for name, spec in architectures.items()
        if spec[width_key] == width and spec[layer_key] == len(layers)
    ]
    if len(matches) != 1:
        raise ValidationError("Checkpoint shapes do not identify a supported architecture")
    return matches[0]


def restore_state(module: torch.nn.Module, state: Mapping, device="cpu"):
    """Strictly assign saved tensors and rebuild derived rotary buffers."""
    module.load_state_dict(state, strict=True, assign=True)
    from esmc.layers.rotary import RotaryEmbedding

    for layer in module.modules():
        if isinstance(layer, RotaryEmbedding) and layer.inv_freq.is_meta:
            layer.reset_parameters()
    return module.to(device).eval()


@dataclass
class LoadedModel:
    module: torch.nn.Module
    model_id: str
    backbone: str
    arch: str
    sequence_convention: str
    artifact_sha256: str
    manifest: dict
    catalog_sha256: str
    device: torch.device
    num_layers: int

    @property
    def info(self) -> dict:
        """Return model, artifact, runtime, and scoring identities."""
        return {
            "model_id": self.model_id,
            "backbone": self.backbone,
            "arch": self.arch,
            "sequence_convention": self.sequence_convention,
            "artifact_sha256": self.artifact_sha256,
            "manifest_sha256": self.manifest["sha256"],
            "catalog_sha256": self.catalog_sha256,
            "num_layers": self.num_layers,
            "device": str(self.device),
            "recipe_id": SCORING_RECIPE,
        }

    def score(
        self,
        data,
        *,
        mask_region: str = "peptide",
        cache_size: int = 256,
        species: str = "human",
    ) -> pd.DataFrame:
        """Reconstruct and score gene components with this exact release."""
        return self._score(
            data,
            mask_region=mask_region,
            cache_size=cache_size,
            species=species,
            prepared=False,
        )

    def _score_prepared(
        self,
        data,
        *,
        mask_region: str = "peptide",
        cache_size: int = 256,
    ) -> pd.DataFrame:
        """Score an internal prepared-sequence bundle."""
        return self._score(
            data,
            mask_region=mask_region,
            cache_size=cache_size,
            species="human",
            prepared=True,
        )

    def _score(self, data, *, mask_region, cache_size, species, prepared) -> pd.DataFrame:
        from . import __version__
        from .reconstruct._resources import validate_species

        if not prepared:
            species = validate_species(species)
        dataset_identity = (
            dict(data.attrs["dataset"])
            if isinstance(data, pd.DataFrame) and "dataset" in data.attrs
            else None
        )
        if prepared:
            frame, entries = prepare_inputs(data, self.sequence_convention)
        else:
            if mask_region in ("cdr", "cdr3", "bg"):
                raise ValidationError(
                    f"mask_region={mask_region!r} is unsupported for gene-component scoring. "
                    "Reconstruction provides inclusive-end CDR3 annotations"
                )
            from .reconstruct.components import prepare_gene_inputs

            frame, entries = prepare_gene_inputs(data, self.sequence_convention, species=species)
        reserved = {"pll", "model_id", "artifact_sha256", "inference_version"}
        if collision := reserved.intersection(frame.columns):
            raise ValidationError(f"Input uses reserved result columns: {sorted(collision)}")
        output = frame.copy()
        if prepared:
            output["pll"] = run_pll_benchmark(
                self.module, self.num_layers, entries, mask_region, self.device, cache_size
            )
        else:
            valid_entries = [entry for entry in entries if entry is not None]
            valid_scores = iter(
                run_pll_benchmark(
                    self.module,
                    self.num_layers,
                    valid_entries,
                    mask_region,
                    self.device,
                    cache_size,
                )
            )
            output["pll"] = [
                next(valid_scores) if entry is not None else float("nan") for entry in entries
            ]
        output["model_id"] = self.model_id
        output["artifact_sha256"] = self.artifact_sha256
        output["inference_version"] = __version__
        output.attrs["reproduction"] = {
            "model_id": self.model_id,
            "artifact_sha256": self.artifact_sha256,
            "manifest_sha256": self.manifest["sha256"],
            "catalog_sha256": self.catalog_sha256,
            "inference_version": __version__,
            "recipe_id": SCORING_RECIPE,
            "sequence_convention": self.sequence_convention,
            "mask_region": mask_region,
            "normalization": "full-vocabulary",
            "reduction": "mean-residue-log-probability",
            "device": str(self.device),
            "precision": str(next(self.module.parameters()).dtype),
            "batch_size": 1,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else "CPU",
        }
        if dataset_identity is not None:
            output.attrs["dataset"] = dataset_identity
        return output


def load(model_id: str, device="cuda", *, registry=None) -> LoadedModel:
    """Read current availability before opening an exact member's weights."""
    from .registry import Registry

    registry = registry if isinstance(registry, Registry) else Registry(registry)
    resolved = registry.resolve(model_id)
    member = resolved.member
    spec = model_spec(member["backbone"], member["arch"])
    if member["sequence_convention"] not in ("v1", "v2"):
        raise ValidationError("Unsupported sequence convention. Upgrade DecoderTCR")
    try:
        selected_device = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValidationError(f"Invalid device {device!r}") from exc
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValidationError("CUDA is unavailable. Request device='cpu' explicitly if appropriate")
    weights = registry.verify_artifact(resolved.bundle, member["weights"])
    try:
        state = load_file(str(weights), device="cpu")
        with torch.device("meta"):
            module = build_model(member["backbone"], member["arch"], device="meta")
        module = restore_state(module, state, selected_device)
    except (RuntimeError, ValueError, OSError) as exc:
        raise IntegrityError(
            "Released weights cannot be loaded with their declared architecture"
        ) from exc
    return LoadedModel(
        module,
        model_id,
        member["backbone"],
        member["arch"],
        member["sequence_convention"],
        member["weights"]["sha256"],
        resolved.manifest,
        resolved.catalog_sha256,
        selected_device,
        spec.get("n_layers", spec.get("num_layers")),
    )
