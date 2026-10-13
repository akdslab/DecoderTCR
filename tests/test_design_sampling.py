"""Design algorithm parity checks against DecoderTCR release e842d5b4."""

import numpy as np
import pandas as pd
import pytest
import torch

from DecoderTCR.constants import AA20, AA20_IDS
from DecoderTCR.core import ValidationError
from DecoderTCR.design import (
    FORWARD_BUDGET_PER_DESIGN,
    _block_gibbs,
    _entropy,
    _iegr,
    _iegr_profile,
    _profile_row,
    _sample,
    block_gibbs,
    design_peptides,
    iegr,
    iegr_profile,
    sample_from_profile,
)


def make_profile(matrix):
    profile = pd.DataFrame(np.asarray(matrix, dtype=np.float64), columns=list(AA20))
    profile.index = pd.RangeIndex(1, len(profile) + 1, name="position")
    profile["entropy"] = 0.0
    return profile


def flat_profile(length=9):
    return make_profile(np.full((length, 20), 1 / 20))


def one_hot_profile(length=9):
    matrix = np.zeros((length, 20))
    matrix[:, 0] = 1
    return make_profile(matrix)


def _committed_reference_sample(profile, n, temperature, seed, cap):
    """Independent transcription of the committed e842d5b4 sampler."""
    probabilities = profile[list(AA20)].to_numpy(dtype=np.float64, copy=True)
    if temperature <= 0:
        return ["".join(profile[list(AA20)].idxmax(axis=1))]
    if temperature != 1:
        logp = np.log(np.clip(probabilities, 1e-12, None)) / temperature
        probabilities = np.exp(logp - logp.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    seen = set()
    result = []
    used = 0
    while len(result) < n and used < cap:
        wanted = n - len(result)
        size = min(250_000, cap - used, max(1024, 4 * wanted))
        draw = np.stack([rng.choice(20, size=size, p=row) for row in probabilities], axis=1)
        used += size
        for indices in draw:
            sequence = "".join(AA20[index] for index in indices)
            if sequence not in seen:
                seen.add(sequence)
                result.append(sequence)
                if len(result) == n:
                    break
    return result


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_sampler_matches_independent_committed_reference(temperature):
    profile = make_profile(np.tile(np.arange(1, 21), (5, 1)))
    actual, _ = sample_from_profile(profile, n=25, temperature=temperature, seed=17, cap=5000)
    expected = _committed_reference_sample(profile, 25, temperature, 17, 5000)
    assert actual == expected


def test_sampler_preserves_draw_order_and_reports_budget_state():
    sampled, stats = sample_from_profile(flat_profile(), n=2000, seed=0, cap=1500)
    assert len(sampled) == len(set(sampled)) == 1500
    assert stats == {
        "n_requested": 2000,
        "n_returned": 1500,
        "n_draws_used": 1500,
        "saturated": True,
        "cap_reached": True,
        "support_exhausted": False,
    }


def test_sampler_distinguishes_support_exhaustion_and_keeps_stats_shape():
    _, exhausted = sample_from_profile(one_hot_profile(), n=20, cap=5000)
    _, consensus_only = sample_from_profile(flat_profile(), n=20, temperature=0)
    assert exhausted["support_exhausted"] is True
    assert set(exhausted) == set(consensus_only)


def test_temperature_changes_distribution_shape():
    matrix = np.full((9, 20), 0.1 / 19)
    matrix[:, 0] = 0.9
    profile = make_profile(matrix)
    cold, _ = sample_from_profile(profile, n=200, temperature=0.5, seed=0)
    hot, _ = sample_from_profile(profile, n=200, temperature=2.0, seed=0)
    cold_distance = np.mean([sum(amino_acid != "A" for amino_acid in seq) for seq in cold])
    hot_distance = np.mean([sum(amino_acid != "A" for amino_acid in seq) for seq in hot])
    assert hot_distance > cold_distance + 1


def test_sampler_rejects_invalid_scalar_inputs():
    profile = flat_profile()
    for kwargs in ({"n": 0}, {"cap": 0}, {"n": True}, {"temperature": np.nan}):
        with pytest.raises(ValidationError):
            sample_from_profile(profile, **kwargs)


def test_iegr_sampling_and_entropy_are_float32_over_aa20_only():
    logits = torch.randn(4, 64, dtype=torch.float64)
    logits[:, 63] = 1000
    entropy = _entropy(logits)
    assert entropy.dtype == torch.float32
    row = _profile_row(logits[0])
    assert row.dtype == np.float64 and row.sum() == pytest.approx(1)
    generator = torch.Generator(device="cpu").manual_seed(2)
    token = _sample(logits[0], 1.0, generator)
    assert token in AA20_IDS


class PolicyModel(torch.nn.Module):
    def __init__(self, policy="always_w"):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.policy = policy
        self.calls = 0

    def forward(self, tokens, repr_layers=None, return_contacts=False):
        self.calls += 1
        logits = torch.full((*tokens.shape, 64), -30.0)
        w_id = AA20_IDS[AA20.index("W")]
        for token_index in range(tokens.shape[1]):
            target = w_id
            if self.policy == "identity" and token_index > 0:
                current = int(tokens[0, token_index])
                if current in AA20_IDS:
                    target = current
            logits[0, token_index, target] = 30
        return {"logits": logits}


class Handle:
    def __init__(self, module=None, backbone="esmc"):
        self.module = module or PolicyModel()
        self.model_id = "fixture@1.0.0:a"
        self.artifact_sha256 = "b" * 64
        self.backbone = backbone
        self.sequence_convention = "v1"
        self.device = torch.device("cpu")
        self.num_layers = 2
        self.score_handles = []

    @property
    def info(self):
        return {
            "model_id": self.model_id,
            "artifact_sha256": self.artifact_sha256,
            "manifest_sha256": "c" * 64,
            "catalog_sha256": "d" * 64,
            "backbone": self.backbone,
            "arch": "fixture",
            "sequence_convention": self.sequence_convention,
            "num_layers": self.num_layers,
            "device": str(self.device),
        }

    def _score_prepared(self, data, **kwargs):
        self.score_handles.append((data, kwargs))
        return pd.DataFrame({"pll": np.linspace(-1, -0.1, len(data))})


CONTEXT = {"HLA_a": "AAA", "HLA_b": "", "TCR_a": "", "TCR_b": ""}


def test_iegr_phase2_uses_committed_checkpoint_interval():
    loaded = Handle()
    output = _iegr(
        loaded,
        CONTEXT,
        length=4,
        n=20,
        temperature=0,
        gibbs_rounds=2,
        gibbs_subset_size=1,
        prepared=True,
    )
    assert loaded.module.calls == 4 + 8
    assert output["phase"].tolist()[0] == 1
    assert output.attrs["reproduction"]["model_id"] == loaded.model_id


def test_iegr_profile_commits_lowest_entropy_first_and_costs_one_forward_per_residue():
    class EntropyModel(PolicyModel):
        def forward(self, tokens, repr_layers=None, return_contacts=False):
            self.calls += 1
            logits = torch.zeros((*tokens.shape, 64))
            for token_index in range(tokens.shape[1]):
                logits[0, token_index, AA20_IDS[0]] = token_index
            return {"logits": logits}

    loaded = Handle(EntropyModel())
    profile = _iegr_profile(loaded, CONTEXT, length=5, temperature=0, prepared=True)
    assert loaded.module.calls == 5
    assert sorted(profile["commit_order"]) == [1, 2, 3, 4, 5]
    assert profile[list(AA20)].to_numpy().dtype == np.float64


def test_iterative_masking_warns_for_esm2():
    loaded = Handle(backbone="esm2")
    with pytest.warns(UserWarning, match="token-dropout"):
        _iegr_profile(loaded, CONTEXT, length=3, prepared=True)


def test_block_gibbs_changes_exactly_k_positions_in_one_round():
    for k in (1, 3, 5):
        loaded = Handle()
        output, stats = _block_gibbs(
            loaded,
            CONTEXT,
            ["ACDEFGHIK"],
            k=k,
            rounds=1,
            temperature=0,
            prepared=True,
        )
        assert output[0].count("W") == k
        assert stats["n_forwards"] == 1


def test_block_gibbs_retains_chain_history_and_honors_budget():
    loaded = Handle()
    output, stats = _block_gibbs(
        loaded,
        CONTEXT,
        ["ACDEFGHIK"],
        k=2,
        n=100,
        temperature=0,
        max_forwards=6,
        prepared=True,
    )
    assert stats["n_forwards"] == 6
    assert stats["budget_exhausted"] is True
    assert max(sequence.count("W") for sequence in output) > 2


def test_block_gibbs_default_target_budget_is_per_design():
    loaded = Handle()
    _, stats = _block_gibbs(
        loaded,
        CONTEXT,
        ["A"],
        k=1,
        n=50,
        temperature=0,
        prepared=True,
    )
    assert stats["n_forwards"] == FORWARD_BUDGET_PER_DESIGN * 50
    assert stats["budget_exhausted"] is True


def test_design_rescore_reuses_same_handle_and_records_identity(monkeypatch):
    loaded = Handle()
    profile = flat_profile(3)
    seen = []
    entry = {
        "sequences": {**CONTEXT, "Peptide": "AAA"},
        "pocket_idx": {"CDRa": [], "CDRb": []},
        "annotation_contract": "inclusive-end",
    }

    def fake_profile(handle, data, region, length, **kwargs):
        seen.append(handle)
        return profile

    monkeypatch.setattr(
        "DecoderTCR.reconstruct.components.prepare_gene_inputs",
        lambda *args, **kwargs: (pd.DataFrame([{"reconstruction_ok": True}]), [entry]),
    )
    monkeypatch.setattr("DecoderTCR.design._region_profile", fake_profile)
    output = design_peptides(
        loaded,
        {"trav": "TRAV1-1"},
        length=3,
        n=5,
        seed=0,
        rescore=True,
    )
    assert seen == [loaded]
    assert len(loaded.score_handles) == 1
    assert output["pll"].is_monotonic_decreasing
    assert output.attrs["reproduction"]["artifact_sha256"] == loaded.artifact_sha256


def test_composite_gene_design_reconstructs_once(monkeypatch):
    from unittest.mock import Mock

    entry = {
        "sequences": {
            "HLA_a": "AAA",
            "HLA_b": "",
            "Peptide": "AAA",
            "TCR_a": "",
            "TCR_b": "",
        },
        "pocket_idx": {"CDRa": [], "CDRb": []},
        "annotation_contract": "inclusive-end",
        "reconstruction": {"sequence_convention": "v1"},
    }
    prepare = Mock(return_value=(pd.DataFrame([{"reconstruction_ok": True}]), [entry]))
    monkeypatch.setattr("DecoderTCR.reconstruct.components.prepare_gene_inputs", prepare)
    loaded = Handle()

    output = design_peptides(
        loaded,
        {"trav": "TRAV1-1"},
        length=3,
        n=2,
        seed=0,
        rescore=True,
        gibbs_k=1,
        gibbs_max_forwards=2,
    )

    assert prepare.call_count == 1
    assert len(loaded.score_handles) == 1
    assert output.attrs["reconstruction"] == entry["reconstruction"]


def test_cdr3_design_uses_inclusive_endpoint_and_requires_contract():
    loaded = Handle()
    data = {
        "sequences": {
            "HLA_a": "AAA",
            "HLA_b": "",
            "Peptide": "ACD",
            "TCR_a": "EFGHI",
            "TCR_b": "",
        },
        "pocket_idx": {"CDRa": [[0, 0], [1, 1], [2, 4]]},
        "annotation_contract": "inclusive-end",
    }
    output = _iegr(
        loaded,
        data,
        region="cdr3a",
        n=1,
        temperature=0,
        gibbs_rounds=0,
        prepared=True,
    )
    assert output["sequence"].str.len().tolist() == [3]
    del data["annotation_contract"]
    with pytest.raises(ValidationError, match="annotation_contract"):
        _iegr(
            loaded,
            data,
            region="cdr3a",
            gibbs_rounds=0,
            prepared=True,
        )


def test_public_gene_cdr3_design_is_retained_and_all_cdr_is_rejected(monkeypatch):
    loaded = Handle()
    entry = {
        "sequences": {
            "HLA_a": "AAA",
            "HLA_b": "",
            "Peptide": "ACD",
            "TCR_a": "EFGHI",
            "TCR_b": "",
        },
        "pocket_idx": {"CDRa": [[2, 4]], "CDRb": []},
        "annotation_contract": "inclusive-end",
    }

    def prepare(*args, **kwargs):
        return pd.DataFrame([{"reconstruction_ok": True}]), [entry]

    monkeypatch.setattr("DecoderTCR.reconstruct.components.prepare_gene_inputs", prepare)

    output = iegr(
        loaded,
        {"trav": "TRAV1-1"},
        region="cdr3a",
        n=1,
        temperature=0,
        gibbs_rounds=0,
    )
    assert output["sequence"].str.len().tolist() == [3]
    with pytest.raises(ValidationError, match="Unknown region"):
        iegr(loaded, {"trav": "TRAV1-1"}, region="cdr", gibbs_rounds=0)


@pytest.mark.parametrize("operation", ["profile", "design", "gibbs", "peptide_design"])
def test_public_design_operations_accept_all_documented_single_context_forms(
    monkeypatch, operation
):
    context = {
        "trav": "TRAV1-1",
        "traj": "TRAJ1",
        "cdr3a": "CAVF",
        "trbv": "TRBV1",
        "trbj": "TRBJ1-1",
        "cdr3b": "CASSF",
        "hla": "A0201",
        "peptide": "ACD",
    }
    entry = {
        "sequences": {
            "HLA_a": "AAA",
            "HLA_b": "",
            "Peptide": "ACD",
            "TCR_a": "EFGHI",
            "TCR_b": "",
        },
        "pocket_idx": {"CDRa": [[2, 4]], "CDRb": []},
        "annotation_contract": "inclusive-end",
        "reconstruction": {"sequence_convention": "v1"},
    }
    seen = []

    def prepare(row, convention, *, species):
        assert isinstance(row, dict)
        assert convention == "v1"
        assert species == "human"
        seen.append(row)
        return pd.DataFrame([{"reconstruction_ok": True}]), [entry]

    monkeypatch.setattr("DecoderTCR.reconstruct.components.prepare_gene_inputs", prepare)
    forms = [context, pd.Series(context), pd.DataFrame([context])]
    outputs = []
    for form in forms:
        loaded = Handle()
        if operation == "profile":
            outputs.append(iegr_profile(loaded, form, region="cdr3a", temperature=0, seed=42))
        elif operation == "design":
            outputs.append(
                iegr(
                    loaded,
                    form,
                    region="cdr3a",
                    n=1,
                    temperature=0,
                    gibbs_rounds=0,
                    seed=42,
                )
            )
        elif operation == "gibbs":
            outputs.append(
                block_gibbs(
                    loaded,
                    form,
                    ["AAA"],
                    region="cdr3a",
                    k=1,
                    rounds=1,
                    temperature=0,
                    seed=42,
                )
            )
        else:
            outputs.append(
                design_peptides(
                    loaded,
                    form,
                    3,
                    n=1,
                    temperature=0,
                    seed=42,
                    rescore=False,
                )
            )

    if operation == "gibbs":
        assert outputs[1:] == outputs[:1] * 2
    else:
        for output in outputs[1:]:
            pd.testing.assert_frame_equal(output, outputs[0])
            assert output.attrs == outputs[0].attrs
    expected_context = {**context, "peptide": "AAA"} if operation == "peptide_design" else context
    assert [row for row in seen] == [expected_context, expected_context, expected_context]
