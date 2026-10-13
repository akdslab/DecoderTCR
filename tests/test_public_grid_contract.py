from test_benchmarks import convention_entry

from DecoderTCR.benchmarks import freeze_entries, inventory, population_digest


def test_public_grid_freezes_existing_pairs_without_cross_pairing():
    frames = {}
    for convention in ("v1", "v2"):
        rows = {}
        for index, (offered, label) in enumerate((("AAA", 1), ("CCC", 0))):
            row = convention_entry(convention, peptide=offered, clone="stable", label=label)
            row["meta_data"].update(true_epitope="AAA", tcr_convention=convention)
            row["pocket_idx"] = {"fixture": [1, 2]}
            rows[str(index)] = row
        frame, exclusions = freeze_entries(rows, "viral-public-grid", convention)
        assert not exclusions and len(frame) == 2
        assert frame.group_id.tolist() == ["AAA", "CCC"]
        assert frame.cognate_epitope.tolist() == ["AAA", "AAA"]
        assert frame.pocket_idx_json.nunique() == 1
        frames[convention] = frame
    assert population_digest(frames["v1"]) == population_digest(frames["v2"])
    assert set(frames["v1"].row_id).isdisjoint(frames["v2"].row_id)


def test_public_grid_retains_historical_reference_caveat():
    public = inventory()["viral-public-grid"]
    assert public["needs_cross_pair"] is False
    assert public["group_key"] == "offered_peptide"
    assert public["status"] == "historical_reference"
    assert "foreign HLA" in public["note"]
