# DecoderTCR

DecoderTCR scores TCR and pMHC sequence complexes and designs peptide candidates. This checkout contains `DecoderTCR==1.5.0`. Package and model release versions are independent, and every model operation requires an exact model ID.

## Install and connect

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. Install the locked runtime dependencies from this checkout without the development group.

```bash
uv sync --locked --no-dev
```

This staging build defaults to the private `hf://Decoder-CHI/DecoderTCR-staging/registry`. Authenticate through the Hugging Face CLI or `HF_TOKEN`. Do not store a token in this repository. The `registry/` prefix holds model metadata and artifacts inside the Hugging Face repository.

```bash
uv run --locked --no-dev decodertcr models
```

The default private repository contains six models and two gene-input datasets. Select an explicit local mirror for offline work.

## Select an exact model

| Exact model ID | Nominal size | Reconstruction species |
|---|---:|---|
| `decodertcr@1.5.0:300M` | 300M | human, mouse |
| `decodertcr@1.5.0:600M` | 600M | human, mouse |
| `decodertcr@1.5.0:6B` | 6B | human, mouse |
| `decodertcr@0.3.0:300M` | 300M | human |
| `decodertcr@0.3.0:600M` | 600M | human |
| `decodertcr@0.3.0:6B` | 6B | human |

Nominal sizes are release member names, not measured parameter counts. There is no implicit model or short alias. APIs accept an exact ID or a loaded `LoadedModel` where documented.

Read metadata without downloading weights, then load the selected artifact.

```python
import DecoderTCR as dt

model_id = "decodertcr@1.5.0:300M"
metadata = dt.artifact_info(model_id)
loaded = dt.load(model_id, device="cuda")
```

`dt.artifact()` downloads and verifies an artifact but does not construct a model. `dt.load()` verifies the artifact and returns a `LoadedModel`.

## Score gene and CSV inputs

Model operations accept gene-component input. A row contains paired alpha and beta V genes, J genes, CDR3 sequences, an MHC allele, and a peptide. Extended CDR3 aliases take precedence over ordinary CDR3 columns. The exact model selects its sequence preparation automatically.

List the registered public inputs, then export an exact dataset as its original CSV bytes. The JSON sidecar records its manifest, artifact, and catalog identities.

```bash
uv run --locked --no-dev decodertcr datasets
uv run --locked --no-dev decodertcr dataset demo-gene-pairs@1.0.0 \
  --output demo-gene-pairs.csv
```

The commands above use the default private staging registry. The bundled CSV remains available for direct local input.

Score the bundled CSV example on CPU.

```bash
uv run --locked --no-dev decodertcr score \
  --model decodertcr@1.5.0:300M \
  --input Demo/sample_data/genes_pairs.csv \
  --output scores.csv \
  --device cpu
```

The result preserves input columns and adds `pll`, `model_id`, `artifact_sha256`, and `inference_version`. Higher `pll` is more binder-like under the selected model, not a calibrated binding probability. Gene scoring also returns `reconstruction_ok` and `reconstruction_reason`. Failed rows remain aligned with `pll=NaN` and a reason.

The equivalent Python workflow is:

```python
import pandas as pd
import DecoderTCR as dt

model_id = "decodertcr@1.5.0:300M"
rows = pd.read_csv("Demo/sample_data/genes_pairs.csv", keep_default_na=False)
loaded = dt.load(model_id, device="cpu")
scored = loaded.score(rows)
```

The dataset loader verifies the catalog, manifest, and CSV without loading model weights.

```python
available = dt.datasets()
info = dt.dataset_info("demo-gene-pairs@1.0.0")
rows = dt.dataset("demo-gene-pairs@1.0.0")
loaded = dt.load("decodertcr@1.5.0:300M", device="cpu")
scored = loaded.score(rows)
```

`dt.dataset()` keeps blank gene fields as blank strings. Python tables retain dataset, model, artifact, recipe, and reconstruction identities in `DataFrame.attrs`. CSV cannot preserve these attributes, so CLI score, profile, and design commands write JSON sidecars with input and output hashes, settings, artifact identity, and result metadata.

## Human gene input

This row matches `V01` in the bundled sample. Use `dt.list_alleles(model=...)` to inspect registered allele keys for the selected model. Allele availability does not establish predictive validation.

```python
human_row = {
    "name": "V01",
    "trav": "TRAV12-1",
    "traj": "TRAJ41",
    "cdr3a": "CVVNPSNSNSGYALNF",
    "trbv": "TRBV19",
    "trbj": "TRBJ2-7",
    "cdr3b": "CASSIRASHEQYF",
    "hla": "HLA-A*02:01",
    "peptide": "GILGFVFTL",
    "label": 1,
}
human_hla = dt.lookup_hla(
    "HLA-A*02:01",
    model="decodertcr@1.5.0:300M",
)
human_score = dt.score(
    [human_row],
    model="decodertcr@1.5.0:300M",
    device="cpu",
)
```

## Profiles and peptide design

The API provides conditional region and AA20 peptide profiles, sequence logos, sampling, refinement, and rescored peptide design.

```python
context = rows.iloc[0]
profile = dt.peptide_profile(context, 9, model=loaded)
candidates = dt.design_peptides(context, 9, model=loaded, n=20, seed=42)
rows.iloc[:1].to_csv("context.csv", index=False)
```

The CLI accepts the same gene-component input, but each profile or design call requires one complex. The commands below use the one-row `context.csv` created above.

```bash
uv run --locked --no-dev decodertcr profile \
  --model decodertcr@1.5.0:300M \
  --input context.csv \
  --output peptide-profile.csv \
  --region peptide \
  --length 9 \
  --device cpu

uv run --locked --no-dev decodertcr design \
  --model decodertcr@1.5.0:300M \
  --input context.csv \
  --output designed-peptides.csv \
  --length 9 \
  --num 20 \
  --seed 42 \
  --device cpu
```

Profile and design accept one complex as a mapping, `Series`, or one-row `DataFrame`, not a list of records.

DecoderTCR scoring masks the complete peptide in one forward pass. It averages true-residue log probabilities over the full model vocabulary. Conditional profiles normalize the standard 20 amino acids. Released inference runs in FP32. The selected model prepares gene inputs automatically.

## Demo and reproduction availability

`Demo/quick_start.py` is a direct scoring example. `decodertcr demo` and `decodertcr reproduce` require registered result, environment, and reproduction dataset bundles. Raw gene-input datasets are not reproduction bundles. No public result or environment entries are registered, so both commands are unavailable.

`viral-public-grid` is a historical cross-lineage reference dataset. Its negatives often contain foreign HLA. It is distinct from the within-allele Viral benchmark.

## Advanced registry selection

Remote registries use `hf://OWNER/REPOSITORY/PREFIX`. The global CLI `--registry` option precedes the subcommand. `init` has its own `--registry` option.

```bash
uv run --locked --no-dev decodertcr \
  --registry hf://OWNER/REPOSITORY/PREFIX \
  models

uv run --locked --no-dev decodertcr init \
  --registry registry-mirror
```

Python calls accept the same URI or a local mirror through `registry=`. `DECODERTCR_PUBLIC_REGISTRY` selects a registry for one process. A saved public selection applies after the environment variable. Selection precedence is an explicit argument, `DECODERTCR_PUBLIC_REGISTRY`, saved public configuration, then the staging default.

Offline work requires an explicit local mirror.

```bash
DECODERTCR_PUBLIC_REGISTRY=registry-mirror \
  uv run --locked --no-dev decodertcr models
```

## Migration from the legacy API

Legacy code such as `model, n_layers = dt.load("DecoderTCR-ESMC_600M")` now receives one `LoadedModel` from `dt.load("decodertcr@0.3.0:600M")`. The earlier ESM2 release is not in the six-model inventory and its old aliases have no mapping.

Scoring, embedding, profiling, and gene-component APIs require `model=<exact ID>` or `model=<LoadedModel>`. Prefer `loaded.score(data)` or `dt.score(data, model=loaded)`. Raw checkpoint paths, tuple handles, `num_layers`, and architecture overrides are removed. `dt.score_from_components(data, model=loaded)` remains an alias for gene-component scoring.

## Limits and licensing

The retained historical results do not validate this package or the six exact model artifacts. Redistribution rights and attribution for the bundled HLA and MHC references and benchmark datasets are not established by this source license. Do not publicly distribute those assets without separate permission.

First-party source is MIT licensed. Vendored backbone implementations retain their upstream license files. Model weights and datasets carry their own terms. The package license does not relicense weights, HLA reference data, or benchmark datasets.
