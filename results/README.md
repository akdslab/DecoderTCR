# Historical figures and IMMREP23 inputs

The retained figures and IMMREP23 table are historical public-repository materials. They are not results produced by `DecoderTCR==1.5.0` or by the six exact model artifacts.

`results/immrep23/immrep23_test.csv` remains a direct gene-input CSV. It contains all 3,484 shipped rows, including both `Usage=Private` and `Usage=Public`. Score it with the current gene-input API without filtering either group.

```bash
uv run --locked --no-dev decodertcr score \
  --model decodertcr@1.5.0:300M \
  --input results/immrep23/immrep23_test.csv \
  --output immrep23-scores.csv \
  --device cpu
```

The same bytes are registered as `immrep23-public-inputs@1.0.0`. Export the dataset and feed the file to the same `score` command.

```bash
uv run --locked --no-dev decodertcr dataset immrep23-public-inputs@1.0.0 \
  --output immrep23-public-inputs.csv
```

The input's `Label` and `Usage` fields are preserved as metadata. Score this CSV directly with the Python API or CLI. The `decodertcr reproduce` command instead requires a prepared reproduction dataset with matching registered result and environment bundles. No public result or environment entries are registered.
