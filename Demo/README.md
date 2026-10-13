# DecoderTCR quick start

`quick_start.py` scores the bundled gene-component rows with one exact model. It prints `name`, `peptide`, `reconstruction_ok`, and `pll`. Select the device explicitly and authenticate for the private default repository before running it.

```bash
uv run --locked --no-dev python Demo/quick_start.py \
  decodertcr@1.5.0:300M \
  --device cpu
```

The script accepts `--rows` to limit the input and owns its own `--registry` option. Use a local mirror for offline work.

```bash
uv run --locked --no-dev python Demo/quick_start.py \
  decodertcr@1.5.0:300M \
  --device cpu \
  --registry registry-mirror
```

This script runs direct scoring. It does not compare against registered expected results. The separate `decodertcr demo` command requires a registered demo result, dataset, and environment bundle. Those catalog entries are not currently available.

The same 30 rows are registered as `demo-gene-pairs@1.0.0`. Export and score them through the normal commands.

```bash
uv run --locked --no-dev decodertcr dataset demo-gene-pairs@1.0.0 \
  --output demo-gene-pairs.csv
uv run --locked --no-dev decodertcr score \
  --model decodertcr@1.5.0:300M \
  --input demo-gene-pairs.csv \
  --output scores.csv \
  --device cpu
```

The registry export is byte-for-byte identical to `Demo/sample_data/genes_pairs.csv`. It does not filter or transform rows.
