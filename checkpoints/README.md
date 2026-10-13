# Model artifacts

DecoderTCR resolves immutable Safetensors artifacts through its registry. Weights do not belong in this source tree.

Inspect a local mirror without downloading the default remote artifact:

```bash
uv run --locked --no-dev decodertcr --registry registry-mirror models
```

Download and verify one exact artifact from the selected default repository:

```bash
uv run --locked --no-dev python scripts/download_weights.py decodertcr@1.5.0:300M
```

The verified file remains in the Hugging Face cache for a remote registry or at its existing location for an explicit local mirror.
