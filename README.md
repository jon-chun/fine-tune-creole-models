# fine-tune-creole-models

Public, release-safe components for fine-tuning models for under-resourced Louisiana languages, beginning with Cajun French and Kouri-Vini.

## Current scope

This repository currently contains the ingestion data contract and its eligibility pre-filter. The filter keeps rights, training permission, cultural-sensitivity review, and release classification explicit before an item can enter annotation or model-training workflows.

No training corpus or community-contributed language data is included in this release. Internal planning documents, agent instructions, chat/session artifacts, and non-public material remain outside this repository.

## Development

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --dev --frozen
uv run --frozen pytest -q
uv run --frozen mypy
```

## License

The code in this repository is licensed under the Apache License 2.0. Any future datasets will carry their own source-level rights and consent metadata and may use different terms.
