# fine-tune-creole-models

*STATUS: (as of Sep 1, 2026) - non-functional, in spec-driven development for MVP 0.1 release*

Public, release-safe components for fine-tuning models for under-resourced Louisiana languages, beginning with Cajun French and Kouri-Vini.

## Current scope

This repository currently contains:

- **`data_contract`** — the ingestion data contract and its eligibility pre-filter, keeping
  rights, training permission, cultural-sensitivity review, and release classification explicit
  before an item can enter annotation or model-training workflows.
- **`lid`** — a marker-wordlist language-identification heuristic distinguishing Cajun French,
  Kouri-Vini, French, English, and code-switched text.
- **`bakeoff`** — the base-model bake-off config loader and orchestration skeleton (candidate
  iteration, disqualification precedence, winner selection), with the actual fine-tuning, scoring,
  and red-team stages left as injectable seams for infrastructure not yet built.
- **`eval`** — the silver-vs-gold acceptance gate: a hard runtime check that refuses to compute an
  acceptance report against any non-gold-tagged test-split item.
- **`train`** — a LoRA/QLoRA/DoRA hyperparameter config loader with validation (alpha derived from
  rank, 4-bit quantization rejected, out-of-range values flagged) and a learning-rate sweep helper.
- **`coreset`** — a coverage scorecard reconciling the eligible item pool against a diff/edge-case
  catalog's per-cell coverage status, producing a prioritized next-collection list. The coreset
  *selection* algorithm itself remains an unbuilt seam.
- **`governance`** — an append-only consent-ledger event log, keyed to item id, recording every
  consent-tier grant or change (including withdrawal) with a full audit trail.
- **`tracking`** — a typed run-metadata record and an injectable persist seam, ahead of any real
  MLflow/DVC backend being wired in.

No training corpus or community-contributed language data is included in this release. Internal
planning documents, agent instructions, chat/session artifacts, and non-public material remain
outside this repository. Several modules ship real, decided configuration data
(`configs/models/bakeoff_candidates.yml`, `configs/training/lora_defaults.yml`,
`configs/diff_catalog/*.yml`) — not corpus data, but committed pipeline configuration each
module's tests load directly.

## Development

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --dev --frozen
uv run --frozen pytest -q
uv run --frozen mypy
```

## License

The code in this repository is licensed under the Apache License 2.0. Any future datasets will carry their own source-level rights and consent metadata and may use different terms.
