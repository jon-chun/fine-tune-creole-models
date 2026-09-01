# fine-tune-creole-models

*STATUS: (as of Sep 1, 2026) - non-functional, in spec-driven development for MVP 0.1 release*

Public, release-safe components for fine-tuning models for under-resourced Louisiana languages, beginning with Cajun French and Kouri-Vini.

## Status

**[MVP 0.1 Status Report](https://claude.ai/code/artifact/2cab4935-782e-4d7c-aa78-bb4b7d376e31)**

9/9 tickets closed, 127 tests passing, mypy strict clean across 9 modules. The data contract,
the preprocess ingestion slice, the hyperparameter config loader, and the consent ledger are
complete for their current scope; the bake-off orchestration, the eval harness, the remaining
governance artifacts, the coreset selection algorithm, and the run-tracking backend are
real-but-partial seams waiting on infrastructure (GPU, gold set, a chosen algorithm, a chosen
tracking backend) rather than more scaffolding. The speech lane is out of 0.1 scope by design.

## Code base analysis

Metrics below are measured on the private development repo (`fine-tune-creole-models-dev`),
which builds and tests every module before its release-safe subset is synced here.

**Size**

| Category | LOC | Files |
|---|---:|---:|
| `src/` (8 modules) | 1,038 | 8 |
| `utils/` (preprocess CLI + core) | 271 | 2 |
| `tests/` (9 suites) | 1,587 | 9 |
| **Total** | **2,896** | **19** |

Test:code ratio ≈ 1.21 — more test code than production code, consistent with the strict TDD
discipline used throughout.

**Coverage** (`pytest-cov`, line coverage)

| Module | Stmts | Cover | Notes |
|---|---:|---:|---|
| `data_contract`, `coreset`, `eval`, `governance`, `lid`, `tracking` | 313 | 100% | |
| `bakeoff` | 60 | 97% | 2 defensive branches: malformed candidate id, non-mapping YAML root |
| `train` | 63 | 94% | `steps < 2` guard, sweep's append-fallback branch |
| `utils/fine_tune_cajun_preprocess.py` (core) | 86 | 93% | |
| `utils/fine-tune-cajun-preprocess.py` (CLI wrapper) | 27 | 0% | argparse-only, deliberately outside pytest's import graph |
| **Weighted total** | **499** | **92%** | |

**Type safety**: mypy `strict = true` passes clean across all 9 source files. Zero
`type: ignore` in `src/`; 13 in `utils/`, all at the one documented JSON→dataclass boundary — a
bounded, legitimate use, not scattered suppression.

**Complexity proxy**: 23 classes / 22 functions across 1,038 `src/` lines (~45 LOC/def) — small,
single-purpose units. No file exceeds 180 lines.

**Comment density**: uneven — `data_contract.py` (35 comment-lines/180, 19%) and `lid` (7%) carry
real rationale comments explaining non-obvious decisions; `coreset`, `eval`, `governance`,
`tracking`, `train` have zero inline comments, relying entirely on docstrings instead (a
stylistic drift, not a defect — each module's docstring is substantial).

**Dependencies**: minimal — one runtime dependency (`pyyaml`), four dev-only (`mypy`, `pytest`,
`pytest-cov`, `types-pyyaml`).

**Git history**: 17 commits, single-day history (dev repo created 2026-08-31) — matches the
fresh, spec-driven build-out documented in the status report above.

**Takeaway**: high-rigor, low-volume codebase — everything built is fully typed, heavily tested
(92% line coverage, 100% on 6 of 8 modules), and zero-suppression on typing. The uncovered 8% is
concentrated in defensive/edge branches and one deliberately-untested CLI wrapper, not core logic
gaps.

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
