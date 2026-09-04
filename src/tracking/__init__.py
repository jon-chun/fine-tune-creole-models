"""Run-metadata record + injectable persist seam (tech-spec §10).

tech-spec §10 names MLflow (experiment tracking) and DVC (data/artifact
versioning) as new dependencies this repo needs, and its reproducibility
contract requires "sufficient run metadata to re-derive every reported
number." Installing either backend now, with no real training run to
track yet, would be premature — the same infra-before-need mistake this
repo has avoided in every prior ticket.

This module builds the typed run-metadata shape and a pure record_run()
function instead: `persist` is an injectable seam (Callable[[RunMetadata],
None]), matching the pattern already used for src/bakeoff/'s fine_tune/
score/run_red_team callables.

ADR 0016 records the backend choice (MLflow + DVC on S3-compatible storage,
per tech-spec §10) and `src/tracking/backends.py` (issue #37, backlog 0014)
implements the offline local file store first: `LocalFileBackend.persist`
is a real, usable `persist` target, plus `flatten_config`/`to_mlflow_payload`
(config flattening for MLflow's params/metrics/tags) and
`read_git_commit_sha` (a `git rev-parse HEAD` reader). The real MLflow
server/DVC remote wiring itself is still deferred (T-006) — record_run()'s
validation logic below is unaffected either way.

No file, network, or git subprocess I/O happens in this module itself;
that lives in `src/tracking/backends.py`. Callers supply git_commit_sha,
started_at, and completed_at explicitly (backends.read_git_commit_sha is
one way to obtain git_commit_sha).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, get_args

from data_contract import LanguageTag, validate_literal

# Stage: tech-spec v2 §1's repo layout adds src/speech_eval/, src/align/,
# src/hitl/ as new module directories (decision brief §3 item 7: "and
# stages `speech_eval`, `align`, `hitl`"), so those three stages are added
# alongside the original five even though the modules themselves don't
# exist yet — a run-metadata home is needed from day one (MIG-01g story 6).
Stage = Literal[
    "bakeoff", "train", "eval", "preprocess", "augment", "speech_eval", "align", "hitl"
]

# _STAGES derives from get_args(Stage) rather than a hand-maintained set, so
# extending Stage above requires no change here (verified: this was already
# true before MIG-01g).
_STAGES = frozenset(get_args(Stage))


class RunMetadataError(ValueError):
    """Raised when record_run() is given invalid RunMetadata, or a
    RunMetadata is constructed with a stage outside its closed set (issue
    #15) — before `persist` is ever called."""


class FrozenMapping(Mapping[str, object]):
    """A truly hashable, immutable string-keyed mapping.

    `types.MappingProxyType` is read-only but NOT hashable — it is a view
    over a (possibly mutable) backing dict, and delegates `__hash__` to it
    (`hash(MappingProxyType({}))` raises TypeError: unhashable type: 'dict').
    RunMetadata.config needs a mapping that is actually hashable so
    `hash(run_metadata)` works, hence this class instead."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, object]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        return hash(frozenset(self._data.items()))

    def __repr__(self) -> str:
        return f"FrozenMapping({self._data!r})"


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """What ran, with what config, producing what result, when. `config`
    is deliberately typed as a read-only mapping — different stages need
    different config shapes, and this module does not depend on
    bakeoff.Candidate, train.Hyperparameters, or any eval config type.

    frozen=True/slots=True is meant to make RunMetadata hashable (a run
    record is a natural set/dict-key candidate), so every field is an
    immutable, hashable type: `config` is a FrozenMapping (build via
    `to_hashable_config()` below) and `artifact_refs`/`warnings` are tuples,
    never a plain dict/list — those raise TypeError under hash().

    `seed` stays a single required `int`: one `RunMetadata` is one run, and
    one run is one seed's execution (issue #28's MIG-01d discussion of
    `Hyperparameters.seeds`, resolved there — a *bake-off* aggregates across
    multiple `RunMetadata` records with different seeds; `seed` here is
    never a list). `src/train/`'s `Hyperparameters.seeds: tuple[int, ...]`
    (MIG-01d) is the separate, configured seed list a bake-off iterates
    over — the two are not the same shape and are not meant to be.

    `image_digest`, `manifest_sha256`, `gpu_hours`, `usd`, `instance` are
    new v2 fields (tech-spec v2 §10: "every run logs config YAML, dev-repo
    git SHA, image digest, dataset manifest hash, seeds, metrics, adapter
    URI, GPU-hours and dollars"; decision brief §3 item 7). All five are
    `| None` with a `None` default — a `preprocess`-stage run has no
    meaningful GPU-hours/cost/instance/image, mirroring `src/train/`'s
    existing "optional until derivable" posture for corpus-derived knobs
    (issue #20). `manifest_sha256` is a single hash summarizing the whole
    `--cloud-ok-only` manifest (MIG-01b), not a per-item map. `instance` is
    free text (e.g. `"H100-80GB"`), not a closed enum, since tech-spec v2
    §10's cost table lists several interchangeable providers/instance
    shapes."""

    run_id: str
    stage: Stage
    language: LanguageTag
    config: FrozenMapping
    git_commit_sha: str
    started_at: datetime
    completed_at: datetime
    artifact_refs: tuple[str, ...]
    seed: int
    split_id: str
    lock_hash: str
    tree_dirty: bool
    warnings: tuple[str, ...] = ()
    image_digest: str | None = None
    manifest_sha256: str | None = None
    gpu_hours: float | None = None
    usd: float | None = None
    instance: str | None = None

    def __post_init__(self) -> None:
        # Same enum-membership discipline as DataItem's own __post_init__
        # (issue #15): a bad stage value must be rejected at construction,
        # not silently accepted into a run record.
        try:
            validate_literal(self.stage, tuple(_STAGES), "stage")
        except ValueError as exc:
            raise RunMetadataError(str(exc)) from exc


def new_run_id() -> str:
    """The project's one run-id-generation convention. Future tickets call
    this rather than inventing their own scheme."""
    return str(uuid.uuid4())


def to_hashable_config(config: dict[str, object]) -> FrozenMapping:
    """Convert a plain config dict (e.g. `dataclasses.asdict(hyperparameters)`)
    into the hashable mapping RunMetadata.config expects. Recursive: nested
    `dict`/`list` values are converted too (list -> tuple, dict ->
    FrozenMapping), since a stage's config may nest either."""

    def _freeze(value: object) -> object:
        if isinstance(value, dict):
            return FrozenMapping({k: _freeze(v) for k, v in value.items()})
        if isinstance(value, list):
            return tuple(_freeze(v) for v in value)
        return value

    return FrozenMapping({k: _freeze(v) for k, v in config.items()})


def warnings_for_run(loaded_config_warnings: Sequence[str]) -> tuple[str, ...]:
    """Carry a loaded config's warnings (e.g. train.LoadedTrainingConfig.warnings)
    into RunMetadata.warnings. Takes a plain Sequence[str] rather than
    train.LoadedTrainingConfig itself — this module must not import train
    (no cross-sibling imports inside src/)."""
    return tuple(loaded_config_warnings)


def record_run(metadata: RunMetadata, *, persist: Callable[[RunMetadata], None]) -> None:
    """Validates `metadata`, then calls `persist(metadata)` exactly once.
    Raises RunMetadataError — without ever calling `persist` — if run_id is
    empty or completed_at is before started_at. Any exception `persist`
    itself raises propagates directly; this function does not catch or
    convert it."""
    if not metadata.run_id:
        raise RunMetadataError("RunMetadata.run_id is required and must be non-empty")
    if metadata.completed_at < metadata.started_at:
        raise RunMetadataError(
            f"RunMetadata.completed_at ({metadata.completed_at}) is before "
            f"started_at ({metadata.started_at})"
        )
    persist(metadata)
