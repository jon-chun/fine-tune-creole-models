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
score/run_red_team callables. A future ticket chooses and wires the real
MLflow/DVC (or other) backend behind that one signature — record_run()'s
validation logic never needs to change when that happens.

No file, network, or git subprocess I/O happens in this module. Callers
supply git_commit_sha, started_at, and completed_at explicitly.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from data_contract import LanguageTag

Stage = Literal["bakeoff", "train", "eval"]


class RunMetadataError(ValueError):
    """Raised when record_run() is given invalid RunMetadata — before
    `persist` is ever called."""


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """What ran, with what config, producing what result, when. `config`
    is deliberately typed as a plain dict — different stages need
    different config shapes, and this module does not depend on
    bakeoff.Candidate, train.Hyperparameters, or any eval config type."""

    run_id: str
    stage: Stage
    language: LanguageTag
    config: dict[str, object]
    git_commit_sha: str
    started_at: datetime
    completed_at: datetime
    artifact_refs: list[str]


def new_run_id() -> str:
    """The project's one run-id-generation convention. Future tickets call
    this rather than inventing their own scheme."""
    return str(uuid.uuid4())


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
