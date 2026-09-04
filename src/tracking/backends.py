"""Local file tracking backend, MLflow payload flattening, git SHA reader.

Backlog 0014 / tech-spec v2 §10 chose MLflow (experiment tracking) + DVC
(data/artifact versioning) as the real backend, with the offline local file
store staying as the fallback and test double (ADR 0016 records the choice
and why the local backend is not merely scaffolding). Issue #23 row D9:
"MLflow flattening of `config`; who reads `git rev-parse`" — both answered
here. No mlflow/DVC dependency is added in this ticket (T-006 wires the
remote); `MlflowClientLike` is a Protocol a Wave 3 ticket implements against
a real `mlflow` client without this module importing it.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Protocol

from tracking import FrozenMapping, RunMetadata

# Scalar types tech-spec §10's MLflow params/tags accept without further
# conversion; anything else (nested FrozenMapping/tuple) is flattened by
# flatten_config below rather than passed through as-is.
_ConfigScalar = str | int | float | bool


class TrackingBackendError(RuntimeError):
    """Raised for backend I/O failures (git subprocess, file writes) that
    are not RunMetadata validation errors — record_run()'s RunMetadataError
    stays reserved for validation, never backend failure (issue #37)."""


def _freeze_to_plain(value: object) -> object:
    """Invert to_hashable_config's FrozenMapping/tuple freezing for JSON
    serialization: FrozenMapping -> dict, tuple -> list, everything else
    passed through unchanged (str/int/float/bool/None already JSON-native)."""
    if isinstance(value, FrozenMapping):
        return {k: _freeze_to_plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_freeze_to_plain(v) for v in value]
    return value


def run_metadata_to_dict(metadata: RunMetadata) -> dict[str, object]:
    """RunMetadata -> a plain JSON-serializable dict. datetimes become
    ISO-8601 strings (`datetime.isoformat()`); `config` (a FrozenMapping)
    becomes a plain dict; `artifact_refs`/`warnings` (tuples) become lists.
    Every dataclass field is included by name, so a future RunMetadata field
    round-trips without this function changing (iterates `dataclasses.fields`
    rather than naming each field)."""
    result: dict[str, object] = {}
    for field in fields(metadata):
        value = getattr(metadata, field.name)
        if isinstance(value, datetime):
            result[field.name] = value.isoformat()
        elif isinstance(value, FrozenMapping):
            result[field.name] = _freeze_to_plain(value)
        elif isinstance(value, tuple):
            result[field.name] = [_freeze_to_plain(v) for v in value]
        else:
            result[field.name] = value
    return result


def _plain_to_frozen(value: object) -> object:
    """Invert _freeze_to_plain: dict -> FrozenMapping, list -> tuple,
    recursively, for reconstructing RunMetadata.config from parsed JSON."""
    if isinstance(value, dict):
        return FrozenMapping({k: _plain_to_frozen(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_plain_to_frozen(v) for v in value)
    return value


def run_metadata_from_dict(data: Mapping[str, object]) -> RunMetadata:
    """Inverse of run_metadata_to_dict: parses ISO-8601 datetime strings back
    into `datetime`, rebuilds `config` as a FrozenMapping, and
    `artifact_refs`/`warnings` as tuples. Raises RunMetadataError (via
    RunMetadata.__post_init__) if `stage` is invalid, matching
    construction-time validation elsewhere in this repo (issue #15)."""
    started_at = data["started_at"]
    completed_at = data["completed_at"]
    assert isinstance(started_at, str)
    assert isinstance(completed_at, str)
    config = data["config"]
    assert isinstance(config, dict)
    artifact_refs = data["artifact_refs"]
    assert isinstance(artifact_refs, list)
    warnings = data.get("warnings", [])
    assert isinstance(warnings, list)

    kwargs: dict[str, object] = dict(data)
    kwargs["started_at"] = datetime.fromisoformat(started_at)
    kwargs["completed_at"] = datetime.fromisoformat(completed_at)
    kwargs["config"] = _plain_to_frozen(config)
    kwargs["artifact_refs"] = tuple(artifact_refs)
    kwargs["warnings"] = tuple(warnings)
    return RunMetadata(**kwargs)  # type: ignore[arg-type]


class LocalFileBackend:
    """Offline local-file `persist` target (ADR 0016): the fallback backend
    when no MLflow server is configured, and the test double every existing
    tracking test with a real backend should be able to use instead of a
    hand-rolled list-append double.

    Layout under `root`: `runs/<run_id>.json` (one file per run, full
    RunMetadata) and `runs/index.jsonl` (one line appended per run, the same
    dict as the JSON file — an append-only index mirroring the ledger JSONL
    convention governance already uses)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._runs_dir = root / "runs"

    def persist(self, metadata: RunMetadata) -> None:
        """Writes `runs/<run_id>.json` and appends one line to
        `runs/index.jsonl`. Directories are created as needed. Matches the
        `Callable[[RunMetadata], None]` shape record_run() calls exactly
        once (issue #37 acceptance: persist is called exactly once)."""
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        payload = run_metadata_to_dict(metadata)
        run_json_path = self._runs_dir / f"{metadata.run_id}.json"
        run_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        index_path = self._runs_dir / "index.jsonl"
        with index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def flatten_config(config: Mapping[str, object]) -> dict[str, str | int | float | bool]:
    """Flattens a (possibly nested) config mapping into dotted-key scalar
    pairs, the shape MLflow's `log_param`/`log_params` require (MLflow
    params/tags are flat string-keyed scalars; tech-spec §10 names MLflow
    flattening of `config` as issue #23 row D9's hardest question).

    Nesting: a FrozenMapping or plain dict value recurses with
    `"{parent}.{child}"` keys; a tuple/list value's items get
    `"{parent}.{index}"` keys (0-based). Every leaf value is coerced to
    `str` unless it is already `int`/`float`/`bool` (MLflow accepts those
    four types directly; anything else — e.g. `None` — is stringified so no
    value is silently dropped)."""
    result: dict[str, str | int | float | bool] = {}

    def _walk(prefix: str, value: object) -> None:
        if isinstance(value, FrozenMapping | dict):
            for key, nested in value.items():
                _walk(f"{prefix}.{key}" if prefix else str(key), nested)
        elif isinstance(value, tuple | list):
            for index, nested in enumerate(value):
                _walk(f"{prefix}.{index}" if prefix else str(index), nested)
        elif isinstance(value, bool | int | float):
            result[prefix] = value
        else:
            result[prefix] = str(value)

    for key, value in config.items():
        _walk(key, value)
    return result


class MlflowRunPayload(Protocol):
    """Structural shape of the three dicts `to_mlflow_payload` returns —
    named here so a Wave 3 ticket's `MlflowClientLike.log_*` calls can be
    typed against it without this module depending on `mlflow`."""

    params: Mapping[str, str | int | float | bool]
    metrics: Mapping[str, float]
    tags: Mapping[str, str]


def to_mlflow_payload(
    metadata: RunMetadata,
) -> tuple[dict[str, str | int | float | bool], dict[str, float], dict[str, str]]:
    """Splits a RunMetadata into MLflow's three logging surfaces:
    `(params, metrics, tags)`.

    - `params`: the flattened `config` (via flatten_config) plus the
      run-identifying, non-numeric knobs (`run_id`, `stage`, `language`,
      `git_commit_sha`, `split_id`, `lock_hash`, `seed`, `image_digest`,
      `manifest_sha256`, `instance`) that MLflow convention logs as params
      rather than metrics — none of these change during a run.
    - `metrics`: the numeric, potentially-plottable-over-time fields
      (`gpu_hours`, `usd`); `None` values are omitted rather than logged as
      0.0, since "not applicable to this stage" and "measured as zero" are
      different facts (tech-spec §10: a `preprocess`-stage run has no
      meaningful GPU-hours/cost).
    - `tags`: free-text/boolean metadata MLflow conventionally tags rather
      than params/metrics: `tree_dirty` (stringified) and `warnings` (joined
      with `"; "`, empty string if none).

    `started_at`/`completed_at`/`artifact_refs` are not included — those are
    better carried by MLflow's own run start/end times and artifact logging
    API respectively (out of scope for a plain params/metrics/tags split)."""
    params: dict[str, str | int | float | bool] = dict(flatten_config(metadata.config))
    params["run_id"] = metadata.run_id
    params["stage"] = metadata.stage
    params["language"] = metadata.language
    params["git_commit_sha"] = metadata.git_commit_sha
    params["split_id"] = metadata.split_id
    params["lock_hash"] = metadata.lock_hash
    params["seed"] = metadata.seed
    if metadata.image_digest is not None:
        params["image_digest"] = metadata.image_digest
    if metadata.manifest_sha256 is not None:
        params["manifest_sha256"] = metadata.manifest_sha256
    if metadata.instance is not None:
        params["instance"] = metadata.instance

    metrics: dict[str, float] = {}
    if metadata.gpu_hours is not None:
        metrics["gpu_hours"] = metadata.gpu_hours
    if metadata.usd is not None:
        metrics["usd"] = metadata.usd

    tags: dict[str, str] = {
        "tree_dirty": str(metadata.tree_dirty),
        "warnings": "; ".join(metadata.warnings),
    }

    return params, metrics, tags


class MlflowClientLike(Protocol):
    """The minimal subset of `mlflow`'s client surface a Wave 3 backend
    needs to implement `persist` for real. Named here (not `mlflow.client.
    MlflowClient` itself) so this module never imports `mlflow` — the repo's
    runtime dependency stays `pyyaml` alone until T-006 wires the remote."""

    def log_params(self, params: Mapping[str, str | int | float | bool]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...

    def set_tags(self, tags: Mapping[str, str]) -> None: ...


def read_git_commit_sha(repo_root: Path) -> str:
    """Runs `git rev-parse HEAD` in `repo_root` and returns the commit SHA,
    answering issue #23 row D9's "who reads git rev-parse" (previously
    `RunMetadata.git_commit_sha` was caller-supplied with no reader).

    Raises TrackingBackendError if `repo_root` is not a git repository or
    `git` is not on PATH — this is an I/O failure, not a RunMetadata
    validation error, so it is a distinct exception from RunMetadataError."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise TrackingBackendError("git executable not found on PATH") from exc
    if completed.returncode != 0:
        raise TrackingBackendError(
            f"git rev-parse HEAD failed in {repo_root}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()
