"""Training driver skeleton behind `bakeoff.run_bakeoff`'s `fine_tune` seam
(backlog 0009; #23 row D1; tech-spec v2 §5 stack, §10 ephemeral lifecycle).

`src/bakeoff/__init__.py` names `fine_tune` as an injected callable so
`run_bakeoff` never imports a concrete trainer (module docstring there); this
module is that concrete trainer's *skeleton*. It builds everything around the
real backend call — request resolution, the cloud_ok / nf4 gates, artifact
layout, run-metadata persistence, and the `make_fine_tune` adapter that
matches `run_bakeoff`'s `fine_tune` parameter exactly — with a `NullBackend`
that produces deterministic stub adapters/metrics, so the whole lifecycle
(driver -> run_job.sh -> tracking) is provable without a GPU (ADR 0007: local
machines never train). `UnslothBackend`/`PeftTrlBackend`/`AxolotlBackend`
import their framework lazily inside `train()` and raise
`BackendUnavailableError` when it is missing; their real call bodies are
`# backlog 0009: GPU integration` and are not exercised here (needs T-006
accounts and the container, backlog 0023).

No cross-sibling imports of Wave 3 siblings' new symbols (redteam/eval/coreset
tickets #42-#46 run in parallel on this repo's branch and do not exist here);
this module only reads `bakeoff` (`Candidate`, `TrainedArtifact`,
`load_bakeoff_config`), `tracking`/`tracking.backends`, `governance.store`
(`manifest_sha256`) and `data_contract` (`read_manifest`), exactly the
"Owned files" read-only import list in issue #41.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

import yaml

from bakeoff import Candidate, TrainedArtifact, load_bakeoff_config
from data_contract import DataContractError, TargetLanguage, read_manifest
from governance.store import manifest_sha256
from tracking import RunMetadata, new_run_id, record_run, to_hashable_config
from tracking.backends import LocalFileBackend, read_git_commit_sha
from train import Hyperparameters, assert_nf4_requires_stretch_arm, load_hyperparameters

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# TRAIN_CMD line format (this wave's "Shared formats fixed for this wave"
# section, W3-a defines / W3-g consumes): fixed field order, `$OUT_DIR` left
# literal so scripts/cloud/run_job.sh can expand it at its step 4. Kept as a
# module constant so the CLI and any test asserting the exact string share
# one source.
_TRAIN_CMD_TEMPLATE = (
    "uv run python scripts/train/driver.py "
    "--job {job} --candidate {candidate} --training-config {training_config} "
    "--manifest {manifest} --run-manifest {run_manifest} --run-id {run_id} "
    '--seed {seed} --backend {backend} --out-dir "$OUT_DIR"'
)

# Exit codes (issue #41 item 5): 0 success; 2 usage/config error; 4 cloud_ok
# refusal (mirrors run_job.sh's own step-3 exit code); 5 backend failure
# (mirrors run_job.sh's step-4 exit code for a training failure).
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CLOUD_OK_REFUSED = 4
EXIT_BACKEND_FAILURE = 5


class TrainDriverError(ValueError):
    """Raised for a resolution-time problem: malformed job YAML, an unknown
    candidate id, an nf4/stretch-arm violation, or a non-cloud_ok manifest
    row. Distinct from `BackendUnavailableError` (a backend-time problem) so
    a caller/CLI can map the two to different exit codes (2/4 vs 5)."""


class BackendUnavailableError(RuntimeError):
    """Raised by a real backend's `train()` when its training framework
    (Unsloth, PEFT+TRL, Axolotl) is not importable in the current
    environment. Names the missing module. Never raised by `NullBackend`."""


# --- Backend protocol + null/real backends (issue #41 item 1) ---------------


@dataclass(frozen=True, slots=True)
class TrainRequest:
    """Everything one `train()` call needs (issue #41 item 2). `candidate`
    and `hyperparameters` are the same typed values `run_bakeoff` threads
    through its `fine_tune` callable; `manifest_path` is the job's
    `DataItem`-shaped JSONL (the cloud_ok gate target); `run_manifest_path`
    is this wave's per-run coreset manifest JSON (`manifest_sha256` is read
    from it, never recomputed from `DataItem`s — the cross-ticket trap
    between W3-a and W3-e, this wave's shared-formats section)."""

    candidate: Candidate
    hyperparameters: Hyperparameters
    language: TargetLanguage
    split_id: str
    seed: int
    run_id: str
    manifest_path: Path
    run_manifest_path: Path
    out_dir: Path
    job_yaml_path: Path


@dataclass(frozen=True, slots=True)
class TrainOutcome:
    """What a backend's `train()` returns: the adapter directory it wrote,
    the metrics dict it recorded (backend-defined keys; `NullBackend` writes
    `train_loss`/`steps`/`backend`), and any warnings (e.g. a seed outside
    `hyperparameters.seeds`, issue #41 item 2) to carry into
    `RunMetadata.warnings`."""

    adapter_dir: Path
    metrics: dict[str, object]
    warnings: tuple[str, ...] = ()


class TrainingBackend(Protocol):
    """The seam every concrete trainer implements (issue #41 item 1):
    `name` identifies the backend in `RunMetadata`/`TRAIN_CMD --backend`;
    `train()` performs (or stubs) the actual fine-tuning and returns a
    `TrainOutcome`."""

    name: str

    def train(self, request: TrainRequest) -> TrainOutcome: ...


def _adapter_weights_bytes(digest_hex: str) -> bytes:
    """A small, deterministic stub adapter payload derived from the request
    digest (issue #41 item 1: "a small deterministic adapter_weights.bin
    derived from that digest so the adapter sha256 is pinnable"). Not real
    model weights — `NullBackend` never trains anything; this is bytes a
    test can pin a sha256 against."""
    return hashlib.sha256(f"null-backend-adapter:{digest_hex}".encode("utf-8")).digest()


def _request_digest(request: TrainRequest) -> str:
    """sha256 hex of the request's identifying fields, used to derive
    `NullBackend`'s stub adapter content — not `TrainedArtifact.
    hyperparameters_digest` (see `_hyperparameters_digest` below), though
    both are sha256 hex digests of a canonical JSON encoding."""
    payload = {
        "candidate_id": request.candidate.id,
        "hyperparameters_digest": _hyperparameters_digest(request.hyperparameters),
        "split_id": request.split_id,
        "seed": request.seed,
        "run_id": request.run_id,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hyperparameters_digest(hyperparameters: Hyperparameters) -> str:
    """sha256 hex of `json.dumps(dataclasses.asdict(hp), sort_keys=True,
    separators=(",", ":"))` (issue #41 item 3) — the formula `TrainedArtifact.
    hyperparameters_digest` uses. Verified against `src/bakeoff/__init__.py`
    and `tests/test_bakeoff.py` (`grep -n hyperparameters_digest
    src/bakeoff/__init__.py tests/test_bakeoff.py`): only the field and a
    `"digest-fixed"` stub exist there, no formula is fixed elsewhere, so this
    module is the first to define one."""
    payload = dataclasses.asdict(hyperparameters)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class NullBackend:
    """Deterministic stub backend (issue #41 item 1): writes an adapter
    directory (`adapter_config.json` carrying the request digest, plus
    `adapter_weights.bin` derived from that digest) and `metrics.json`
    (`train_loss: null, steps: 0, backend: "null"`). Used to prove the whole
    lifecycle — driver, run_job.sh, tracking — without a GPU (ADR 0007)."""

    name = "null"

    def train(self, request: TrainRequest) -> TrainOutcome:
        digest = _request_digest(request)
        adapter_dir = (
            request.out_dir
            / "adapters"
            / request.candidate.id
            / f"seed-{request.seed}"
        )
        adapter_dir.mkdir(parents=True, exist_ok=True)

        adapter_config = {
            "candidate_id": request.candidate.id,
            "seed": request.seed,
            "request_digest": digest,
            "backend": self.name,
        }
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (adapter_dir / "adapter_weights.bin").write_bytes(_adapter_weights_bytes(digest))

        metrics: dict[str, object] = {"train_loss": None, "steps": 0, "backend": self.name}
        (request.out_dir / "metrics.json").write_text(
            json.dumps(metrics, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

        warnings: tuple[str, ...] = ()
        if request.seed not in request.hyperparameters.seeds:
            warnings = (
                f"seed={request.seed} is outside hyperparameters.seeds="
                f"{request.hyperparameters.seeds} — allowed, but recorded "
                "(issue #41 item 2)",
            )

        return TrainOutcome(adapter_dir=adapter_dir, metrics=metrics, warnings=warnings)


def _real_backend_train(backend_name: str, module_name: str, request: TrainRequest) -> TrainOutcome:
    """Shared lazy-import guard for the three real backends (issue #41 item
    1: "import their framework inside train(), raise BackendUnavailableError
    naming the missing module; no top-level import of any of them")."""
    try:
        __import__(module_name)
    except ImportError as exc:
        raise BackendUnavailableError(
            f"{backend_name} backend requires {module_name!r}, which is not "
            f"installed in this environment (backlog 0009: GPU integration "
            f"needs the rented-GPU container, backlog 0023)"
        ) from exc
    # backlog 0009: GPU integration — the real training call body goes here,
    # once the imported module above is available. Unreachable in this repo's
    # own environment (stdlib + pyyaml only, ADR 0007) since the import above
    # always raises first.
    raise NotImplementedError(  # pragma: no cover - unreachable without the real framework
        f"{backend_name}.train() body is backlog 0009: GPU integration"
    )


class UnslothBackend:
    """tech-spec v2 §5's primary framework (Unsloth, CUDA, one rented 80 GB
    GPU). Lazily imports `unsloth`; its real call body is backlog 0009."""

    name = "unsloth"

    def train(self, request: TrainRequest) -> TrainOutcome:
        return _real_backend_train(self.name, "unsloth", request)


class PeftTrlBackend:
    """tech-spec v2 §5's first fallback (Hugging Face PEFT + TRL). Lazily
    imports `peft`; its real call body is backlog 0009."""

    name = "peft_trl"

    def train(self, request: TrainRequest) -> TrainOutcome:
        return _real_backend_train(self.name, "peft", request)


class AxolotlBackend:
    """tech-spec v2 §5's second fallback, for multi-GPU (out of scope this
    ticket). Lazily imports `axolotl`; its real call body is backlog 0009."""

    name = "axolotl"

    def train(self, request: TrainRequest) -> TrainOutcome:
        return _real_backend_train(self.name, "axolotl", request)


_BACKENDS: dict[str, Callable[[], TrainingBackend]] = {
    "null": NullBackend,
    "unsloth": UnslothBackend,
    "peft_trl": PeftTrlBackend,
    "axolotl": AxolotlBackend,
}


def backend_by_name(name: str) -> TrainingBackend:
    """Looks up a `TrainingBackend` by its `--backend`/`RunMetadata`-facing
    name. Raises `TrainDriverError` for an unknown name (a usage error, exit
    2), never a `KeyError`."""
    try:
        factory = _BACKENDS[name]
    except KeyError:
        raise TrainDriverError(
            f"unknown backend {name!r}; expected one of {sorted(_BACKENDS)}"
        ) from None
    return factory()


# --- Request resolution (issue #41 item 2) -----------------------------------


def _load_job_yaml(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrainDriverError(f"{path}: cannot read job YAML: {exc}") from exc
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise TrainDriverError(f"{path}: top level must be a mapping")
    return raw


def _assert_manifest_cloud_ok(manifest_path: Path) -> None:
    """Re-checks that every job-manifest row is `cloud_ok`, using
    `data_contract.read_manifest` (issue #41 item 2: "read the real
    modules... pick the reader that accepts the job-manifest row shape and
    say which"). The job YAML's `data.manifest` key (see
    `configs/cloud/jobs/text_lora_single.yml`) points at a
    `DataItem`-shaped JSONL (a real ingest manifest, one full `DataItem`
    per row, unlike `scripts/cloud/assert_cloud_ok.py`'s looser
    `{"item_id", "cloud_ok"}`-only row shape) — `read_manifest` is the
    reader whose input format matches this, since it round-trips real
    preprocess output (`DataItem(**row)` per line) rather than accepting an
    arbitrary partial dict. `cloud_ok` is a real, non-derived `DataItem`
    field (validated against its own derivation at construction time by
    `DataItem.__post_init__`), so re-reading it here re-checks the same
    fail-closed rule `assert_cloud_ok.py` enforces at the shell level,
    against the typed objects this driver already needs for other reasons.
    Raises `TrainDriverError` naming every non-cloud_ok item_id."""
    try:
        items = read_manifest(manifest_path)
    except DataContractError as exc:
        raise TrainDriverError(f"{manifest_path}: {exc}") from exc
    if not items:
        raise TrainDriverError(f"{manifest_path}: empty manifest (fail-closed)")
    not_ok = [item.item_id for item in items if not item.cloud_ok]
    if not_ok:
        raise TrainDriverError(
            f"{manifest_path}: {len(not_ok)} item(s) are not cloud_ok: {not_ok}"
        )


def resolve_train_request(
    *,
    job_yaml_path: Path,
    candidate_id: str,
    training_config_path: Path,
    manifest_path: Path,
    run_manifest_path: Path,
    run_id: str,
    seed: int,
    out_dir: Path,
) -> TrainRequest:
    """Resolves and validates everything `train()` needs (issue #41 item 2):
    loads the job YAML (shape checked structurally the same way
    `tests/test_cloud_templates.py::test_job_template_shape` does), the
    candidate via `bakeoff.load_bakeoff_config`, hyperparameters via
    `train.load_hyperparameters(path, language=...)`; enforces
    `assert_nf4_requires_stretch_arm`; re-checks the manifest is cloud_ok;
    records an out-of-`hp.seeds` seed as a warning rather than refusing it.

    Raises `TrainDriverError` for any resolution-time problem (malformed
    job YAML, unknown candidate id, the nf4/stretch-arm rule, or a
    non-cloud_ok manifest row)."""
    job = _load_job_yaml(job_yaml_path)

    config_section = job.get("config")
    if not isinstance(config_section, dict):
        raise TrainDriverError(f"{job_yaml_path}: missing 'config' section")
    language_raw = config_section.get("language")
    if language_raw not in ("frc", "lou"):
        raise TrainDriverError(
            f"{job_yaml_path}: config.language must be 'frc' or 'lou', got {language_raw!r}"
        )
    language = cast(TargetLanguage, language_raw)

    candidates_path_raw = config_section.get("candidates")
    if not isinstance(candidates_path_raw, str):
        raise TrainDriverError(f"{job_yaml_path}: config.candidates must be a string path")
    candidates_path = REPO_ROOT / candidates_path_raw
    try:
        loaded_bakeoff = load_bakeoff_config(candidates_path)
    except Exception as exc:  # noqa: BLE001 - re-raised as TrainDriverError below
        raise TrainDriverError(f"{candidates_path}: {exc}") from exc

    arms = {arm.id: arm for arm in loaded_bakeoff.config.arms_for(language)}
    candidate = arms.get(candidate_id)
    if candidate is None:
        raise TrainDriverError(
            f"{job_yaml_path}: candidate {candidate_id!r} not found for language "
            f"{language!r} in {candidates_path} (known ids: {sorted(arms)})"
        )

    try:
        loaded_training = load_hyperparameters(training_config_path, language=language)
    except Exception as exc:  # noqa: BLE001 - re-raised as TrainDriverError below
        raise TrainDriverError(f"{training_config_path}: {exc}") from exc
    hyperparameters = loaded_training.hyperparameters

    try:
        assert_nf4_requires_stretch_arm(
            hyperparameters.quantization_train, base_size_b=candidate.size_b
        )
    except Exception as exc:  # noqa: BLE001 - re-raised as TrainDriverError below
        raise TrainDriverError(str(exc)) from exc

    _assert_manifest_cloud_ok(manifest_path)

    split_id_raw = job.get("job")
    split_id = str(split_id_raw) if split_id_raw is not None else str(job_yaml_path.stem)

    return TrainRequest(
        candidate=candidate,
        hyperparameters=hyperparameters,
        language=language,
        split_id=split_id,
        seed=seed,
        run_id=run_id,
        manifest_path=manifest_path,
        run_manifest_path=run_manifest_path,
        out_dir=out_dir,
        job_yaml_path=job_yaml_path,
    )


# --- fine_tune seam adapter (issue #41 item 3) -------------------------------


def _run_metadata_for(
    request: TrainRequest,
    outcome: TrainOutcome,
    *,
    started_at: datetime,
    completed_at: datetime,
) -> RunMetadata:
    """Builds the post-train() `RunMetadata` (issue #41 item 4):
    `manifest_sha256` is read from the run-manifest file via
    `governance.store.manifest_sha256`, never recomputed from `DataItem`s
    (this wave's shared-formats trap between W3-a and W3-e);
    `git_commit_sha`/`tree_dirty` via `tracking.backends.read_git_commit_sha`
    (see `test_tracking.py`'s convention: `tree_dirty` from `git status
    --porcelain`); `image_digest` from the `IMAGE_DIGEST` env var if set;
    `gpu_hours`/`usd`/`instance` are `None` (cost accounting is the
    benchmark's, per utils-spec benchmark v2 §4)."""
    import os
    import subprocess

    git_commit_sha = read_git_commit_sha(REPO_ROOT)
    dirty_check = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tree_dirty = bool(dirty_check.stdout.strip())

    resolved_manifest_sha256 = manifest_sha256(request.run_manifest_path)

    return RunMetadata(
        run_id=request.run_id,
        stage="train",
        language=request.language,
        config=to_hashable_config(dataclasses.asdict(request.hyperparameters)),
        git_commit_sha=git_commit_sha,
        started_at=started_at,
        completed_at=completed_at,
        artifact_refs=(str(outcome.adapter_dir),),
        seed=request.seed,
        split_id=request.split_id,
        lock_hash=_hyperparameters_digest(request.hyperparameters),
        tree_dirty=tree_dirty,
        warnings=outcome.warnings,
        image_digest=os.environ.get("IMAGE_DIGEST"),
        manifest_sha256=resolved_manifest_sha256,
        gpu_hours=None,
        usd=None,
        instance=None,
    )


def run_training(backend: TrainingBackend, request: TrainRequest) -> TrainedArtifact:
    """Calls `backend.train(request)`, persists the resulting `RunMetadata`
    via `record_run(..., persist=LocalFileBackend(out_dir).persist)` (issue
    #41 item 4), and returns the `TrainedArtifact` `run_bakeoff` expects.
    Never writes `out_dir/run_metadata.json` — that is `run_job.sh`'s own
    sidecar (this wave's shared-formats section)."""
    started_at = datetime.now(timezone.utc)
    outcome = backend.train(request)
    completed_at = datetime.now(timezone.utc)

    metadata = _run_metadata_for(request, outcome, started_at=started_at, completed_at=completed_at)
    local_backend = LocalFileBackend(request.out_dir)
    record_run(metadata, persist=local_backend.persist)

    return TrainedArtifact(
        candidate_id=request.candidate.id,
        adapter_ref=str(outcome.adapter_dir),
        run_id=request.run_id,
        hyperparameters_digest=_hyperparameters_digest(request.hyperparameters),
        seed=request.seed,
    )


def make_fine_tune(
    backend: TrainingBackend,
    *,
    language: TargetLanguage,
    manifest_path: Path,
    run_manifest_path: Path,
    out_dir: Path,
    job_yaml: Path,
    run_id: str | None = None,
) -> Callable[[Candidate, Hyperparameters, str, int], TrainedArtifact]:
    """Returns a callable matching `run_bakeoff`'s `fine_tune` parameter
    exactly (issue #41 item 3): `Callable[[Candidate, Hyperparameters, str,
    int], TrainedArtifact]` — verified against
    `inspect.signature(bakeoff.run_bakeoff)`'s real `fine_tune:
    Callable[[Candidate, H, str, int], TrainedArtifact]` (candidate,
    hyperparameters, split_id, seed; backlog 0009's stale three-argument
    text is superseded by this signature). A caller-pinned `run_id` (via
    `--run-id`) is used when given, so one run_id joins
    `TrainedArtifact.run_id`, the run manifest's own `run_id`, and
    `RunMetadata.run_id` (this wave's `run_id` join shared format);
    otherwise `tracking.new_run_id()` mints a fresh one, the project's one
    generator."""
    resolved_run_id = run_id if run_id is not None else new_run_id()

    def _fine_tune(
        candidate: Candidate,
        hyperparameters: Hyperparameters,
        split_id: str,
        seed: int,
    ) -> TrainedArtifact:
        request = TrainRequest(
            candidate=candidate,
            hyperparameters=hyperparameters,
            language=language,
            split_id=split_id,
            seed=seed,
            run_id=resolved_run_id,
            manifest_path=manifest_path,
            run_manifest_path=run_manifest_path,
            out_dir=out_dir,
            job_yaml_path=job_yaml,
        )
        return run_training(backend, request)

    return _fine_tune


# --- CLI (issue #41 item 5) ---------------------------------------------------


def build_train_cmd_line(
    *,
    job: Path,
    candidate: str,
    training_config: Path,
    manifest: Path,
    run_manifest: Path,
    run_id: str,
    seed: int,
    backend: str,
) -> str:
    """The exact `TRAIN_CMD=<single-quoted command>` line `--dry-run` prints
    (this wave's shared-formats section, TRAIN_CMD; W3-g consumes it in
    phase 2). `$OUT_DIR` is left literal for `run_job.sh` to expand."""
    inner = _TRAIN_CMD_TEMPLATE.format(
        job=job,
        candidate=candidate,
        training_config=training_config,
        manifest=manifest,
        run_manifest=run_manifest,
        run_id=run_id,
        seed=seed,
        backend=backend,
    )
    return f"TRAIN_CMD='{inner}'"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--job", required=True, type=Path, help="job YAML path")
    parser.add_argument("--candidate", required=True, help="bake-off candidate id")
    parser.add_argument("--training-config", required=True, type=Path, help="lora_defaults.yml-shaped path")
    parser.add_argument("--manifest", required=True, type=Path, help="DataItem-shaped JSONL manifest")
    parser.add_argument(
        "--run-manifest", required=True, type=Path, help="per-run coreset manifest JSON"
    )
    parser.add_argument("--run-id", default=None, help="pin a run_id (default: mint a fresh one)")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--backend", default="null", choices=sorted(_BACKENDS), help="training backend to use"
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="output directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the resolved plan + TRAIN_CMD line; write nothing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    run_id = args.run_id if args.run_id is not None else new_run_id()

    try:
        request = resolve_train_request(
            job_yaml_path=args.job,
            candidate_id=args.candidate,
            training_config_path=args.training_config,
            manifest_path=args.manifest,
            run_manifest_path=args.run_manifest,
            run_id=run_id,
            seed=args.seed,
            out_dir=args.out_dir,
        )
    except TrainDriverError as exc:
        message = str(exc)
        if "not cloud_ok" in message or "empty manifest" in message:
            print(f"train driver: cloud_ok refusal: {exc}", file=sys.stderr)
            return EXIT_CLOUD_OK_REFUSED
        print(f"train driver: config error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    train_cmd_line = build_train_cmd_line(
        job=args.job,
        candidate=args.candidate,
        training_config=args.training_config,
        manifest=args.manifest,
        run_manifest=args.run_manifest,
        run_id=run_id,
        seed=args.seed,
        backend=args.backend,
    )

    if args.dry_run:
        print("train driver: resolved plan (dry run, nothing written)")
        print(f"  candidate:               {request.candidate.id}")
        print(f"  hyperparameters_digest:  {_hyperparameters_digest(request.hyperparameters)}")
        print(f"  seed:                    {request.seed}")
        print(f"  quantization_train:      {request.hyperparameters.quantization_train}")
        print(f"  rank/alpha:              {request.hyperparameters.rank}/{request.hyperparameters.alpha}")
        print(f"  backend:                 {args.backend}")
        print(f"  out_dir:                 {request.out_dir}")
        print(train_cmd_line)
        return EXIT_OK

    backend = backend_by_name(args.backend)
    try:
        run_training(backend, request)
    except BackendUnavailableError as exc:
        print(f"train driver: backend unavailable: {exc}", file=sys.stderr)
        return EXIT_BACKEND_FAILURE
    except Exception as exc:  # noqa: BLE001 - any other backend failure is exit 5
        print(f"train driver: training failed: {exc}", file=sys.stderr)
        return EXIT_BACKEND_FAILURE

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
