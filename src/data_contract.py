"""Pipeline ingestion gate: item schema + eligibility pre-filter.

Every ingested item — text or (later) audio — must conform to DataItem
before it can enter any downstream pipeline stage (language-ID tagging,
orthographic normalization, coreset selection, synthetic augmentation,
training, benchmarking). is_eligible() is the single, pure eligibility
verdict every one of those stages relies on instead of re-implementing
rights/consent logic itself.

Pure and side-effect-free: no file, network, or database access. Callers
(the preprocess utility, the coreset selector, etc.) own all I/O and are
responsible for routing ineligible items to an inventory-only sink.

Schema field sourcing: dev/tech-spec_fine-tune-cajun_v1_20260831.md §2, plus
the `rights`/`training_permission` reconciliation against the binding
linguist-handoff source described in GitHub issue #1's Implementation
Decisions (the tech-spec's own §2 schema block under-specifies `rights` and
omits `training_permission` as a field; that source's v2 correction
promotes `training_permission` to first-class, never derived on the fly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

# language_tag: the closed LID taxonomy. CONTEXT.md "Language-ID taxonomy";
# tech-spec §2. No value outside this set is valid — an item's variety must
# never be collapsed or guessed beyond these seven tags.
LanguageTag = Literal["lou", "frc", "fra", "hat", "eng", "mixed", "unknown"]

# orthography_system: tech-spec §2. `ad_hoc` is the deliberate, code-enforced
# fallback for frc items pending a ratified orthography convention (ARD §12)
# — a valid non-error state, not a rejection.
OrthographySystem = Literal["KVO", "French-like", "ad_hoc", "English-phonetic", "mixed"]

# consent_tier: tech-spec §2; CONTEXT.md "Consent tier". Travels with the
# item from ingestion through training; a pipeline stage must never use data
# above the tier it was granted.
ConsentTier = Literal["display", "research", "model-eval", "training", "commercial-prohibited", "withdrawal"]

# rights: the linguist-handoff data-contract source's closed enum (not the
# tech-spec's looser "license id" free text) — see module docstring.
Rights = Literal["public_domain", "cc_open", "cc_restricted", "archive_permission", "rights_unknown", "all_rights_reserved"]

# training_permission: first-class per the linguist-handoff source's v2
# correction — recorded explicitly on every item, never derived from other
# fields. `uncertain` is treated as `no` by is_eligible (fail-safe).
TrainingPermission = Literal["yes_general", "yes_scoped", "no", "uncertain"]

# cultural_sensitivity: tech-spec §2. `restricted`/`sacred` items must never
# reach an eligible verdict via a coding mistake elsewhere in the pipeline.
# tech-spec §2's eligibility condition is specifically
# `community_review-with-signoff`, a sub-state of `community_review` — see
# `community_review_signed_off` below, which the enum alone can't represent.
CulturalSensitivity = Literal["open", "community_review", "restricted", "sacred"]

# release_class: tech-spec §2.
ReleaseClass = Literal["public", "gated", "do_not_use"]

_LANGUAGE_TAGS = frozenset(get_args(LanguageTag))
_ORTHOGRAPHY_SYSTEMS = frozenset(get_args(OrthographySystem))
_CONSENT_TIERS = frozenset(get_args(ConsentTier))
_RIGHTS = frozenset(get_args(Rights))
_TRAINING_PERMISSIONS = frozenset(get_args(TrainingPermission))
_CULTURAL_SENSITIVITIES = frozenset(get_args(CulturalSensitivity))
_RELEASE_CLASSES = frozenset(get_args(ReleaseClass))

# Eligible rights: everything except the two "not provably cleared" values.
_ELIGIBLE_RIGHTS = frozenset({"public_domain", "cc_open", "cc_restricted", "archive_permission"})
# Eligible training permissions: `uncertain` is fail-safe-excluded, not just `no`.
_ELIGIBLE_TRAINING_PERMISSIONS = frozenset({"yes_general", "yes_scoped"})


class DataContractError(ValueError):
    """Raised when a DataItem is constructed with an invalid field value."""


@dataclass(frozen=True, slots=True)
class DataItem:
    """The minimum per-item record every ingested asset must carry (tech-spec §2)."""

    item_id: str
    source: str
    language_tag: LanguageTag
    lect: str | None
    # genre: free text, optional — same shape as lect. Named only in
    # tech-spec §8's coreset stratification prose ("language_tag/lect/
    # genre"), not in §2's own schema block; no source names a closed genre
    # taxonomy, so this follows lect's precedent rather than inventing one.
    genre: str | None
    orthography_system: OrthographySystem
    consent_tier: ConsentTier
    rights: Rights
    training_permission: TrainingPermission
    cultural_sensitivity: CulturalSensitivity
    # Meaningful only when cultural_sensitivity == "community_review"; the
    # tech-spec §2 eligibility condition is "community_review-with-signoff"
    # specifically, not bare community_review. Ignored for open/restricted/
    # sacred, where it carries no eligibility meaning either way.
    community_review_signed_off: bool
    release_class: ReleaseClass
    synthetic: bool
    generator: str | None
    provenance: str
    schema_version: str

    def __post_init__(self) -> None:
        # Hard-required, non-nullable fields (tech-spec §2 explicit invariant):
        # rejected at ingest, never silently defaulted.
        if not self.item_id:
            raise DataContractError("item_id is required and must be non-empty")
        if not self.language_tag:
            raise DataContractError("language_tag is required and must be non-empty")
        if not self.orthography_system:
            raise DataContractError("orthography_system is required and must be non-empty")

        _require_member(self.language_tag, _LANGUAGE_TAGS, "language_tag")
        _require_member(self.orthography_system, _ORTHOGRAPHY_SYSTEMS, "orthography_system")
        _require_member(self.consent_tier, _CONSENT_TIERS, "consent_tier")
        _require_member(self.rights, _RIGHTS, "rights")
        _require_member(self.training_permission, _TRAINING_PERMISSIONS, "training_permission")
        _require_member(self.cultural_sensitivity, _CULTURAL_SENSITIVITIES, "cultural_sensitivity")
        _require_member(self.release_class, _RELEASE_CLASSES, "release_class")

        # synthetic=true items must be traceable to their generator (tech-spec
        # §6.4 gold/silver/synthetic discipline; CONTEXT.md "Gold / silver /
        # synthetic").
        if self.synthetic and not self.generator:
            raise DataContractError("generator is required when synthetic=True")


def _require_member(value: str, allowed: frozenset[str], field_name: str) -> None:
    if value not in allowed:
        raise DataContractError(f"{field_name}={value!r} is not one of {sorted(allowed)}")


EligibilityReason = Literal[
    "rights_not_cleared",
    "training_permission_not_granted",
    "cultural_sensitivity_restricted",
    "release_class_do_not_use",
]


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    """The verdict is_eligible() returns: whether an item may proceed, and why not."""

    eligible: bool
    reasons: tuple[EligibilityReason, ...]


def is_eligible(item: DataItem) -> EligibilityResult:
    """Decide whether `item` may proceed past ingestion into annotation or training selection.

    Pure function: no I/O, no side effects, deterministic. Reflects
    tech-spec §2's eligibility pre-filter, reconciled against the binding
    linguist-handoff source's rights/training_permission enums (see module
    docstring). Ineligible items are not this function's concern to route
    anywhere — that's the caller's job (tech-spec's inventory-only sink).
    """
    reasons: list[EligibilityReason] = []

    if item.rights not in _ELIGIBLE_RIGHTS:
        reasons.append("rights_not_cleared")

    if item.training_permission not in _ELIGIBLE_TRAINING_PERMISSIONS:
        reasons.append("training_permission_not_granted")

    cultural_sensitivity_ok = item.cultural_sensitivity == "open" or (
        item.cultural_sensitivity == "community_review" and item.community_review_signed_off
    )
    if not cultural_sensitivity_ok:
        reasons.append("cultural_sensitivity_restricted")

    if item.release_class == "do_not_use":
        reasons.append("release_class_do_not_use")

    return EligibilityResult(eligible=not reasons, reasons=tuple(reasons))
