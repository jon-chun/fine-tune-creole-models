"""Pipeline ingestion gate: item schema + eligibility pre-filter (contract v2).

Every ingested item — text or (later) audio — must conform to DataItem
before it can enter any downstream pipeline stage (language-ID tagging,
orthographic normalization, coreset selection, synthetic augmentation,
training, benchmarking). is_eligible() is the single, pure eligibility
verdict every one of those stages relies on instead of re-implementing
rights/consent logic itself.

Pure and side-effect-free: no file, network, or database access. Callers
(the preprocess utility, the coreset selector, etc.) own all I/O and are
responsible for routing ineligible items to an inventory-only sink.
`read_manifest` below is the one deliberate exception (issue #18): it reads
a preprocess-written manifest.jsonl and is the file-I/O counterpart callers
use to turn that output back into `DataItem` records.

Schema field sourcing: dev/tech-spec_fine-tune-cajun_v2_fable51max_20260902.md
§2 (annotation schema v2 / data contract v2), ratified for this repo by ADR
0015 (MIG-01a, issue #25). Supersedes the v1 hybrid schema (schema_version
"1.1.0") that this module implemented through HEAD 94cef88.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Literal, get_args

# --- Closed enums (tech-spec v2 §2.1) ---------------------------------------

# language_tag: the closed LID taxonomy, v2 (CONTEXT.md "Language-ID
# taxonomy"; tech-spec v2 §2.1). Adds `spa` to the v1 seven-tag set. No value
# outside this set is valid — an item's variety must never be collapsed or
# guessed beyond these eight tags.
LanguageTag = Literal["lou", "frc", "fra", "hat", "eng", "spa", "mixed", "unknown"]

# eng_dialect: sub-tag carried only when language_tag == "eng" (tech-spec v2
# §2.1, §3.1). `None` for every non-`eng` item.
EngDialect = Literal["aae", "other", "unknown"]

# record_type: tech-spec v2 §2.1. New in v2; no v1 equivalent.
RecordType = Literal["text", "audio_segment", "pair", "hitl_turn"]

# orthography_system: tech-spec v2 §2.1; CONTEXT.md "Reversible orthographic
# normalization" (ADR 0011). ONE unified enum spanning both languages — the
# first four values are `lou`-valid, the next five `frc`-valid, plus the two
# shared fallbacks. Lowercase spellings only; v1 code spellings (`KVO`,
# `French-like`, `English-phonetic`) and annex spellings (`DLF-normalized`,
# `Faulk`, `Daigle`) are normalized to these via `_ALIASES`, never accepted
# directly.
OrthographySystem = Literal[
    "kvo",
    "french_like",
    "english_phonetic",
    "ad_hoc",
    "dlf_normalized",
    "faulk",
    "daigle",
    "house",
    "diplomatic",
    "mixed",
    "unknown",
]

# genre: tech-spec v2 §2.1, referencing annotation schema v2 §5. Closed enum,
# replacing v1's free-text field.
Genre = Literal[
    "conversation",
    "interview",
    "narrative_folktale",
    "song_performance",
    "sermon_oratory",
    "lesson_pedagogical",
    "formulaic",
    "other",
]

# register: tech-spec v2 §2.1, annotation schema v2 §5. New in v2; no v1
# equivalent (the v1 module had no register field at all).
Register = Literal["casual", "careful", "formal", "performance", "ritual", "pedagogical", "unknown"]

# rights: the linguist-handoff data-contract source's closed enum (unchanged
# from v1; tech-spec v2 §2.1 keeps this set verbatim).
Rights = Literal["public_domain", "cc_open", "cc_restricted", "archive_permission", "rights_unknown", "all_rights_reserved"]

# consent: tech-spec v2 §2.1; CONTEXT.md "`consent`" (ADR 0009). Replaces v1
# `ConsentTier`'s six display/research/model-eval/training/commercial-
# prohibited/withdrawal set with the v2 five-value grant-state enum. Travels
# with the item from ingestion through training; a pipeline stage must never
# use data beyond the grant this field records.
Consent = Literal[
    "informed_consent_training",
    "informed_consent_research",
    "legacy_no_consent",
    "consent_pending",
    "consent_withdrawn",
]

# training_permission: first-class per the linguist-handoff source's v2
# correction — recorded explicitly on every item, never derived from other
# fields (unchanged from v1). `uncertain` is treated as `no` by is_eligible
# (fail-safe).
TrainingPermission = Literal["yes_general", "yes_scoped", "no", "uncertain"]

# cultural_sensitivity: tech-spec v2 §2.1. Drops v1's `sacred` value (CONTEXT.md
# "Language-ID taxonomy" area; ADR 0009) — `restricted` covers what `sacred`
# covered under the v2 enum. `restricted`/`consent_pending` items must never
# reach an eligible verdict via a coding mistake elsewhere in the pipeline.
CulturalSensitivity = Literal["open", "community_review", "restricted", "consent_pending"]

# sensitivity_tier / access_tier / object_tier: CONTEXT.md "Tier axes" — three
# orthogonal fields, new in v2, that must never be conflated with `data_class`
# or each other. tech-spec v2 §2.1.
SensitivityTier = Literal["S0", "S1", "S2", "S3"]
AccessTier = Literal[1, 2, 3, 4]
ObjectTier = Literal["T0", "T1", "T2", "T3", "T4", "T5"]

# release_class: tech-spec v2 §2.1/§2.2/§1 (contract v2 §1); CONTEXT.md
# "`release_class`". Six computed values, replacing v1's three hand-set
# values (`public | gated | do_not_use`). Never a constructor argument the
# caller free-sets — see `derive_release_class` below.
ReleaseClass = Literal[
    "public_train_ok",
    "public_eval_only",
    "internal_eval_only",
    "research_agreement",
    "streaming_only",
    "do_not_use",
]

# speaker_generation / speaker_role / gender / attribution_mode / pii_status /
# reading_type / split: new v2 speaker- and split-level fields (tech-spec v2
# §2.1). No v1 equivalents.
SpeakerGeneration = Literal["elder_fluent", "heritage", "learner_revitalization", "unknown"]
SpeakerRole = Literal["interviewer", "interviewee", "performer", "narrator", "other", "unknown"]
Gender = Literal["f", "m", "other_unknown"]
AttributionMode = Literal["named", "pseudonym", "anonymous"]
PiiStatus = Literal["none", "tagged", "redacted"]
ReadingType = Literal["read", "spontaneous", "wordlist", "control"]
Split = Literal["gold_eval", "gold_train", "silver_unreviewed"]

# data_class: CONTEXT.md "`data_class`" (ADR 0009); tech-spec v2 §2.1/§2.4.
# Renames v1's `Tier = Literal["gold","silver","synthetic"]` and adds
# `bronze`. Hard rename, no compatibility alias (issue #25 decision:
# `DataItem(tier=...)` is a TypeError, not a deprecated kwarg) — the single
# owner of the gold/silver/bronze/synthetic partition.
DataClass = Literal["gold", "silver", "bronze", "synthetic"]

# normalizer_status: CONTEXT.md "Reversible orthographic normalization"; v2
# tech-spec §2.1 adds `partial` to v1's `not_ready | ready` pair.
NormalizerStatus = Literal["ready", "not_ready", "partial"]

# normalization_difficulty / diff_catalog_flags: new v2 fields (tech-spec v2
# §2.1). `diff_catalog_flags` carries §4 cell ids; `list[str]`, not a closed
# enum.
NormalizationDifficulty = Literal["low", "medium", "high"]

# target_language: pre-agreed addition for issue #14 (base-model/transfer
# bake-off target selection). Definition only — no consumer yet. Unchanged
# from v1.
TargetLanguage = Literal["frc", "lou"]

# ModelReleaseClass: CONTEXT.md "Model release class". Defined here as the
# shared home per MIG-01g's needs (tech-spec v2 §2.4, issue #25 "Out of
# scope"); distinct from the per-item `ReleaseClass` above, which governs
# data, not models. Not consumed by this module — wiring is MIG-01c
# (`CandidateResult.release_class`) and MIG-01e (`AcceptanceReport.release_class`).
ModelReleaseClass = Literal["release_ready", "research_only", "internal_only", "withdrawn"]

_LANGUAGE_TAGS = frozenset(get_args(LanguageTag))
_ENG_DIALECTS = frozenset(get_args(EngDialect))
_RECORD_TYPES = frozenset(get_args(RecordType))
_ORTHOGRAPHY_SYSTEMS = frozenset(get_args(OrthographySystem))
_GENRES = frozenset(get_args(Genre))
_REGISTERS = frozenset(get_args(Register))
_RIGHTS = frozenset(get_args(Rights))
_CONSENTS = frozenset(get_args(Consent))
_TRAINING_PERMISSIONS = frozenset(get_args(TrainingPermission))
_CULTURAL_SENSITIVITIES = frozenset(get_args(CulturalSensitivity))
_SENSITIVITY_TIERS = frozenset(get_args(SensitivityTier))
_ACCESS_TIERS = frozenset(get_args(AccessTier))
_OBJECT_TIERS = frozenset(get_args(ObjectTier))
_RELEASE_CLASSES = frozenset(get_args(ReleaseClass))
_SPEAKER_GENERATIONS = frozenset(get_args(SpeakerGeneration))
_SPEAKER_ROLES = frozenset(get_args(SpeakerRole))
_GENDERS = frozenset(get_args(Gender))
_ATTRIBUTION_MODES = frozenset(get_args(AttributionMode))
_PII_STATUSES = frozenset(get_args(PiiStatus))
_READING_TYPES = frozenset(get_args(ReadingType))
_SPLITS = frozenset(get_args(Split))
_DATA_CLASSES = frozenset(get_args(DataClass))
_NORMALIZER_STATUSES = frozenset(get_args(NormalizerStatus))
_NORMALIZATION_DIFFICULTIES = frozenset(get_args(NormalizationDifficulty))
_TARGET_LANGUAGES = frozenset(get_args(TargetLanguage))
_MODEL_RELEASE_CLASSES = frozenset(get_args(ModelReleaseClass))

# Eligible rights: everything except the two "not provably cleared" values.
_ELIGIBLE_RIGHTS = frozenset({"public_domain", "cc_open", "cc_restricted", "archive_permission"})
# Eligible training permissions: `uncertain` is fail-safe-excluded, not just `no`.
_ELIGIBLE_TRAINING_PERMISSIONS = frozenset({"yes_general", "yes_scoped"})
# Consent values that never grant training under any circumstance (tech-spec
# v2 §2.2's consent clause; CONTEXT.md "`consent`"). `legacy_no_consent`
# trains only conditionally (with community_review_signed_off — the Charter
# review carve-out), so it is checked separately in is_eligible rather than
# folded into this set.
_NEVER_TRAINING_CONSENTS = frozenset({"consent_pending", "consent_withdrawn"})

SCHEMA_VERSION: Literal["2.0.0"] = "2.0.0"

# --- Alias normalization (tech-spec v2 §2.1) --------------------------------

# _ALIASES: module-level table keyed by field name, applied by
# `_normalize_aliases` on every construction path (both direct `DataItem(...)`
# construction and `read_manifest`), so v1/annex spellings never need
# special-casing at every call site (issue #25 story 4).
_ALIASES: dict[str, dict[str, str]] = {
    "language_tag": {
        "lf": "frc",
    },
    "speaker_generation": {
        "elder_L1": "elder_fluent",
        "new_speaker": "learner_revitalization",
    },
    "orthography_system": {
        "DLF-normalized": "dlf_normalized",
        "Faulk": "faulk",
        "Daigle": "daigle",
        "KVO": "kvo",
        "French-like": "french_like",
        "English-phonetic": "english_phonetic",
    },
}


def _normalize_aliases(raw: dict[str, object]) -> dict[str, object]:
    """Apply `_ALIASES` to `raw`'s values, field by field.

    Called before validation on every construction path so a caller may pass
    either the canonical spelling or a known v1/annex alias and the stored
    value is always canonical. Leaves fields/values it has no alias entry for
    untouched; does not know about or validate closed-enum membership itself
    (that is `__post_init__`'s job, and `is_eligible`'s callers rely on it
    running strictly after this normalization).
    """
    normalized = dict(raw)
    for field_name, field_aliases in _ALIASES.items():
        if field_name in normalized:
            value = normalized[field_name]
            if isinstance(value, str) and value in field_aliases:
                normalized[field_name] = field_aliases[value]
    return normalized


class DataContractError(ValueError):
    """Raised when a DataItem is constructed with an invalid field value."""


def validate_literal(value: object, allowed: tuple[str, ...], field_name: str) -> str:
    """Reject `value` unless it is a str member of `allowed`; return it
    otherwise (narrowed to str, for callers that want the value back).

    The one shared enum-membership check every module's `__post_init__`
    should call instead of re-implementing its own (issue #15: every
    loader). `_require_member` below is this function's private
    frozenset-based predecessor, kept as a thin wrapper over it so existing
    call sites here don't need to change shape.
    """
    if not isinstance(value, str) or value not in allowed:
        raise DataContractError(f"{field_name}={value!r} is not one of {sorted(allowed)}")
    return value


def _require_member(value: object, allowed: frozenset[object], field_name: str) -> None:
    if value not in allowed:
        raise DataContractError(f"{field_name}={value!r} is not one of {sorted(allowed, key=str)}")


# --- release_class / cloud_ok derivation (contract v2 §1; tech-spec v2 §2.3) -


@dataclass(frozen=True, slots=True)
class ReleaseClassInputs:
    """The small field subset `derive_release_class` needs.

    `release_class` is itself a `DataItem` field and must not require itself
    as input, so callers building a `DataItem` compute `release_class` from
    this narrower input first (issue #25 "Proposed resolution").
    """

    rights: Rights
    training_permission: TrainingPermission
    consent: Consent
    cultural_sensitivity: CulturalSensitivity
    community_review_signed_off: bool


def derive_release_class(inputs: ReleaseClassInputs) -> ReleaseClass:
    """Pure derivation of `release_class` from rights/consent/sensitivity
    (contract v2 §1; tech-spec v2 §2.2). Evaluation order is most-restrictive-
    first: `do_not_use` is checked before any of the "better" classes, so it
    always wins a conflict (tech-spec v2 §2.2: "do_not_use wins any
    conflict"). Anything not provably a better class is `do_not_use` — the
    fail-safe default.
    """
    do_not_use = (
        inputs.rights in {"rights_unknown", "all_rights_reserved"}
        or inputs.consent in {"consent_pending", "consent_withdrawn"}
        or (inputs.consent == "legacy_no_consent" and not inputs.community_review_signed_off)
        or inputs.cultural_sensitivity in {"restricted", "consent_pending"}
    )
    if do_not_use:
        return "do_not_use"

    if (
        inputs.rights in {"public_domain", "cc_open"}
        and inputs.training_permission == "yes_general"
        and inputs.consent == "informed_consent_training"
        and inputs.cultural_sensitivity == "open"
    ):
        return "public_train_ok"

    # public_eval_only: as public_train_ok but training_permission =
    # yes_scoped (evaluation scope), or rights = cc_restricted (cleared).
    if (
        inputs.rights in {"public_domain", "cc_open"}
        and inputs.training_permission == "yes_scoped"
        and inputs.consent == "informed_consent_training"
        and inputs.cultural_sensitivity == "open"
    ) or (
        inputs.rights == "cc_restricted"
        and inputs.training_permission in {"yes_general", "yes_scoped"}
        and inputs.consent == "informed_consent_training"
        and inputs.cultural_sensitivity == "open"
    ):
        return "public_eval_only"

    # internal_eval_only = rights = archive_permission OR consent =
    # informed_consent_research OR a signed-off community_review item OR a
    # Charter-reviewed legacy_no_consent item. Neither signed-off case is
    # provably `open`/`informed_consent_training`, so neither reaches
    # public_*, but the do_not_use branch above only excludes the
    # *unreviewed* form of each (`community_review` without signoff is
    # `cultural_sensitivity ∈ {restricted, consent_pending}`'s sibling gap;
    # `legacy_no_consent (unreviewed)` is named explicitly in the do_not_use
    # formula) — so the reviewed form must land somewhere better, consistent
    # with is_eligible's retained consent clause treating a Charter-reviewed
    # legacy_no_consent item as consent_ok (tech-spec v2 §2.2; exact
    # cross-class precedence remains Open Q10).
    if (
        inputs.rights == "archive_permission"
        or inputs.consent == "informed_consent_research"
        or (inputs.cultural_sensitivity == "community_review" and inputs.community_review_signed_off)
        or (inputs.consent == "legacy_no_consent" and inputs.community_review_signed_off)
    ):
        return "internal_eval_only"

    # research_agreement / streaming_only: release only under a signed
    # research agreement, or display/listen only respectively. Neither has a
    # positive rule expressible from the fields this function takes (the
    # tech-spec names them as distinct release paths gated by scoping this
    # function's inputs cannot see — e.g. an out-of-band signed agreement).
    # Nothing provably better falls through to do_not_use below, matching
    # the fail-safe default (tech-spec v2 §2.2; Open Q10 leaves the exact
    # precedence a linguist-ratification item).

    return "do_not_use"


def derive_cloud_ok(
    *,
    release_class: ReleaseClass,
    training_permission: TrainingPermission,
    sensitivity_tier: SensitivityTier,
    pii_status: PiiStatus,
) -> bool:
    """Pure derivation of `cloud_ok` (tech-spec v2 §2.3; ADR 0007).

    The single owner of this rule: both the preprocess utility (MIG-01b) and
    the future 0.3 `hitl-export` path call this same function — no other
    module may re-derive it (tech-spec v2 §2.3: "It must be ONE pure
    function ... called by preprocess and by the 0.3 hitl-export").
    """
    if release_class not in {"public_train_ok", "public_eval_only"}:
        return False
    if training_permission != "yes_general":
        return False
    if sensitivity_tier not in {"S0", "S1"}:
        return False
    if sensitivity_tier == "S1" and pii_status == "tagged":
        return False
    return True


@dataclass(frozen=True, slots=True)
class DataItem:
    """The minimum per-item record every ingested asset must carry (tech-spec
    v2 §2.1, the annex Appendix B manifest)."""

    item_id: str
    source: str
    record_type: RecordType
    language_tag: LanguageTag
    eng_dialect: EngDialect | None
    lect: str | None
    orthography_system: OrthographySystem
    genre: Genre
    register: Register
    rights: Rights
    consent: Consent
    training_permission: TrainingPermission
    cultural_sensitivity: CulturalSensitivity
    # Meaningful only when cultural_sensitivity == "community_review"; the
    # tech-spec v2 §2.2 eligibility condition is "community_review-with-
    # signoff" specifically, not bare community_review.
    community_review_signed_off: bool
    sensitivity_tier: SensitivityTier
    access_tier: AccessTier
    object_tier: ObjectTier
    # release_class: computed by derive_release_class, never hand-set.
    # __post_init__ recomputes it from this item's own fields and raises if
    # the caller-supplied value disagrees (issue #25: "a caller passing
    # values that disagree with the derivation must get a DataContractError
    # naming the field").
    release_class: ReleaseClass
    speaker_id: str | None
    speaker_generation: SpeakerGeneration
    speaker_role: SpeakerRole
    gender: Gender
    attribution_mode: AttributionMode
    pii_status: PiiStatus
    reading_type: ReadingType | None
    passage_id: str | None
    pair_id: str | None
    split: Split
    # data_class: the single owner of the gold/silver/bronze/synthetic
    # partition (renames v1's `tier`; hard rename, no alias — issue #25
    # decision, story 6).
    data_class: DataClass
    synthetic: bool
    generator: str | None
    provenance: str
    normalizer_status: NormalizerStatus
    normalization_difficulty: NormalizationDifficulty
    diff_catalog_flags: list[str]
    # cloud_ok: computed by derive_cloud_ok, never hand-set. Same disagreement
    # check as release_class.
    cloud_ok: bool
    schema_version: Literal["2.0.0"]

    def __post_init__(self) -> None:
        # Alias normalization (issue #25 "Proposed resolution": applied "on
        # every construction path", not just read_manifest). DataItem is
        # frozen+slots, so fields can't be reassigned via `self.x = ...`
        # after dataclass.__init__ has already bound them; object.__setattr__
        # is the standard escape hatch a frozen dataclass's own
        # __post_init__ uses for this (the language's documented pattern,
        # not a workaround specific to this module). Applied before any
        # validation below, so a caller may pass either the canonical
        # spelling or a known v1/annex alias directly to DataItem(...) and
        # the stored value is always canonical.
        for field_name, field_aliases in _ALIASES.items():
            value = getattr(self, field_name)
            if isinstance(value, str) and value in field_aliases:
                object.__setattr__(self, field_name, field_aliases[value])

        # Hard-required, non-nullable fields (tech-spec v2 §2.2 explicit
        # invariant): rejected at ingest, never silently defaulted.
        if not self.item_id:
            raise DataContractError("item_id is required and must be non-empty")
        if not self.source:
            raise DataContractError("source is required and must be non-empty")
        if not self.language_tag:
            raise DataContractError("language_tag is required and must be non-empty")
        if not self.orthography_system:
            raise DataContractError("orthography_system is required and must be non-empty")
        if not self.provenance:
            raise DataContractError("provenance is required and must be non-empty")

        _require_member(self.record_type, _RECORD_TYPES, "record_type")
        _require_member(self.language_tag, _LANGUAGE_TAGS, "language_tag")
        if self.eng_dialect is not None:
            _require_member(self.eng_dialect, _ENG_DIALECTS, "eng_dialect")
        _require_member(self.orthography_system, _ORTHOGRAPHY_SYSTEMS, "orthography_system")
        _require_member(self.genre, _GENRES, "genre")
        _require_member(self.register, _REGISTERS, "register")
        _require_member(self.rights, _RIGHTS, "rights")
        _require_member(self.consent, _CONSENTS, "consent")
        _require_member(self.training_permission, _TRAINING_PERMISSIONS, "training_permission")
        _require_member(self.cultural_sensitivity, _CULTURAL_SENSITIVITIES, "cultural_sensitivity")
        _require_member(self.sensitivity_tier, _SENSITIVITY_TIERS, "sensitivity_tier")
        _require_member(self.access_tier, _ACCESS_TIERS, "access_tier")
        _require_member(self.object_tier, _OBJECT_TIERS, "object_tier")
        _require_member(self.release_class, _RELEASE_CLASSES, "release_class")
        _require_member(self.speaker_generation, _SPEAKER_GENERATIONS, "speaker_generation")
        _require_member(self.speaker_role, _SPEAKER_ROLES, "speaker_role")
        _require_member(self.gender, _GENDERS, "gender")
        _require_member(self.attribution_mode, _ATTRIBUTION_MODES, "attribution_mode")
        _require_member(self.pii_status, _PII_STATUSES, "pii_status")
        if self.reading_type is not None:
            _require_member(self.reading_type, _READING_TYPES, "reading_type")
        _require_member(self.split, _SPLITS, "split")
        _require_member(self.data_class, _DATA_CLASSES, "data_class")
        _require_member(self.normalizer_status, _NORMALIZER_STATUSES, "normalizer_status")
        _require_member(
            self.normalization_difficulty, _NORMALIZATION_DIFFICULTIES, "normalization_difficulty"
        )
        if self.schema_version != SCHEMA_VERSION:
            raise DataContractError(
                f"schema_version={self.schema_version!r} is not {SCHEMA_VERSION!r}"
            )

        # synthetic=true items must be traceable to their generator (tech-spec
        # v2 §6.4 gold/silver/bronze/synthetic discipline; CONTEXT.md
        # "`data_class`").
        if self.synthetic and not self.generator:
            raise DataContractError("generator is required when synthetic=True")

        # data_class/synthetic must agree in both directions (issue #11,
        # carried forward under the v2 name): a synthetic item must be tagged
        # data_class="synthetic" (never smuggled in as gold/silver/bronze),
        # and data_class="synthetic" must not appear on an item that isn't
        # flagged synthetic.
        if self.synthetic and self.data_class != "synthetic":
            raise DataContractError(
                f"synthetic=True requires data_class='synthetic', got data_class={self.data_class!r}"
            )
        if self.data_class == "synthetic" and not self.synthetic:
            raise DataContractError("data_class='synthetic' requires synthetic=True")

        # release_class/cloud_ok are computed, never free-set (issue #25
        # "Proposed resolution": "a caller passing values that disagree with
        # the derivation must get a DataContractError naming the field").
        expected_release_class = derive_release_class(
            ReleaseClassInputs(
                rights=self.rights,
                training_permission=self.training_permission,
                consent=self.consent,
                cultural_sensitivity=self.cultural_sensitivity,
                community_review_signed_off=self.community_review_signed_off,
            )
        )
        if self.release_class != expected_release_class:
            raise DataContractError(
                f"release_class={self.release_class!r} disagrees with the derivation "
                f"({expected_release_class!r}) from this item's rights/consent/"
                f"training_permission/cultural_sensitivity (field: release_class)"
            )

        expected_cloud_ok = derive_cloud_ok(
            release_class=self.release_class,
            training_permission=self.training_permission,
            sensitivity_tier=self.sensitivity_tier,
            pii_status=self.pii_status,
        )
        if self.cloud_ok != expected_cloud_ok:
            raise DataContractError(
                f"cloud_ok={self.cloud_ok!r} disagrees with the derivation "
                f"({expected_cloud_ok!r}) from this item's release_class/"
                f"training_permission/sensitivity_tier/pii_status (field: cloud_ok)"
            )


EligibilityReason = Literal[
    "rights_not_cleared",
    "training_permission_not_granted",
    "cultural_sensitivity_restricted",
    "release_class_do_not_use",
    "consent_not_granted",
]


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    """The verdict is_eligible() returns: whether an item may proceed, and why not."""

    eligible: bool
    reasons: tuple[EligibilityReason, ...]


def is_eligible(item: DataItem) -> EligibilityResult:
    """Decide whether `item` may proceed past ingestion into annotation or training selection.

    Pure function: no I/O, no side effects, deterministic. Reflects
    tech-spec v2 §2.2's eligibility pre-filter under the v2 `consent` enum,
    keeping the retained fifth (consent) clause per the owner's 2026-09-02
    decision (issue #10 follow-up, reconciled onto v2 in issue #25's
    "Proposed resolution"): defense in depth against a `release_class`
    derivation bug, even though the clause is redundant by design with
    `release_class != do_not_use`.
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

    # consent gate (tech-spec v2 §2.2 "Consent clause"; CONTEXT.md
    # "`consent`"): consent_pending/consent_withdrawn never train;
    # legacy_no_consent trains only after Charter review
    # (community_review_signed_off); informed_consent_research stays
    # training-eligible (the owner's widening) but always derives
    # release_class = internal_eval_only, so it is never cloud_ok.
    consent_ok = item.consent not in _NEVER_TRAINING_CONSENTS and (
        item.consent != "legacy_no_consent" or item.community_review_signed_off
    )
    if not consent_ok:
        reasons.append("consent_not_granted")

    return EligibilityResult(eligible=not reasons, reasons=tuple(reasons))


_DATA_ITEM_FIELD_NAMES = frozenset(f.name for f in dataclass_fields(DataItem))
# Nullable fields (tech-spec v2 §2.2 hard invariant list; issue #25 "read_manifest
# update"): excluded from the hard-required-field check below. `generator` is
# nullable in general (required only when synthetic=True, enforced separately
# in __post_init__).
_NULLABLE_DATA_ITEM_FIELDS = frozenset(
    {"eng_dialect", "lect", "speaker_id", "reading_type", "passage_id", "pair_id", "generator"}
)

__all__ = [
    "SCHEMA_VERSION",
    "LanguageTag",
    "EngDialect",
    "RecordType",
    "OrthographySystem",
    "Genre",
    "Register",
    "Rights",
    "Consent",
    "TrainingPermission",
    "CulturalSensitivity",
    "SensitivityTier",
    "AccessTier",
    "ObjectTier",
    "ReleaseClass",
    "SpeakerGeneration",
    "SpeakerRole",
    "Gender",
    "AttributionMode",
    "PiiStatus",
    "ReadingType",
    "Split",
    "DataClass",
    "NormalizerStatus",
    "NormalizationDifficulty",
    "TargetLanguage",
    "ModelReleaseClass",
    "DataContractError",
    "validate_literal",
    "ReleaseClassInputs",
    "derive_release_class",
    "derive_cloud_ok",
    "DataItem",
    "EligibilityReason",
    "EligibilityResult",
    "is_eligible",
    "read_manifest",
]


def read_manifest(path: Path, *, strict: bool = False) -> list[DataItem]:
    """Read a preprocess-written manifest.jsonl (utils/fine_tune_cajun_preprocess.py's
    `_write_jsonl(manifest_out, eligible_rows)` output) back into `DataItem` records
    (issue #18 / tech-spec §8 review R10: `DataItem(**row)` raises `TypeError` on real
    preprocess output because each row is `asdict(item)` plus extra keys the preprocess
    utility adds for its own downstream stages — `code_switch_spans` (dicts from
    `dataclasses.asdict(CodeSwitchSpan)`). This reader is the round-trip loader issue
    #18/R10/D5-1 asked for: it keeps exactly the `DataItem` fields and drops the rest,
    so callers (compute_coverage_scorecard and friends) can consume real preprocess
    output without depending on its extra, downstream-only keys.

    Every row must supply every required (non-nullable) `DataItem` field (issue #15
    discipline reused: missing/malformed fields are rejected loudly at the boundary,
    not silently defaulted or skipped). A missing required field or invalid JSON
    raises `DataContractError` naming the 1-indexed line number and, for a missing
    field, the offending key — mirroring `PreprocessInputError`'s own
    `f"manifest.jsonl:{line_number}: ..."` message shape so both loaders' errors
    read the same way.

    `schema_version` gate (issue #25 "Proposed resolution"): any row whose
    `schema_version` starts with `"1."` is a v1 record and is rejected with a
    `DataContractError` naming `schema_version` and the offending value — this
    is a distinct, earlier check from the general enum-validation `DataContractError`
    a malformed field raises, so a caller can tell "your input is stale" from "your
    input is malformed."

    Alias normalization (`_normalize_aliases`) is applied to each row before field
    validation, so v1-spelled sidecar values still round-trip through this reader at
    the *value* level even though the schema_version gate above rejects whole v1
    *records*.

    `strict=False` (default) drops any key on a row that isn't a `DataItem` field —
    the tolerant behavior a manifest schema still gaining new sidecar keys (e.g. a
    future ticket's new PP output) needs. `strict=True` raises `DataContractError`
    instead, naming every unknown key — for callers that want to catch an
    unexpected/typo'd key rather than silently drop it.
    """
    items: list[DataItem] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataContractError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise DataContractError(
                f"{path}:{line_number}: each line must be a JSON object, got {type(row).__name__}"
            )

        schema_version_value = row.get("schema_version")
        if isinstance(schema_version_value, str) and schema_version_value.startswith("1."):
            raise DataContractError(
                f"{path}:{line_number}: schema_version={schema_version_value!r} is a v1 "
                "record; migrate via MIG-01 before re-ingesting (field: schema_version)"
            )

        row = _normalize_aliases(row)

        unknown_keys = set(row) - _DATA_ITEM_FIELD_NAMES
        if strict and unknown_keys:
            raise DataContractError(
                f"{path}:{line_number}: unknown key(s) {sorted(unknown_keys)} not on DataItem"
            )
        # Typed dict[str, Any], not dict[str, object]: values here are about
        # to be passed as **kwargs to DataItem's per-field Literal-typed
        # constructor. `object` blocks that at the type-checker level even
        # though __post_init__ is what actually enforces every field's
        # closed enum at runtime; `Any` reflects that this dict's real
        # shape-checking happens there, not in the type system.
        known_fields: dict[str, Any] = {k: v for k, v in row.items() if k in _DATA_ITEM_FIELD_NAMES}

        missing = _DATA_ITEM_FIELD_NAMES - _NULLABLE_DATA_ITEM_FIELDS - set(known_fields)
        if missing:
            raise DataContractError(
                f"{path}:{line_number}: missing required field(s) {sorted(missing)}"
            )

        # Nullable fields (tech-spec v2 §2.2) that the row omits entirely
        # default to None here, since DataItem itself has no field defaults
        # (every field is explicit at construction) — omission is only
        # tolerated for the fields the hard-required-field check above
        # excludes.
        for nullable_field in _NULLABLE_DATA_ITEM_FIELDS:
            known_fields.setdefault(nullable_field, None)

        try:
            # DataItem fields are flat scalars/None/list[str] (no nested
            # dataclasses), so a dict produced by dataclasses.asdict(item)
            # round-trips through the constructor unchanged — same shape in,
            # same shape (typed) out. __post_init__ above is what actually
            # enforces each field's closed enum at runtime.
            items.append(DataItem(**known_fields))
        except DataContractError as exc:
            raise DataContractError(f"{path}:{line_number}: {exc}") from exc

    return items
