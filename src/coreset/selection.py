"""Coreset selection algorithm and per-run manifest (tech-spec v2 §8; §7).

`coreset/__init__.py` builds the coverage *scorecard* (the measurement
layer) but explicitly leaves the selection algorithm itself as a seam for
this ticket (backlog 0012, issue #38). Tech-spec v2 §8: "Coreset selection
runs the eligibility pre-filter (§2) first, then stratifies on
`language_tag`/`lect`/`genre`/`speaker_generation`, then runs
uncertainty×diversity selection (e.g. k-center, DPP, or MMR-style methods)
over a feature vector combining linguistic (diff-catalog cell flags),
lexical, and categorical features — never raw embeddings alone."

This module implements:
  - `select_coreset`: eligibility -> per-stratum quotas -> greedy k-center
    with an MMR-style priority boost for unmet diff-catalog cells.
  - `write_run_manifest` / `manifest_sha256_for_items`: the per-run
    `item_id` + content-hash manifest tech-spec v2 §7 requires ("Every run
    records the item_id + content-hash manifest of its training set
    (data/coreset/manifest_<run_id>.json, hashed into the MLflow run)").
  - `TextSource` / `text_source_from_sidecar_dir`: the real text-in seam
    (issue #45, following #38's "a real text-in seam replacing the item_id
    lexical proxy" close-out note). `DataItem` carries no raw text field, so
    every function above accepts an optional `text_source` callable and
    falls back to the pre-#45 `item_id` lexical proxy per item when it is
    omitted or returns `None` for a given item — existing callers that never
    pass `text_source` see byte-identical behavior to #38, pinned by
    `test_manifest_sha256_formula_unchanged_pinned_fixture`.

Out of scope (issue #38 "Out of scope"): DPP, model embeddings, real data,
the augment utility's own use of the scorecard (separate ticket). Out of
scope for #45 specifically (issue #45 "Out of scope"): wiring the preprocess
utility to actually pass texts (a Wave 4 one-liner), dedup, the normalizer,
changing the stratification fields or quotas.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from data_contract import DataItem, is_eligible

from coreset import CoverageScorecard

# TextSource: a caller-supplied lookup from a DataItem to its raw source text
# (issue #45 "Proposed resolution" 1: "TextSource = Callable[[DataItem], str
# | None]"), the text-in seam this module lacked at #38 close-out ("a real
# text-in seam replacing the item_id lexical proxy" — a later ticket). `None`
# means "no text available for this item" (never raised as an exception),
# so every caller of select_coreset/write_run_manifest/
# manifest_sha256_for_items falls back to the pre-#45 item_id proxy on a
# per-item basis rather than failing the whole call.
TextSource = Callable[[DataItem], "str | None"]

# --- feature vector -----------------------------------------------------

# Categorical fields one-hot'd into the feature vector (issue #38 "Proposed
# resolution": "feature vector = categorical one-hot (lect, genre, register,
# speaker_generation, orthography_system) + hashed char-n-gram bag ... +
# diff_catalog_flags one-hot weighted by the scorecard's gate_class and
# coverage status"). Order fixed here only for readability/debugging; the
# feature vector itself is a dict keyed by feature name, not a positional
# array, so field order carries no semantic weight.
_CATEGORICAL_FIELDS: tuple[str, ...] = (
    "lect",
    "genre",
    "register",
    "speaker_generation",
    "orthography_system",
)

# Stratification fields (tech-spec v2 §8 verbatim: "language_tag/lect/genre/
# speaker_generation"). language_tag is included even though a real caller
# will usually already have pre-filtered to one language, so a mixed-language
# `items` list still stratifies correctly rather than silently conflating
# languages into one cell.
StratifyField = Literal["language_tag", "lect", "genre", "speaker_generation"]
DEFAULT_STRATIFY: tuple[StratifyField, ...] = (
    "language_tag",
    "lect",
    "genre",
    "speaker_generation",
)

# N-GRAM_DIMENSION: dimension of the hashed char-n-gram bag (issue #38
# "hashed char-n-gram bag (stdlib hashlib, dimension configurable)"). No
# source document names a number; 64 buckets is the documented MVP choice —
# large enough that same-stratum items with different surface text rarely
# collapse to identical hashed bags, small enough the feature vector stays a
# small dict. Exposed as a `select_coreset` parameter, not hardcoded, so a
# future ticket can retune it without touching this module (Open Q2-style
# placeholder, tech-spec v2 §8 does not fix this value).
DEFAULT_NGRAM_DIMENSION = 64
_NGRAM_SIZE = 3

# SPEAKER_GENERATION_FLOOR_FRACTION: per-stratum quota floor (issue #38
# "per-stratum quotas proportional to targets with a floor per
# speaker_generation"). No source document names a number; the documented
# MVP choice is that every speaker_generation value actually present in the
# eligible pool receives at least this fraction of an equal a-priori share
# (k / distinct_speaker_generations_present), so a small but real
# speaker_generation group is never rounded to zero purely by proportional
# allocation. Revisit once project management sets a real number (same
# posture as coreset/__init__.py's PARTIAL_THRESHOLD_FRACTION).
SPEAKER_GENERATION_FLOOR_FRACTION = 0.5


class CoresetSelectionError(ValueError):
    """Raised when select_coreset is given invalid arguments (k <= 0, an
    empty eligible pool, or an unknown stratify field)."""


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """The result of one `select_coreset` call (issue #38 "Proposed
    resolution").

    `text_missing_ids`: item_ids of every *eligible* item for which
    `text_source` was supplied but returned `None` (issue #45 "Proposed
    resolution" 1: "items whose source returns None fall back to the proxy
    and are listed in a new SelectionResult.text_missing_ids"). Default
    empty tuple so every pre-#45 caller (no `text_source` argument) sees an
    unchanged `SelectionResult` shape. Reports every eligible item text was
    missing for, not just the ones actually selected — a caller auditing
    text coverage of the eligible pool needs the full picture, not a sample
    biased by which items k-center happened to pick.
    """

    selected_ids: tuple[str, ...]
    per_stratum_counts: dict[str, int]
    coverage_gain_by_cell: dict[str, int]
    text_missing_ids: tuple[str, ...] = ()


def _stratum_key(item: DataItem, stratify: tuple[str, ...]) -> str:
    """`"<field>=<value>|<field>=<value>|..."` in `stratify`'s given order —
    a single string key so per_stratum_counts (and internal bucketing) never
    need a tuple-vs-string branch."""
    return "|".join(f"{field}={getattr(item, field)!r}" for field in stratify)


def _one_hot(value: object, field: str) -> dict[str, float]:
    return {f"{field}={value!r}": 1.0}


def _hashed_ngram_bag(text: str, *, dimension: int, n: int = _NGRAM_SIZE) -> dict[str, float]:
    """A stdlib-only hashed char-n-gram bag: each character n-gram of `text`
    is sha256-hashed and folded into one of `dimension` buckets (the
    hashing-trick pattern, issue #38 "hashed char-n-gram bag (stdlib
    hashlib, dimension configurable)"). Deterministic for the same text and
    dimension — sha256 is a pure function, no seed involved — so this
    contributes no randomness of its own to `select_coreset`'s determinism
    requirement.

    `text` shorter than `n` characters contributes the whole string as its
    only "n-gram" (never raises), matching how a one-word item should still
    get a (weak) lexical signal instead of an empty bag.
    """
    if not text:
        return {}
    grams = [text[i : i + n] for i in range(max(1, len(text) - n + 1))]
    bag: dict[str, float] = {}
    for gram in grams:
        digest = hashlib.sha256(gram.encode("utf-8")).hexdigest()
        bucket = int(digest, 16) % dimension
        key = f"ngram_bucket={bucket}"
        bag[key] = bag.get(key, 0.0) + 1.0
    return bag


def _diff_catalog_flag_features(
    item: DataItem, *, scorecard: CoverageScorecard | None
) -> dict[str, float]:
    """One-hot `diff_catalog_flags` entries, each weighted by the
    scorecard's own `gate_class` (higher gate_class = more severe = weighted
    higher) and coverage status (an `unmet` cell's flag is weighted higher
    than a `met` one) — issue #38 "diff_catalog_flags one-hot weighted by
    the scorecard's gate_class and coverage status." A flag id absent from
    the scorecard (e.g. `scorecard=None`, or a cell id the scorecard's
    catalog doesn't carry) still gets a one-hot entry, weight 1.0 (the
    unweighted baseline) — an item is never dropped from feature space
    purely because the scorecard doesn't recognize one of its flags.
    """
    features: dict[str, float] = {}
    for flag in item.diff_catalog_flags:
        weight = 1.0
        if scorecard is not None:
            # gate_class: 0-3, higher = more severe. +1 keeps the weight
            # strictly positive even at gate_class 0.
            gate_class = scorecard.gate_class.get(flag)
            if gate_class is not None:
                weight += float(gate_class)
            status = scorecard.diff_catalog_coverage.get(flag)
            if status == "unmet":
                weight += 2.0
            elif status == "partial":
                weight += 1.0
            # "met" cells contribute no additional weight: covered
            # constructions are lower priority for further selection.
        features[f"diff_catalog_flag={flag}"] = weight
    return features


def _canonical_text_or_none(item: DataItem, *, text_source: TextSource | None) -> str | None:
    """Resolve `item`'s canonical (NFC-normalized) source text via
    `text_source`, or `None` when no text source was supplied or it returned
    `None` for this item (issue #45 "Proposed resolution" 1). NFC
    normalization matches the exact canonicalization the preprocess utility
    applies before hashing `item_id`
    (`utils/fine_tune_cajun_preprocess.py:_canonicalize_text`, line 212-218,
    `unicodedata.normalize("NFC", text)`) so a text-hashed n-gram bag or
    manifest sha256 is computed over the same canonical bytes `item_id`
    itself was derived from, never a differently-normalized variant."""
    if text_source is None:
        return None
    text = text_source(item)
    if text is None:
        return None
    return unicodedata.normalize("NFC", text)


def _feature_vector(
    item: DataItem,
    *,
    scorecard: CoverageScorecard | None,
    ngram_dimension: int,
    text_source: TextSource | None,
) -> dict[str, float]:
    """The full per-item feature vector: categorical one-hot fields, the
    hashed char-n-gram bag, and weighted diff_catalog_flags one-hot (issue
    #38 "Proposed resolution").

    Lexical signal source: when `text_source` resolves real text for this
    item (issue #45's text-in seam), the n-gram bag hashes that NFC-
    canonicalized text directly. When it does not (`text_source=None`, or
    the source returns `None` for this item), this falls back to the #38
    proxy: `DataItem` carries no raw text field itself (verified:
    `dataclass_fields(DataItem)` has none), but the pipeline's `item_id` for
    text records already *is* a sha256 content hash of that same canonical
    text (`utils/fine_tune_cajun_preprocess.py:_compute_item_id`, confirmed
    by issue #26 "Proposed resolution": "for text records the two [item_id
    and the cloud-ok-manifest hash] are the same value"), so hashing
    `item_id` itself still gives a stable, deterministic (if weaker) lexical
    signal rather than an empty bag.
    """
    features: dict[str, float] = {}
    for field in _CATEGORICAL_FIELDS:
        value = getattr(item, field)
        features.update(_one_hot(value, field))
    text = _canonical_text_or_none(item, text_source=text_source)
    ngram_input = text if text is not None else item.item_id
    features.update(_hashed_ngram_bag(ngram_input, dimension=ngram_dimension))
    features.update(_diff_catalog_flag_features(item, scorecard=scorecard))
    return features


def _sparse_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Euclidean distance between two sparse feature dicts (implicit zeros
    for keys missing from one side). Stdlib only, no numpy: this repo's sole
    runtime dependency is pyyaml (issue #38 wave preamble rule 5)."""
    keys = set(a) | set(b)
    total = 0.0
    for key in keys:
        diff = a.get(key, 0.0) - b.get(key, 0.0)
        total += diff * diff
    return float(total**0.5)


def _stratum_quotas(
    strata: dict[str, list[DataItem]],
    *,
    k: int,
    stratify: tuple[str, ...],
) -> dict[str, int]:
    """Per-stratum quotas proportional to each stratum's share of the
    eligible pool, with a floor per `speaker_generation` (issue #38
    "per-stratum quotas proportional to targets with a floor per
    speaker_generation").

    Three passes: (1) a raw proportional share of `k`, floored to an int;
    (2) a `speaker_generation` floor pass that raises any stratum whose
    `speaker_generation` component is under-quota'd relative to
    `SPEAKER_GENERATION_FLOOR_FRACTION * (k / distinct_speaker_generations)`;
    (3) a largest-remainder fill pass that tops the total back up to
    `min(k, total_eligible)` whenever passes (1)-(2) under-fill it — many
    small strata each rounding `share * k` down to zero is the common case
    (e.g. six singleton strata at k=3: every raw share is 0.5, every
    `int(round(...))` is 0, and with no speaker_generation floor to rescue
    it the naive two-pass total would be 0 even though 6 eligible items
    exist). Pass (3) repeatedly gives +1 to the stratum with spare room
    (`len(members) - quota > 0`) whose fractional remainder
    `share * k - quota` is currently largest, ties broken by stratum key —
    the standard largest-remainder (Hamilton) apportionment method, no
    randomness, and deterministic regardless of dict/set iteration order.

    The floor pass (2) is applied at the `speaker_generation` *group* level
    (summed across every stratum sharing that speaker_generation), then
    redistributed across that group's strata proportionally to their own
    sizes — never given a group with a single stratum by construction, but
    correct either way since the redistribution always sums back to the
    group floor. Final quotas are capped at each stratum's own item count
    (a quota never exceeds the population it draws from) and the whole
    dict's total is capped at `k` by trimming the largest strata last-in,
    first-trimmed (deterministic, since `strata` iteration order here is
    insertion order from `_group_by_stratum`, itself built from `items` in
    the caller's given order).
    """
    total_eligible = sum(len(v) for v in strata.values())
    if total_eligible == 0:
        return {}

    raw_share: dict[str, float] = {}
    raw_quota: dict[str, int] = {}
    for key, members in strata.items():
        share = len(members) / total_eligible
        raw_share[key] = share * k
        raw_quota[key] = min(len(members), int(round(share * k)))

    # speaker_generation floor pass, only meaningful when speaker_generation
    # is one of the stratify fields (it always is in DEFAULT_STRATIFY, but
    # select_coreset accepts a caller-narrowed stratify tuple too).
    if "speaker_generation" in stratify:
        sg_index = stratify.index("speaker_generation")
        groups: dict[str, list[str]] = {}
        for key, members in strata.items():
            sg_value = key.split("|")[sg_index]
            groups.setdefault(sg_value, []).append(key)

        distinct_sg = len(groups)
        equal_share = k / distinct_sg if distinct_sg else 0.0
        floor_per_group = SPEAKER_GENERATION_FLOOR_FRACTION * equal_share

        for sg_value, member_keys in groups.items():
            group_population = sum(len(strata[key]) for key in member_keys)
            group_floor = min(group_population, int(round(floor_per_group)))
            group_current = sum(raw_quota[key] for key in member_keys)
            if group_current < group_floor:
                deficit = group_floor - group_current
                # Redistribute the deficit proportionally to each stratum's
                # own population within the group, largest stratum first so
                # rounding remainder favors the stratum most able to supply
                # it (never exceeding that stratum's own population).
                ordered = sorted(member_keys, key=lambda mk: -len(strata[mk]))
                for key in ordered:
                    if deficit <= 0:
                        break
                    room = len(strata[key]) - raw_quota[key]
                    if room <= 0:
                        continue
                    bump = min(room, deficit)
                    raw_quota[key] += bump
                    deficit -= bump

    # Largest-remainder fill pass (3): top the total back up to
    # min(k, total_eligible) whenever passes (1)-(2) under-filled it. Ties
    # in remainder are broken by stratum key so the result never depends on
    # dict iteration order; a stratum with no spare room (quota already at
    # its own population) is skipped even if its remainder is largest.
    target_total = min(k, total_eligible)
    current_total = sum(raw_quota.values())
    if current_total < target_total:
        shortfall = target_total - current_total
        while shortfall > 0:
            candidates = [key for key in strata if raw_quota[key] < len(strata[key])]
            if not candidates:
                break
            best_key = min(
                candidates,
                key=lambda mk: (-(raw_share[mk] - raw_quota[mk]), mk),
            )
            raw_quota[best_key] += 1
            shortfall -= 1

    # Cap the total at k, trimming from the largest quotas first (stable,
    # deterministic: ties broken by stratum key so trimming order never
    # depends on dict/set iteration order).
    total_quota = sum(raw_quota.values())
    if total_quota > k:
        excess = total_quota - k
        for key in sorted(raw_quota, key=lambda mk: (-raw_quota[mk], mk)):
            if excess <= 0:
                break
            trim = min(raw_quota[key], excess)
            raw_quota[key] -= trim
            excess -= trim

    return raw_quota


def _group_by_stratum(
    items: list[DataItem], stratify: tuple[str, ...]
) -> dict[str, list[DataItem]]:
    groups: dict[str, list[DataItem]] = {}
    for item in items:
        key = _stratum_key(item, stratify)
        groups.setdefault(key, []).append(item)
    return groups


def _relevance(item: DataItem, *, scorecard: CoverageScorecard | None) -> float:
    """The item's relevance component of the MMR-style score: the coverage
    gap of the cells its `diff_catalog_flags` touch (issue #38 "relevance =
    coverage gap of the item's cells"). An `unmet` cell contributes the
    largest gap (1.0), `partial` a smaller one (0.5), `met` none (0.0); an
    item touching no cells, or with no scorecard given, has relevance 0.0
    (it can still be selected on diversity alone)."""
    if scorecard is None or not item.diff_catalog_flags:
        return 0.0
    gap = 0.0
    for flag in item.diff_catalog_flags:
        status = scorecard.diff_catalog_coverage.get(flag)
        if status == "unmet":
            gap += 1.0
        elif status == "partial":
            gap += 0.5
    return gap


def _greedy_k_center(
    candidates: list[DataItem],
    *,
    quota: int,
    seed: int,
    scorecard: CoverageScorecard | None,
    ngram_dimension: int,
    lambda_relevance: float,
    text_source: TextSource | None,
) -> list[DataItem]:
    """Deterministic farthest-first (greedy k-center) selection over
    `candidates`, with an MMR-style score combining relevance (coverage gap)
    and diversity (distance to the already-selected set): `score(item) =
    lambda_relevance * relevance(item) + (1 - lambda_relevance) *
    diversity(item)` (issue #38 "greedy k-center (farthest-first) with an
    MMR-style score (relevance = coverage gap of the item's cells, diversity
    = distance to the selected set)").

    Deterministic for a given `seed` (issue #38 "deterministic for a
    seed"): `seed` only breaks ties among items with an identical score
    (never introduces true randomness into the score itself, so re-running
    the same inputs with the same seed always yields the same picks, and
    two different seeds may only disagree on genuine ties). Ties are broken
    by a stable, seed-derived rank: `sha256(f"{seed}:{item_id}")`, lowest
    hash wins — cheap, seed-sensitive, and never favors a particular
    item_id ordering across different seeds.
    """
    if quota <= 0 or not candidates:
        return []

    features = {
        item.item_id: _feature_vector(
            item, scorecard=scorecard, ngram_dimension=ngram_dimension, text_source=text_source
        )
        for item in candidates
    }
    relevance = {item.item_id: _relevance(item, scorecard=scorecard) for item in candidates}

    def tie_break(item_id: str) -> str:
        return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()

    remaining = list(candidates)
    selected: list[DataItem] = []

    # First pick: highest relevance alone (no selected set yet to measure
    # diversity against), ties broken by the seed-derived rank.
    remaining.sort(key=lambda i: (-relevance[i.item_id], tie_break(i.item_id)))
    first = remaining.pop(0)
    selected.append(first)

    while remaining and len(selected) < quota:
        best_item: DataItem | None = None
        best_score = float("-inf")
        best_tie = ""
        for item in remaining:
            diversity = min(
                _sparse_distance(features[item.item_id], features[s.item_id]) for s in selected
            )
            score = lambda_relevance * relevance[item.item_id] + (1 - lambda_relevance) * diversity
            tie = tie_break(item.item_id)
            if best_item is None or score > best_score or (score == best_score and tie < best_tie):
                best_item = item
                best_score = score
                best_tie = tie
        assert best_item is not None
        selected.append(best_item)
        remaining.remove(best_item)

    return selected


def select_coreset(
    items: list[DataItem],
    *,
    scorecard: CoverageScorecard | None = None,
    k: int,
    seed: int,
    stratify: tuple[str, ...] = DEFAULT_STRATIFY,
    lambda_relevance: float = 0.5,
    ngram_dimension: int = DEFAULT_NGRAM_DIMENSION,
    text_source: TextSource | None = None,
) -> SelectionResult:
    """Select up to `k` items from `items` (tech-spec v2 §8; issue #38
    "Proposed resolution").

    Pipeline: `is_eligible` pre-filter -> stratify on `stratify`
    (default `language_tag`/`lect`/`genre`/`speaker_generation`, tech-spec
    v2 §8 verbatim) -> per-stratum quotas (`_stratum_quotas`) -> greedy
    k-center with an MMR-style relevance/diversity score
    (`_greedy_k_center`) run independently per stratum against its own
    quota. Pure and deterministic for a given `seed` and `text_source`; no
    file or network I/O of its own (matches every other coreset/
    data_contract function's purity contract) — `text_source` is a plain
    callable the caller supplies (e.g. `text_source_from_sidecar_dir`,
    below, which *does* do I/O, at the caller's boundary, not inside this
    function).

    `text_source` (issue #45 "Proposed resolution" 1, the text-in seam
    replacing the #38 `item_id` lexical proxy): when supplied, resolves
    each eligible item's real source text for the char-n-gram bag; an item
    for which it returns `None` falls back to the `item_id` proxy and its
    item_id is recorded in the result's `text_missing_ids`. `None` (the
    default) preserves the exact #38 behavior for every existing caller.

    Raises `CoresetSelectionError` if `k <= 0` or `stratify` names a field
    not in `DEFAULT_STRATIFY`'s superset (`language_tag`/`lect`/`genre`/
    `speaker_generation` — the only four fields this module's quota/grouping
    logic understands; a caller-supplied fifth field would silently degrade
    to string-formatting an attribute this module never validates against
    DataItem's own field set otherwise).
    """
    if k <= 0:
        raise CoresetSelectionError(f"k={k!r} must be positive")
    allowed_stratify_fields = frozenset(DEFAULT_STRATIFY)
    for field in stratify:
        if field not in allowed_stratify_fields:
            raise CoresetSelectionError(
                f"stratify field {field!r} is not one of {sorted(allowed_stratify_fields)}"
            )

    eligible_items = [item for item in items if is_eligible(item).eligible]

    strata = _group_by_stratum(eligible_items, stratify)
    quotas = _stratum_quotas(strata, k=k, stratify=stratify)

    selected_ids: list[str] = []
    per_stratum_counts: dict[str, int] = {}
    coverage_gain_by_cell: dict[str, int] = {}
    text_missing_ids: list[str] = []
    if text_source is not None:
        for item in eligible_items:
            if _canonical_text_or_none(item, text_source=text_source) is None:
                text_missing_ids.append(item.item_id)

    for key in strata:
        quota = quotas.get(key, 0)
        chosen = _greedy_k_center(
            strata[key],
            quota=quota,
            seed=seed,
            scorecard=scorecard,
            ngram_dimension=ngram_dimension,
            lambda_relevance=lambda_relevance,
            text_source=text_source,
        )
        per_stratum_counts[key] = len(chosen)
        for item in chosen:
            selected_ids.append(item.item_id)
            for flag in item.diff_catalog_flags:
                coverage_gain_by_cell[flag] = coverage_gain_by_cell.get(flag, 0) + 1

    return SelectionResult(
        selected_ids=tuple(selected_ids),
        per_stratum_counts=per_stratum_counts,
        coverage_gain_by_cell=coverage_gain_by_cell,
        text_missing_ids=tuple(sorted(text_missing_ids)),
    )


# --- per-run manifest (tech-spec v2 §7; issue #38 shared format) -----------

# MANIFEST_SCHEMA_VERSION: the manifest's own schema_version key (distinct
# from DataItem.schema_version / CoverageScorecard.schema_version, though
# currently the same string value) — fixed by the wave's shared format
# ("Shared formats fixed for this wave"), not derived from either.
MANIFEST_SCHEMA_VERSION = "2.0.0"


def _item_content_sha256(item: DataItem, *, text_source: TextSource | None) -> str:
    """The content hash recorded per manifest item (issue #45 "Proposed
    resolution" 2). When `text_source` resolves real text for this item,
    the hash is the sha256 hex of the NFC-canonicalized text's UTF-8 bytes —
    the wave's fixed per-item formula from W3-e on ("a per-item sha256 is
    the sha256 hex of the UTF-8 bytes of the item's canonical source text
    when a text source is supplied, else the item_id proxy"). Otherwise this
    falls back to the #38 proxy: `DataItem` carries no raw text of its own
    (see `_feature_vector`'s docstring for the same finding), but the
    pipeline's `item_id` for text records already *is* a sha256 content hash
    over that same canonical text
    (`utils/fine_tune_cajun_preprocess.py:_compute_item_id`; issue #26
    "Proposed resolution": "for text records the two are the same value"),
    so `item_id` remains a faithful stand-in when no text source is given.
    """
    text = _canonical_text_or_none(item, text_source=text_source)
    if text is not None:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return item.item_id


def manifest_sha256_for_items(items: list[DataItem], *, text_source: TextSource | None = None) -> str:
    """sha256 hex of the UTF-8 bytes of
    `json.dumps(items, sort_keys=True, separators=(",", ":"))` with `items`
    sorted by `item_id` — exactly the wave's shared format ("Shared formats
    fixed for this wave... manifest_sha256 = sha256 hex of the UTF-8 bytes
    of json.dumps(items, sort_keys=True, separators=(",", ":")) with items
    sorted by item_id"). `items` here is the manifest's own
    `[{"item_id": ..., "sha256": ...}, ...]` list, not the caller's
    `DataItem` objects — matching the format's own "items" key.

    This outer formula never changes with `text_source` (binding: "the
    per-run manifest's outer manifest_sha256 formula never changes"); only
    each row's own `sha256` value depends on it, via `_item_content_sha256`.
    `text_source=None` (the default) reproduces the pinned #38 fixture
    exactly, so `test_manifest_sha256_formula_unchanged_pinned_fixture`
    holds without ever passing a text source.
    """
    manifest_items = sorted(
        (
            {"item_id": item.item_id, "sha256": _item_content_sha256(item, text_source=text_source)}
            for item in items
        ),
        key=lambda row: row["item_id"],
    )
    payload = json.dumps(manifest_items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_run_manifest(
    path: Path, run_id: str, items: list[DataItem], *, text_source: TextSource | None = None
) -> None:
    """Write the per-run coreset manifest JSON (tech-spec v2 §7: "Every run
    records the item_id + content-hash manifest of its training set
    (data/coreset/manifest_<run_id>.json, hashed into the MLflow run)"; this
    wave's shared format for the exact shape). Overwrites `path`
    unconditionally — callers own path uniqueness (typically
    `data/coreset/manifest_<run_id>.json`, run_id already unique).

    `text_source`: see `manifest_sha256_for_items` — threaded through so the
    manifest's per-item `sha256` values and its own `manifest_sha256` field
    are always computed from the same row data (never independently, which
    could disagree)."""
    manifest_items = sorted(
        (
            {"item_id": item.item_id, "sha256": _item_content_sha256(item, text_source=text_source)}
            for item in items
        ),
        key=lambda row: row["item_id"],
    )
    manifest = {
        "run_id": run_id,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "items": manifest_items,
        "manifest_sha256": manifest_sha256_for_items(items, text_source=text_source),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


# --- text source from the ADR 0006 sidecar layout --------------------------


def text_source_from_sidecar_dir(raw_dir: Path) -> TextSource:
    """Build a `TextSource` over the raw text files in `raw_dir`, the ADR
    0006 sidecar layout (`utils/fine_tune_cajun_preprocess.py`'s module
    docstring, lines 16-22, quoted by ADR 0006): "`input_dir` contains raw
    text files plus a sidecar `manifest.jsonl`... each carrying a `filename`
    key plus every `data_contract.DataItem` field except `item_id` and
    `language_tag` (both computed by this module)".

    Judgment call (verified before assuming a path, per issue #45's own
    instruction): `DataItem.source` is *not* a filename or path — it is a
    free-text provenance string copied straight from the sidecar's own
    `source` field (e.g. `"test-collection"` in this module's test fixtures;
    `_build_data_item`, `utils/fine_tune_cajun_preprocess.py:363`,
    `source = _require_str(fields, "source", filename)` — `filename` and
    `source` are two independent sidecar keys). `DataItem` itself carries no
    path/filename field at all (its 36 fields have none). The sidecar's
    `filename` key that *does* name the raw file is consumed entirely inside
    `_load_sidecar_manifest`/the ingest loop and never becomes part of
    `DataItem` — ADR 0006's own "Consequences" section notes the round-trip
    gap between the output manifest and a reconstructable `DataItem` is a
    separate, unresolved backlog item, not something this ticket can assume
    away.

    Given that, the only content-addressable way to map a `DataItem` back to
    a raw file without inventing a new `DataItem` field (frozen this wave)
    is by the same identity the preprocess utility already establishes:
    `item_id` *is* `_compute_item_id(_canonicalize_text(raw_text))`
    (`utils/fine_tune_cajun_preprocess.py:212-231`). This function therefore
    scans every regular file directly under `raw_dir` (skipping
    `manifest.jsonl` itself and any subdirectory, matching the ingest loop's
    own `p.is_file() and p.name != "manifest.jsonl"` filter, line 878),
    reads and NFC-canonicalizes each with the identical convention, and
    indexes the result by that same sha256 — so the returned `TextSource`
    looks an item's `item_id` up directly in that index. A raw file that is
    not valid UTF-8, or a `raw_dir` that does not exist, contributes nothing
    to the index rather than raising: the returned callable still answers
    `None` for every item_id it cannot resolve, matching this ticket's "never
    an exception at selection time" requirement. The index is built once,
    eagerly, when this function is called (not lazily per lookup), since a
    caller typically resolves many items against the same `raw_dir` in one
    `select_coreset`/`write_run_manifest` call.
    """
    index: dict[str, str] = {}
    if raw_dir.is_dir():
        for path in sorted(raw_dir.iterdir()):
            if not path.is_file() or path.name == "manifest.jsonl":
                continue
            try:
                raw_text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            text = unicodedata.normalize("NFC", raw_text)
            content_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
            # First file wins on a hash collision (two files with identical
            # canonical content): deterministic under `sorted()` above,
            # never an error at index-build time.
            index.setdefault(content_id, text)

    def _lookup(item: DataItem) -> str | None:
        return index.get(item.item_id)

    return _lookup


__all__ = [
    "StratifyField",
    "DEFAULT_STRATIFY",
    "DEFAULT_NGRAM_DIMENSION",
    "SPEAKER_GENERATION_FLOOR_FRACTION",
    "MANIFEST_SCHEMA_VERSION",
    "TextSource",
    "CoresetSelectionError",
    "SelectionResult",
    "select_coreset",
    "manifest_sha256_for_items",
    "write_run_manifest",
    "text_source_from_sidecar_dir",
]
