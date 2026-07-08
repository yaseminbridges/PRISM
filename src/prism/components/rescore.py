"""C2 — Re-score: pure deterministic fit scoring and conservative re-ranking.

The LLM never emits a rank-affecting number — all arithmetic lives here.

fit_raw = Σ w_match   * g(freq) * ic   for matched features
        + Σ w_partial * g(freq) * ic   for partial matches
        - Σ w_miss    * g(freq) * ic   for expected_absent (not age_excused)
        - Σ w_unexp   * ic             for unexplained patient terms
        - |contradictions| * w_contra

fit_score = tanh(fit_raw)  -- bounded in (-1, 1)

Conservative re-rank: swap two candidates only when BOTH hold:
  (1) |Δ Exomiser combined_score| <= near_tie_threshold
  (2) |Δ fit_score| > margin

Aggressive re-rank (config): key = α·exo_norm + (1-α)·fit_score
"""
import functools
import math
from dataclasses import dataclass
from typing import Literal

from prism.models.candidate import FitEvidence
from prism.models.report import ReRankedCandidate
from prism.ontology.frequency import frequency_class_weight


@dataclass
class RescoreWeights:
    """Tunable weights for the fit score formula."""
    w_match: float = 1.0          # contribution per matched feature
    w_partial: float = 0.5        # contribution per partial match
    w_miss: float = 0.5           # penalty per expected-absent feature
    w_unexp: float = 0.3          # penalty per unexplained patient term
    w_contra: float = 1.0  # flat penalty per contradiction (not IC-weighted)
    # Conservative re-rank thresholds
    near_tie_threshold: float = 0.05  # Exomiser combined_score delta for "near tie"
    margin: float = 0.1               # min |Δfit_score| to trigger a conservative swap
    # Aggressive re-rank blend weight
    alpha: float = 0.7   # weight of Exomiser score; (1-alpha) is fit_score weight


def _compute_fit_raw(
    fit: FitEvidence,
    ic_map: dict[str, float],
    weights: RescoreWeights,
) -> float:
    """Compute the unnormalized fit score (before tanh)."""
    raw = 0.0
    # Positive contributions — features the patient has that the disease expects.
    # Weighted by frequency (an obligate feature match matters more than an occasional one)
    # and by IC (matching a rare/specific term is more informative than a general one).
    for fm in fit.matched:
        raw += weights.w_match * frequency_class_weight(fm.disease_feature.frequency_class) * fm.disease_feature.ic
    for fm in fit.partial:
        # Partial matches (LLM-resolved broader terms) contribute less than exact matches.
        raw += weights.w_partial * frequency_class_weight(fm.disease_feature.frequency_class) * fm.disease_feature.ic
    # Penalties — things that argue against this disease.
    for feat in fit.expected_absent:
        # Disease feature explicitly absent in the patient (age_excused features are not here).
        raw -= weights.w_miss * frequency_class_weight(feat.frequency_class) * feat.ic
    for term in fit.unexplained:
        # Patient has a symptom this disease doesn't account for at all.
        raw -= weights.w_unexp * ic_map.get(term.id, 0.0)
    # Contradictions get a flat penalty — the disease says this feature should NOT be present,
    # but the patient has it. A strong negative signal regardless of how rare the feature is.
    raw -= len(fit.contradictions) * weights.w_contra
    return raw


def compute_fit_score(
    fit: FitEvidence,
    ic_map: dict[str, float],
    weights: RescoreWeights | None = None,
) -> float:
    """Compute a single fit_score from populated FitEvidence.

    Returns tanh(fit_raw) — bounded in (-1, 1).
    ic_map provides per-term IC for unexplained patient terms (not stored on HpoTerm).
    """
    if weights is None:
        weights = RescoreWeights()
    return math.tanh(_compute_fit_raw(fit, ic_map, weights))


def rerank(
    candidates: list[ReRankedCandidate],
    ic_map: dict[str, float],
    weights: RescoreWeights | None = None,
    mode: Literal["conservative", "aggressive"] = "conservative",
) -> list[ReRankedCandidate]:
    """Score every candidate's FitEvidence, then re-sort under the chosen mode.

    Returns a new list with fit_score, new_rank, and rationale filled in.
    """
    if weights is None:
        weights = RescoreWeights()

    # Step 1 — score: compute fit_raw and fit_score = tanh(fit_raw) for each candidate
    scored_pairs: list[tuple[ReRankedCandidate, float]] = []
    for rc in candidates:
        raw = _compute_fit_raw(rc.fit, ic_map, weights)
        scored_pairs.append((
            rc.model_copy(update={"fit": rc.fit.model_copy(update={"fit_score": math.tanh(raw)})}),
            raw,
        ))

    # Step 2 — sort
    if mode == "aggressive":
        # AGGRESSIVE MODE: blend Exomiser score (alpha=0.7) with PRISM fit (1-alpha=0.3).
        # Normalize fit_raw across this candidate set so the blend isn't defeated by
        # tanh saturation.  When many high-IC unexplained terms drive every candidate
        # to fit_score ≈ -1.0, dividing by max_abs_raw restores discrimination.
        max_abs_raw = max(abs(raw) for _, raw in scored_pairs) or 1.0
        max_exo = max(rc.candidate.combined_score for rc, _ in scored_pairs) or 1.0

        def _aggressive_key(pair: tuple[ReRankedCandidate, float]) -> float:
            rc, raw = pair
            exo_norm = rc.candidate.combined_score / max_exo
            fit_norm = raw / max_abs_raw  # in (-1, 1), preserves ordering without saturation
            # Candidates with zero phenotype hits (no matched or partial features)
            # form a lower tier regardless of Exomiser score: +2 pushes them past
            # the (-1, 1) range of all other candidates in the ascending sort key.
            no_hits = not rc.fit.matched and not rc.fit.partial
            tier_penalty = 2.0 if no_hits else 0.0
            return -(weights.alpha * exo_norm + (1 - weights.alpha) * fit_norm) + tier_penalty

        reranked = [rc for rc, _ in sorted(scored_pairs, key=_aggressive_key)]
    else:
        # CONSERVATIVE MODE (default): respect Exomiser's ranking unless two candidates
        # are nearly tied on combined_score AND PRISM's fit_score clearly prefers one.
        # This means PRISM only intervenes when Exomiser itself was uncertain.
        scored = [rc for rc, _ in scored_pairs]
        def _compare(a: ReRankedCandidate, b: ReRankedCandidate) -> int:
            """Conservative comparator: only override Exomiser on near-tie + fit margin."""
            score_diff = abs(a.candidate.combined_score - b.candidate.combined_score)
            if score_diff <= weights.near_tie_threshold:
                fit_diff = (a.fit.fit_score or 0.0) - (b.fit.fit_score or 0.0)
                if abs(fit_diff) > weights.margin:
                    return -1 if fit_diff > 0 else 1
            if a.candidate.combined_score != b.candidate.combined_score:
                return -1 if a.candidate.combined_score > b.candidate.combined_score else 1
            return 0
        reranked = sorted(scored, key=functools.cmp_to_key(_compare))

    # Step 3 — assign new_rank and rationale
    result: list[ReRankedCandidate] = []
    for new_rank, rc in enumerate(reranked, start=1):
        result.append(rc.model_copy(update={
            "new_rank": new_rank,
            "rationale": _rationale(rc, new_rank),
        }))
    return result


def _rationale(rc: ReRankedCandidate, new_rank: int) -> str:
    fit = rc.fit
    score_str = f"fit={fit.fit_score:.3f}" if fit.fit_score is not None else "fit=?"
    parts = []
    if fit.matched:
        parts.append(f"matched={len(fit.matched)}")
    if fit.partial:
        parts.append(f"partial={len(fit.partial)}")
    if fit.unexplained:
        parts.append(f"unexplained={len(fit.unexplained)}")
    if fit.expected_absent:
        parts.append(f"expected_absent={len(fit.expected_absent)}")
    if fit.contradictions:
        parts.append(f"contradictions={len(fit.contradictions)}")
    summary = " ".join(parts) if parts else "no phenotype coverage"
    delta = rc.old_rank - new_rank
    movement = f"↑{delta}" if delta > 0 else (f"↓{abs(delta)}" if delta < 0 else "=")
    return f"{movement} [{score_str}] {summary}"