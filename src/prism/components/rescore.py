"""C2 — Re-score: deterministic fit scoring and re-ranking.

The LLM never emits a rank-affecting number — all arithmetic lives here.

fit_raw = Σ w_match   * g(freq) * ic   for matched features
        + Σ w_partial * g(freq) * ic   for partial matches
        - Σ w_miss    * g(freq) * ic   for expected_absent (not age_excused)
        - Σ w_unexp   * ic             for unexplained patient terms
        - |contradictions| * w_contra

fit_score = sigmoid(fit_raw / T)  -- bounded in (0, 1), T=2 for good spread

Two re-rank modes:
  prism   — sort by fit_score alone (phenotype fit only, ignores Exomiser)
  blended — α·exo_norm + (1-α)·fit_norm  (combines Exomiser + PRISM signal)
"""
import math
from dataclasses import dataclass
from typing import Literal

from prism.models.candidate import FitEvidence
from prism.models.report import ReRankedCandidate
from prism.ontology.frequency import frequency_class_weight

# Sigmoid temperature: controls spread of fit_score across 0–1.
# T=2 means fit_raw=±4 maps to ~0.88/0.12 — good granularity for typical cases.
_SIGMOID_T: float = 2.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x / _SIGMOID_T))


@dataclass
class RescoreWeights:
    """Tunable weights for the fit score formula."""
    w_match: float = 1.0          # contribution per matched feature
    w_partial: float = 0.5        # contribution per partial match
    w_miss: float = 0.5           # penalty per expected-absent feature
    w_unexp: float = 0.3          # penalty per unexplained patient term
    w_contra: float = 1.0  # flat penalty per contradiction (not IC-weighted)
    # Blended re-rank weight
    alpha: float = 0.7            # weight of Exomiser score; (1-alpha) is fit_score weight


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
    return _sigmoid(_compute_fit_raw(fit, ic_map, weights))


def rerank(
    candidates: list[ReRankedCandidate],
    ic_map: dict[str, float],
    weights: RescoreWeights | None = None,
    mode: Literal["prism", "blended"] = "blended",
) -> list[ReRankedCandidate]:
    """Score every candidate's FitEvidence, then re-sort under the chosen mode.

    Returns a new list with fit_score, new_rank, and rationale filled in.

    Modes:
      prism   — rank by PRISM fit_score alone (phenotype fit, ignores Exomiser rank)
      blended — α·exo_norm + (1-α)·fit_norm (default α=0.7)
    """
    if weights is None:
        weights = RescoreWeights()

    if not candidates:
        return []

    # Step 1 — score every candidate
    scored_pairs: list[tuple[ReRankedCandidate, float]] = []
    for rc in candidates:
        raw = _compute_fit_raw(rc.fit, ic_map, weights)
        scored_pairs.append((
            rc.model_copy(update={"fit": rc.fit.model_copy(update={"fit_score": _sigmoid(raw)})}),
            raw,
        ))

    # Step 2 — sort
    if mode == "prism":
        # PRISM mode: rank purely by phenotype fit score, Exomiser score ignored.
        reranked = [rc for rc, _ in sorted(scored_pairs, key=lambda p: -(p[0].fit.fit_score or 0.0))]
    else:
        # BLENDED mode: weighted combination of Exomiser and PRISM scores.
        # Normalise fit_raw (not fit_score) across the candidate set so the blend
        # isn't defeated by sigmoid saturation when all candidates score similarly.
        max_abs_raw = max((abs(raw) for _, raw in scored_pairs), default=1.0) or 1.0
        max_exo = max((rc.candidate.combined_score for rc, _ in scored_pairs), default=1.0) or 1.0

        def _blended_key(pair: tuple[ReRankedCandidate, float]) -> float:
            rc, raw = pair
            exo_norm = rc.candidate.combined_score / max_exo
            fit_norm = raw / max_abs_raw
            # Candidates with no phenotype hits form a lower tier regardless of Exomiser score.
            no_hits = not rc.fit.matched and not rc.fit.partial
            tier_penalty = 2.0 if no_hits else 0.0
            return -(weights.alpha * exo_norm + (1 - weights.alpha) * fit_norm) + tier_penalty

        reranked = [rc for rc, _ in sorted(scored_pairs, key=_blended_key)]

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