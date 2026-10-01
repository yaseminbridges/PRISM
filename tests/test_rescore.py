"""Tests for C2 Re-score (rescore.py).

All assertions are exact (deterministic) — the LLM never touches this layer.
"""
from __future__ import annotations
import math
import pytest

from prism.models.phenopacket import HpoTerm
from prism.models.candidate import DiseaseFeature, FeatureMatch, FitEvidence
from prism.models.exomiser import ExomiserCandidate, Variant
from prism.models.report import ReRankedCandidate
from prism.components.rescore import RescoreWeights, compute_fit_score, rerank


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

W = RescoreWeights(
    w_match=1.0, w_partial=0.5, w_miss=0.5,
    w_unexp=0.3, w_contra=1.0,
)

def _term(hpo_id: str) -> HpoTerm:
    return HpoTerm(id=hpo_id, label=hpo_id)

def _feat(hpo_id: str, freq: str, ic: float, cardinality: str | None = None) -> DiseaseFeature:
    return DiseaseFeature(
        hpo_id=hpo_id, hpo_label=hpo_id, frequency_class=freq, ic=ic, source="HPOA",
        cardinality=cardinality,
    )

def _match(patient_id: str, feat: DiseaseFeature, relation="exact") -> FeatureMatch:
    return FeatureMatch(
        patient_term=_term(patient_id),
        disease_feature=feat,
        relation=relation,
        provenance="test",
    )

def _fit(
    matched=(), partial=(), unexplained=(),
    expected_absent=(), contradictions=(),
) -> FitEvidence:
    return FitEvidence(
        disease_id="OMIM:999",
        matched=list(matched),
        partial=list(partial),
        unexplained=list(unexplained),
        expected_absent=list(expected_absent),
        contradictions=list(contradictions),
    )

def _candidate(disease_id: str, combined_score: float, exomiser_rank: int) -> ExomiserCandidate:
    return ExomiserCandidate(
        gene_symbol="GENE",
        disease_id=disease_id,
        disease_name=disease_id,
        moi="AD",
        exomiser_rank=exomiser_rank,
        combined_score=combined_score,
        phenotype_score=combined_score,
        variant_score=combined_score,
        variants=[Variant(variant_id="1-100-A-T", acmg=None, pathogenicity_score=None, consequence=None)],
    )

def _rc(disease_id: str, combined_score: float, rank: int, fit: FitEvidence) -> ReRankedCandidate:
    return ReRankedCandidate(
        candidate=_candidate(disease_id, combined_score, rank),
        fit=fit,
        old_rank=rank,
        new_rank=rank,
        rationale="",
    )


# ---------------------------------------------------------------------------
# compute_fit_score — exact arithmetic assertions
# ---------------------------------------------------------------------------

def _sig(x: float, T: float = 2.0) -> float:
    return 1.0 / (1.0 + math.exp(-x / T))


class TestComputeFitScore:
    def test_empty_fit_is_neutral(self):
        # fit_raw=0 → sigmoid(0)=0.5 (no evidence either way)
        assert compute_fit_score(_fit(), ic_map={}, weights=W) == pytest.approx(0.5)

    def test_single_obligate_match(self):
        # fit_raw = 1.0 * 1.0 * 2.0 = 2.0  →  sigmoid(2.0/2)
        feat = _feat("HP:0001", "Obligate", ic=2.0)
        fit = _fit(matched=[_match("HP:0001", feat)])
        score = compute_fit_score(fit, ic_map={}, weights=W)
        assert score == pytest.approx(_sig(2.0))

    def test_partial_match_half_weight(self):
        # fit_raw = 0.5 * 1.0 * 2.0 = 1.0  →  sigmoid(1.0/2)
        feat = _feat("HP:0001", "Obligate", ic=2.0)
        fit = _fit(partial=[_match("HP:0001", feat, relation="partial")])
        score = compute_fit_score(fit, ic_map={}, weights=W)
        assert score == pytest.approx(_sig(1.0))

    def test_unexplained_penalty(self):
        # fit_raw = -0.3 * 3.0 = -0.9  →  sigmoid(-0.9/2)
        ic_map = {"HP:0002": 3.0}
        fit = _fit(unexplained=[_term("HP:0002")])
        score = compute_fit_score(fit, ic_map=ic_map, weights=W)
        assert score == pytest.approx(_sig(-0.9))

    def test_expected_absent_penalty(self):
        # fit_raw = -0.5 * 1.0 * 4.0 = -2.0  →  sigmoid(-2.0/2)
        feat = _feat("HP:0003", "Obligate", ic=4.0)
        fit = _fit(expected_absent=[feat])
        score = compute_fit_score(fit, ic_map={}, weights=W)
        assert score == pytest.approx(_sig(-2.0))

    def test_contradiction_flat_penalty(self):
        # fit_raw = -1.0 * 1 = -1.0  →  sigmoid(-1.0/2)
        feat = _feat("HP:0004", "Excluded", ic=0.0)
        fit = _fit(contradictions=[feat])
        score = compute_fit_score(fit, ic_map={}, weights=W)
        assert score == pytest.approx(_sig(-1.0))

    def test_two_contradictions(self):
        # fit_raw = -1.0 * 2 = -2.0  →  sigmoid(-2.0/2)
        feat = _feat("HP:0004", "Excluded", ic=0.0)
        fit = _fit(contradictions=[feat, feat])
        score = compute_fit_score(fit, ic_map={}, weights=W)
        assert score == pytest.approx(_sig(-2.0))

    def test_match_outweighs_unexplained(self):
        # match: 1.0 * 1.0 * 5.0 = 5.0; unexplained: -0.3 * 1.0 = -0.3 → fit_raw=4.7
        feat = _feat("HP:0001", "Obligate", ic=5.0)
        fit = _fit(
            matched=[_match("HP:0001", feat)],
            unexplained=[_term("HP:0002")],
        )
        score = compute_fit_score(fit, ic_map={"HP:0002": 1.0}, weights=W)
        assert score == pytest.approx(_sig(4.7))

    def test_score_bounded_positive(self):
        feats = [_feat(f"HP:{i:04d}", "Obligate", ic=10.0) for i in range(20)]
        fit = _fit(matched=[_match(f"HP:{i:04d}", f) for i, f in enumerate(feats)])
        score = compute_fit_score(fit, ic_map={}, weights=W)
        assert 0.5 < score <= 1.0

    def test_score_bounded_negative(self):
        feats = [_feat(f"HP:{i:04d}", "Excluded", ic=2.0) for i in range(20)]
        fit = _fit(contradictions=feats)
        score = compute_fit_score(fit, ic_map={}, weights=W)
        assert 0.0 < score < 0.5

    def test_frequent_lower_weight_than_obligate(self):
        # Obligate g=1.0, Frequent g=0.5 → same IC, obligate match scores higher
        feat_obl = _feat("HP:0001", "Obligate", ic=2.0)
        feat_frq = _feat("HP:0002", "Frequent", ic=2.0)
        score_obl = compute_fit_score(_fit(matched=[_match("HP:0001", feat_obl)]), {}, W)
        score_frq = compute_fit_score(_fit(matched=[_match("HP:0002", feat_frq)]), {}, W)
        assert score_obl > score_frq


# ---------------------------------------------------------------------------
# Cardinality weighting
# ---------------------------------------------------------------------------

def _w(mode: str) -> RescoreWeights:
    return RescoreWeights(
        w_match=1.0, w_partial=0.5, w_miss=0.5, w_unexp=0.3, w_contra=1.0,
        cardinality_mode=mode,
    )


class TestCardinality:
    def test_off_ignores_cardinality(self):
        # Default mode: labelled features score exactly as before.
        feat = _feat("HP:0001", "Obligate", ic=2.0, cardinality="CARDINAL")
        fit = _fit(matched=[_match("HP:0001", feat)])
        assert compute_fit_score(fit, {}, W) == pytest.approx(_sig(2.0))

    def test_cardinal_match_boosted(self):
        # fit_raw = 1.0 * 1.0 * 1.5 * 2.0 = 3.0
        feat = _feat("HP:0001", "Obligate", ic=2.0, cardinality="CARDINAL")
        fit = _fit(matched=[_match("HP:0001", feat)])
        assert compute_fit_score(fit, {}, _w("match")) == pytest.approx(_sig(3.0))

    def test_non_cardinal_match_damped(self):
        # fit_raw = 1.0 * 1.0 * 0.6 * 2.0 = 1.2
        feat = _feat("HP:0001", "Obligate", ic=2.0, cardinality="NON_CARDINAL")
        fit = _fit(matched=[_match("HP:0001", feat)])
        assert compute_fit_score(fit, {}, _w("both")) == pytest.approx(_sig(1.2))

    def test_partial_uses_match_table(self):
        # fit_raw = 0.5 * 1.0 * 1.5 * 2.0 = 1.5
        feat = _feat("HP:0001", "Obligate", ic=2.0, cardinality="CARDINAL")
        fit = _fit(partial=[_match("HP:0001", feat, relation="partial")])
        assert compute_fit_score(fit, {}, _w("match")) == pytest.approx(_sig(1.5))

    def test_cardinal_absent_penalised_harder(self):
        # fit_raw = -0.5 * 1.0 * 2.0 * 4.0 = -4.0
        feat = _feat("HP:0003", "Obligate", ic=4.0, cardinality="CARDINAL")
        fit = _fit(expected_absent=[feat])
        assert compute_fit_score(fit, {}, _w("miss")) == pytest.approx(_sig(-4.0))

    def test_non_cardinal_absent_barely_penalised(self):
        # fit_raw = -0.5 * 1.0 * 0.1 * 4.0 = -0.2
        feat = _feat("HP:0003", "Obligate", ic=4.0, cardinality="NON_CARDINAL")
        fit = _fit(expected_absent=[feat])
        assert compute_fit_score(fit, {}, _w("both")) == pytest.approx(_sig(-0.2))

    def test_match_mode_leaves_misses_alone(self):
        feat = _feat("HP:0003", "Obligate", ic=4.0, cardinality="CARDINAL")
        fit = _fit(expected_absent=[feat])
        assert compute_fit_score(fit, {}, _w("match")) == pytest.approx(_sig(-2.0))

    def test_miss_mode_leaves_matches_alone(self):
        feat = _feat("HP:0001", "Obligate", ic=2.0, cardinality="CARDINAL")
        fit = _fit(matched=[_match("HP:0001", feat)])
        assert compute_fit_score(fit, {}, _w("miss")) == pytest.approx(_sig(2.0))

    def test_unlabelled_feature_neutral(self):
        # e.g. Orphanet-sourced features carry no cardinality → multiplier 1.0
        feat = _feat("HP:0001", "Obligate", ic=2.0)
        fit = _fit(matched=[_match("HP:0001", feat)], expected_absent=[_feat("HP:0003", "Obligate", ic=4.0)])
        assert compute_fit_score(fit, {}, _w("both")) == pytest.approx(_sig(0.0))

    def test_cardinal_match_outranks_non_cardinal(self):
        # Same frequency and IC — cardinality alone decides the order.
        fit_a = _fit(matched=[_match("HP:0001", _feat("HP:0001", "Frequent", 3.0, "CARDINAL"))])
        fit_b = _fit(matched=[_match("HP:0001", _feat("HP:0001", "Frequent", 3.0, "NON_CARDINAL"))])
        candidates = [_rc("B", 0.8, 1, fit_b), _rc("A", 0.8, 2, fit_a)]
        result = rerank(candidates, ic_map={}, weights=_w("match"), mode="prism")
        assert [rc.candidate.disease_id for rc in result] == ["A", "B"]


# ---------------------------------------------------------------------------
# rerank — prism mode (fit_score only)
# ---------------------------------------------------------------------------

class TestRerankPrism:
    def test_better_fit_ranks_first(self):
        feat = _feat("HP:0001", "Obligate", ic=3.0)
        fit_a = _fit(unexplained=[_term("HP:0001")])        # bad fit
        fit_b = _fit(matched=[_match("HP:0001", feat)])     # good fit
        candidates = [
            _rc("A", combined_score=0.95, rank=1, fit=fit_a),
            _rc("B", combined_score=0.50, rank=2, fit=fit_b),
        ]
        result = rerank(candidates, ic_map={"HP:0001": 3.0}, weights=W, mode="prism")
        # B wins on fit despite much lower Exomiser score
        assert result[0].candidate.disease_id == "B"

    def test_new_rank_assigned(self):
        feat = _feat("HP:0001", "Obligate", ic=3.0)
        fit_a = _fit(unexplained=[_term("HP:0001")])
        fit_b = _fit(matched=[_match("HP:0001", feat)])
        candidates = [
            _rc("A", combined_score=0.85, rank=1, fit=fit_a),
            _rc("B", combined_score=0.84, rank=2, fit=fit_b),
        ]
        result = rerank(candidates, ic_map={"HP:0001": 3.0}, weights=W, mode="prism")
        assert result[0].new_rank == 1
        assert result[1].new_rank == 2

    def test_fit_score_populated(self):
        fit = _fit()
        candidates = [_rc("A", combined_score=0.85, rank=1, fit=fit)]
        result = rerank(candidates, ic_map={}, weights=W, mode="prism")
        assert result[0].fit.fit_score is not None

    def test_fit_score_between_zero_and_one(self):
        feat = _feat("HP:0001", "Obligate", ic=5.0)
        fit = _fit(matched=[_match("HP:0001", feat)])
        result = rerank([_rc("A", 0.85, 1, fit)], ic_map={}, weights=W, mode="prism")
        score = result[0].fit.fit_score
        assert score is not None
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# rerank — blended mode (α·exo + (1-α)·fit)
# ---------------------------------------------------------------------------

class TestRerankBlended:
    def test_blended_combines_scores(self):
        # A: exo=0.9 (high), fit=very bad;  B: exo=0.5 (low), fit=very good
        # With low alpha, B's strong fit overcomes A's Exomiser lead
        feat = _feat("HP:0001", "Obligate", ic=10.0)
        fit_a = _fit(contradictions=[_feat("HP:0002", "Excluded", ic=3.0)] * 5)
        fit_b = _fit(matched=[_match("HP:0001", feat)])
        candidates = [
            _rc("A", combined_score=0.9, rank=1, fit=fit_a),
            _rc("B", combined_score=0.5, rank=2, fit=fit_b),
        ]
        w = RescoreWeights(alpha=0.3)  # low alpha → fit dominates
        result = rerank(candidates, ic_map={}, weights=w, mode="blended")
        assert result[0].candidate.disease_id == "B"

    def test_high_exo_wins_when_alpha_high(self):
        # A: exo=0.95, weak match (low-IC);  B: exo=0.50, strong match (high-IC)
        # With alpha=0.9 the Exomiser advantage should keep A at rank 1
        feat_a = _feat("HP:0001", "Occasional", ic=0.5)  # weak match
        feat_b = _feat("HP:0002", "Obligate", ic=3.0)    # strong match
        fit_a = _fit(matched=[_match("HP:0001", feat_a)])
        fit_b = _fit(matched=[_match("HP:0002", feat_b)])
        candidates = [
            _rc("A", combined_score=0.95, rank=1, fit=fit_a),
            _rc("B", combined_score=0.50, rank=2, fit=fit_b),
        ]
        w = RescoreWeights(alpha=0.9)  # high alpha → Exomiser dominates
        result = rerank(candidates, ic_map={}, weights=w, mode="blended")
        assert result[0].candidate.disease_id == "A"

    def test_rationale_contains_fit_score(self):
        fit = _fit()
        result = rerank([_rc("A", 0.85, 1, fit)], ic_map={}, weights=W)
        assert "fit=" in result[0].rationale