"""Tests for C4 Disambiguator (disambiguate.py).

Scenario: Pfeiffer / Crouzon / Apert — three craniosynostosis syndromes with high
profile overlap but distinct discriminating features.

  Pfeiffer (OMIM:101600): craniosynostosis + broad thumb; excluded: camptodactyly
  Crouzon  (OMIM:123500): craniosynostosis + hypertelorism
  Apert    (OMIM:101200): craniosynostosis + camptodactyly

Patient (Pfeiffer case): craniosynostosis, broad thumb, hydrocephalus;
                          camptodactyly explicitly excluded.

Expected: C4 triggers (Jaccard ≈ 0.33 for all pairs ≥ 0.3 threshold),
          best_supported = OMIM:101600 (Pfeiffer).
"""
from __future__ import annotations
import pytest

from prism.models.phenopacket import HpoTerm, PatientPhenotype
from prism.models.candidate import DiseaseFeature, FitEvidence
from prism.models.exomiser import ExomiserCandidate, Variant
from prism.models.report import ReRankedCandidate
from prism.knowledge.hpoa.tool import DiseaseProfile
from prism.reasoning.llm import MockLLMClient
from prism.components.disambiguate import (
    disambiguate,
    _jaccard,
    _unique_features,
    _patient_stance,
)


# ---------------------------------------------------------------------------
# HPO IDs used throughout
# ---------------------------------------------------------------------------
CRANIOSYNOSTOSIS = "HP:0001363"
BROAD_THUMB      = "HP:0011304"
CAMPTODACTYLY    = "HP:0001161"
HYPERTELORISM    = "HP:0000316"
HYDROCEPHALUS    = "HP:0000238"


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _feat(hpo_id: str, label: str = "", ic: float = 2.0) -> DiseaseFeature:
    return DiseaseFeature(
        hpo_id=hpo_id, hpo_label=label or hpo_id,
        frequency_class="Frequent", ic=ic, source="test",
    )


def _excl_feat(hpo_id: str, label: str = "") -> DiseaseFeature:
    return DiseaseFeature(
        hpo_id=hpo_id, hpo_label=label or hpo_id,
        frequency_class="Excluded", ic=1.0, source="test",
    )


def _profile(disease_id: str, feat_ids: list[str], excl_ids: list[str] = ()) -> DiseaseProfile:
    return DiseaseProfile(
        disease_id=disease_id,
        features=[_feat(hid) for hid in feat_ids],
        excluded_features=[_excl_feat(hid) for hid in excl_ids],
    )


def _candidate(disease_id: str, name: str, rank: int = 1, fit_score: float = 0.5) -> ReRankedCandidate:
    exo = ExomiserCandidate(
        gene_symbol="TEST",
        disease_id=disease_id,
        disease_name=name,
        moi=None,
        exomiser_rank=rank,
        combined_score=0.9,
        phenotype_score=0.8,
        variant_score=0.7,
        variants=[],
    )
    fit = FitEvidence(disease_id=disease_id, fit_score=fit_score)
    return ReRankedCandidate(candidate=exo, fit=fit, old_rank=rank, new_rank=rank, rationale="")


def _pfeiffer_patient() -> PatientPhenotype:
    return PatientPhenotype(
        subject_id="pfeiffer-test",
        sex="MALE",
        age_years=None,
        observed_terms=[
            HpoTerm(id=CRANIOSYNOSTOSIS, label="Craniosynostosis"),
            HpoTerm(id=BROAD_THUMB,      label="Broad thumb"),
            HpoTerm(id=HYDROCEPHALUS,    label="Hydrocephalus"),
        ],
        excluded_terms=[
            HpoTerm(id=CAMPTODACTYLY, label="Camptodactyly", excluded=True),
        ],
    )


# ---------------------------------------------------------------------------
# _jaccard
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical_sets(self):
        assert _jaccard({"A", "B"}, {"A", "B"}) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        assert _jaccard({"A"}, {"B"}) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # |A∩B| = 1, |A∪B| = 3
        assert _jaccard({"A", "B"}, {"A", "C"}) == pytest.approx(1 / 3)

    def test_empty_both(self):
        assert _jaccard(set(), set()) == pytest.approx(0.0)

    def test_one_empty(self):
        assert _jaccard({"A"}, set()) == pytest.approx(0.0)

    def test_cranio_pair(self):
        # Pfeiffer vs Crouzon: share craniosynostosis only
        pfeiffer = {CRANIOSYNOSTOSIS, BROAD_THUMB}
        crouzon = {CRANIOSYNOSTOSIS, HYPERTELORISM}
        j = _jaccard(pfeiffer, crouzon)
        assert j == pytest.approx(1 / 3)
        assert j >= 0.3


# ---------------------------------------------------------------------------
# _unique_features
# ---------------------------------------------------------------------------

class TestUniqueFeatures:
    def test_all_unique(self):
        profiles = {
            "D1": _profile("D1", ["A", "B"]),
            "D2": _profile("D2", ["A", "C"]),
        }
        result = _unique_features(profiles)
        assert result["D1"] == {"B"}
        assert result["D2"] == {"C"}

    def test_shared_features_excluded(self):
        profiles = {
            "D1": _profile("D1", ["A", "B", "C"]),
            "D2": _profile("D2", ["A", "B", "D"]),
        }
        result = _unique_features(profiles)
        assert result["D1"] == {"C"}
        assert result["D2"] == {"D"}

    def test_three_way_unique(self):
        pfeiffer = _profile("OMIM:101600", [CRANIOSYNOSTOSIS, BROAD_THUMB])
        crouzon  = _profile("OMIM:123500", [CRANIOSYNOSTOSIS, HYPERTELORISM])
        apert    = _profile("OMIM:101200", [CRANIOSYNOSTOSIS, CAMPTODACTYLY])
        profiles = {
            "OMIM:101600": pfeiffer,
            "OMIM:123500": crouzon,
            "OMIM:101200": apert,
        }
        result = _unique_features(profiles)
        assert result["OMIM:101600"] == {BROAD_THUMB}
        assert result["OMIM:123500"] == {HYPERTELORISM}
        assert result["OMIM:101200"] == {CAMPTODACTYLY}

    def test_no_unique_if_all_shared(self):
        profiles = {
            "D1": _profile("D1", ["A"]),
            "D2": _profile("D2", ["A"]),
        }
        result = _unique_features(profiles)
        assert result["D1"] == set()
        assert result["D2"] == set()


# ---------------------------------------------------------------------------
# _patient_stance  (no graph subsumption needed for exact matches)
# ---------------------------------------------------------------------------

class TestPatientStance:
    """Exact-match cases that need no HPO graph."""

    def _mini_graph(self):
        """Return a minimal stub HPOGraph-like object with no real subsumption."""
        from unittest.mock import MagicMock
        g = MagicMock()
        g.subsumes.return_value = False
        return g

    def test_observed_term_supports(self):
        patient = _pfeiffer_patient()
        g = self._mini_graph()
        assert _patient_stance(BROAD_THUMB, patient, g) == "supports"

    def test_excluded_term_against(self):
        patient = _pfeiffer_patient()
        g = self._mini_graph()
        assert _patient_stance(CAMPTODACTYLY, patient, g) == "against"

    def test_silent_term(self):
        patient = _pfeiffer_patient()
        g = self._mini_graph()
        assert _patient_stance(HYPERTELORISM, patient, g) == "silent"

    def test_subsumes_observed_supports(self, small_graph):
        """Ancestor of an observed term is still 'supports'."""
        patient = PatientPhenotype(
            subject_id="x", sex="UNKNOWN", age_years=None,
            observed_terms=[HpoTerm(id=CRANIOSYNOSTOSIS, label="CS")],
            excluded_terms=[],
        )
        # CRANIOSYNOSTOSIS has ancestors in small_graph — query the parent, expect supports
        parents = list(small_graph.ancestors(CRANIOSYNOSTOSIS))
        if not parents:
            pytest.skip("No ancestors of HP:0001363 in small fixture")
        parent_id = parents[0]
        assert _patient_stance(parent_id, patient, small_graph) == "supports"


# ---------------------------------------------------------------------------
# disambiguate — full integration
# ---------------------------------------------------------------------------

class TestDisambiguate:
    """Pfeiffer / Crouzon / Apert three-way overlap scenario."""

    def _scenario(self, scores: dict[str, float] | None = None):
        """Build candidates, profiles, and patient for the craniosynostosis scenario."""
        default_scores = {
            "OMIM:101600": 0.7,
            "OMIM:123500": 0.6,
            "OMIM:101200": 0.5,
        }
        s = scores or default_scores
        candidates = [
            _candidate("OMIM:101600", "Pfeiffer syndrome",    rank=1, fit_score=s["OMIM:101600"]),
            _candidate("OMIM:123500", "Crouzon syndrome",     rank=2, fit_score=s["OMIM:123500"]),
            _candidate("OMIM:101200", "Apert syndrome",       rank=3, fit_score=s["OMIM:101200"]),
        ]
        profiles = {
            "OMIM:101600": DiseaseProfile(
                disease_id="OMIM:101600",
                features=[_feat(CRANIOSYNOSTOSIS, "Craniosynostosis"), _feat(BROAD_THUMB, "Broad thumb")],
                excluded_features=[_excl_feat(CAMPTODACTYLY, "Camptodactyly")],
            ),
            "OMIM:123500": _profile("OMIM:123500", [CRANIOSYNOSTOSIS, HYPERTELORISM]),
            "OMIM:101200": _profile("OMIM:101200", [CRANIOSYNOSTOSIS, CAMPTODACTYLY]),
        }
        patient = _pfeiffer_patient()
        return candidates, profiles, patient

    def _mini_graph(self):
        from unittest.mock import MagicMock
        g = MagicMock()
        g.subsumes.return_value = False
        return g

    def test_triggers_on_similar_profiles(self):
        candidates, profiles, patient = self._scenario()
        result = disambiguate(
            candidates, profiles, patient, self._mini_graph(), MockLLMClient(),
            similarity_threshold=0.3,
        )
        assert result is not None

    def test_no_trigger_when_all_dissimilar(self):
        candidates = [
            _candidate("OMIM:101600", "Pfeiffer", rank=1, fit_score=0.7),
            _candidate("OMIM:999999", "Unrelated", rank=2, fit_score=0.5),
        ]
        profiles = {
            "OMIM:101600": _profile("OMIM:101600", ["A", "B", "C"]),
            "OMIM:999999": _profile("OMIM:999999", ["X", "Y", "Z"]),
        }
        patient = _pfeiffer_patient()
        result = disambiguate(
            candidates, profiles, patient, self._mini_graph(), MockLLMClient(),
            similarity_threshold=0.3,
        )
        assert result is None

    def test_best_supported_is_pfeiffer(self):
        candidates, profiles, patient = self._scenario()
        result = disambiguate(
            candidates, profiles, patient, self._mini_graph(), MockLLMClient(),
        )
        assert result is not None
        assert result.best_supported == "OMIM:101600"

    def test_candidate_disease_ids(self):
        candidates, profiles, patient = self._scenario()
        result = disambiguate(
            candidates, profiles, patient, self._mini_graph(), MockLLMClient(),
        )
        assert result is not None
        assert set(result.candidate_disease_ids) == {"OMIM:101600", "OMIM:123500", "OMIM:101200"}

    def test_discriminating_features_correct(self):
        candidates, profiles, patient = self._scenario()
        result = disambiguate(
            candidates, profiles, patient, self._mini_graph(), MockLLMClient(),
        )
        assert result is not None
        disc = set(result.discriminating_features)
        assert BROAD_THUMB in disc       # Pfeiffer unique
        assert HYPERTELORISM in disc     # Crouzon unique
        assert CAMPTODACTYLY in disc     # Apert unique
        assert CRANIOSYNOSTOSIS not in disc   # shared — not discriminating

    def test_rationale_contains_best_supported(self):
        candidates, profiles, patient = self._scenario()
        result = disambiguate(
            candidates, profiles, patient, self._mini_graph(), MockLLMClient(),
        )
        assert result is not None
        assert "OMIM:101600" in result.rationale

    def test_rationale_mentions_silent_inference(self):
        """Silent discriminator (hypertelorism for Crouzon) must appear in rationale."""
        candidates, profiles, patient = self._scenario()
        result = disambiguate(
            candidates, profiles, patient, self._mini_graph(), MockLLMClient(),
        )
        assert result is not None
        assert "HP:0000316" in result.rationale  # hypertelorism HPO ID

    def test_no_best_when_tied(self):
        # Patient has craniosynostosis only — every discriminator is silent
        candidates, profiles, patient = self._scenario()
        patient_cranio_only = PatientPhenotype(
            subject_id="ambiguous",
            sex="UNKNOWN",
            age_years=None,
            observed_terms=[HpoTerm(id=CRANIOSYNOSTOSIS, label="Craniosynostosis")],
            excluded_terms=[],
        )
        result = disambiguate(
            candidates, profiles, patient_cranio_only, self._mini_graph(), MockLLMClient(),
        )
        # All scores start at 0 and patient is silent on all discriminators
        # → no strict winner
        assert result is not None
        assert result.best_supported is None

    def test_negative_score_candidates_not_triggered(self):
        # Candidates with fit_score <= 0 should be excluded from trigger check
        candidates = [
            _candidate("OMIM:101600", "Pfeiffer", rank=1, fit_score=0.7),
            _candidate("OMIM:123500", "Crouzon",  rank=2, fit_score=-0.1),
        ]
        profiles = {
            "OMIM:101600": _profile("OMIM:101600", [CRANIOSYNOSTOSIS, BROAD_THUMB]),
            "OMIM:123500": _profile("OMIM:123500", [CRANIOSYNOSTOSIS, HYPERTELORISM]),
        }
        patient = _pfeiffer_patient()
        result = disambiguate(
            candidates, profiles, patient, self._mini_graph(), MockLLMClient(),
            similarity_threshold=0.3,
        )
        # Only one positive candidate → no pair → no trigger
        assert result is None

    def test_no_trigger_below_threshold(self):
        candidates, profiles, patient = self._scenario()
        result = disambiguate(
            candidates, profiles, patient, self._mini_graph(), MockLLMClient(),
            similarity_threshold=0.99,  # unreachable threshold
        )
        assert result is None

    def test_missing_profile_pair_skipped(self):
        """Candidate without a profile should not crash, just be ignored."""
        candidates = [
            _candidate("OMIM:101600", "Pfeiffer", rank=1, fit_score=0.7),
            _candidate("OMIM:123500", "Crouzon",  rank=2, fit_score=0.6),
        ]
        # Only Pfeiffer profile provided; Crouzon missing
        profiles = {
            "OMIM:101600": _profile("OMIM:101600", [CRANIOSYNOSTOSIS, BROAD_THUMB]),
        }
        patient = _pfeiffer_patient()
        result = disambiguate(
            candidates, profiles, patient, self._mini_graph(), MockLLMClient(),
        )
        # Can't form a pair without Crouzon profile → no trigger
        assert result is None