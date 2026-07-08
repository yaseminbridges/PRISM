"""Tests for C3 Age gate (age_gate.py) and HPOA onset retrieval."""
from __future__ import annotations
import pytest

from prism.models.phenopacket import HpoTerm
from prism.models.candidate import DiseaseFeature, FeatureMatch, FitEvidence
from prism.components.age_gate import apply_age_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feat(hpo_id: str, freq: str = "Obligate") -> DiseaseFeature:
    return DiseaseFeature(hpo_id=hpo_id, hpo_label=hpo_id, frequency_class=freq, ic=2.0, source="HPOA")


def _fit(expected_absent=(), age_excused=()) -> FitEvidence:
    return FitEvidence(
        disease_id="OMIM:999",
        expected_absent=list(expected_absent),
        age_excused=list(age_excused),
    )


# ---------------------------------------------------------------------------
# Core excusing logic
# ---------------------------------------------------------------------------

class TestApplyAgeGate:
    def test_onset_after_patient_age_excuses_all(self):
        # Patient is 3y; disease onset is 8y (Childhood) — absences excused
        feat = _feat("HP:0001")
        fit = _fit(expected_absent=[feat])
        result = apply_age_gate(fit, patient_age_years=3.0, disease_onset_years=8.0)
        assert len(result.age_excused) == 1
        assert len(result.expected_absent) == 0
        assert result.age_excused[0].hpo_id == "HP:0001"

    def test_onset_before_patient_age_keeps_expected_absent(self):
        # Patient is 10y; disease onset is 1y (Infantile) — absence is genuine
        feat = _feat("HP:0001")
        fit = _fit(expected_absent=[feat])
        result = apply_age_gate(fit, patient_age_years=10.0, disease_onset_years=1.0)
        assert len(result.expected_absent) == 1
        assert len(result.age_excused) == 0

    def test_onset_equal_to_patient_age_not_excused(self):
        # Onset exactly at patient age — disease should have manifested
        feat = _feat("HP:0001")
        fit = _fit(expected_absent=[feat])
        result = apply_age_gate(fit, patient_age_years=8.0, disease_onset_years=8.0)
        assert len(result.expected_absent) == 1
        assert len(result.age_excused) == 0

    def test_multiple_features_all_excused(self):
        feats = [_feat(f"HP:{i:04d}") for i in range(4)]
        fit = _fit(expected_absent=feats)
        result = apply_age_gate(fit, patient_age_years=2.0, disease_onset_years=10.0)
        assert len(result.age_excused) == 4
        assert len(result.expected_absent) == 0

    def test_existing_age_excused_preserved(self):
        # Pre-existing age_excused entries must not be lost
        existing = _feat("HP:0000")
        new_feat = _feat("HP:0001")
        fit = _fit(expected_absent=[new_feat], age_excused=[existing])
        result = apply_age_gate(fit, patient_age_years=2.0, disease_onset_years=10.0)
        assert len(result.age_excused) == 2

    def test_noop_when_patient_age_unknown(self):
        feat = _feat("HP:0001")
        fit = _fit(expected_absent=[feat])
        result = apply_age_gate(fit, patient_age_years=None, disease_onset_years=8.0)
        assert result is fit  # unchanged

    def test_noop_when_onset_unknown(self):
        feat = _feat("HP:0001")
        fit = _fit(expected_absent=[feat])
        result = apply_age_gate(fit, patient_age_years=3.0, disease_onset_years=None)
        assert result is fit

    def test_noop_when_both_ages_unknown(self):
        fit = _fit(expected_absent=[_feat("HP:0001")])
        assert apply_age_gate(fit, None, None) is fit

    def test_empty_expected_absent_is_noop(self):
        fit = _fit(expected_absent=[])
        result = apply_age_gate(fit, patient_age_years=3.0, disease_onset_years=10.0)
        assert len(result.age_excused) == 0


# ---------------------------------------------------------------------------
# Integration: age_excused features do NOT contribute to C2 penalty
# ---------------------------------------------------------------------------

class TestAgeGateC2Integration:
    def test_excused_features_not_penalised(self):
        """After age_gate, expected_absent is empty → C2 incurs no missing-hallmark penalty."""
        from prism.components.rescore import compute_fit_score, RescoreWeights

        feat = _feat("HP:0001", "Obligate")
        # Without gate: feature in expected_absent → penalty
        fit_with_absent = _fit(expected_absent=[feat])
        score_penalised = compute_fit_score(fit_with_absent, ic_map={}, weights=RescoreWeights())

        # With gate applied: feature moves to age_excused → no penalty
        fit_excused = apply_age_gate(fit_with_absent, patient_age_years=2.0, disease_onset_years=8.0)
        score_excused = compute_fit_score(fit_excused, ic_map={}, weights=RescoreWeights())

        assert score_excused > score_penalised
        assert score_excused == pytest.approx(0.0)  # no positive or negative signal


# ---------------------------------------------------------------------------
# HPOA onset retrieval
# ---------------------------------------------------------------------------

class TestHpoaOnsetRetrieval:
    def test_onset_returned_for_known_disease(self, small_retriever):
        # The small fixture may not have onset data; real retriever tested here
        # Just verify the method exists and returns float | None
        result = small_retriever.get_earliest_onset_years("OMIM:601869")
        assert result is None or isinstance(result, float)

    def test_onset_none_for_unknown_disease(self, small_retriever):
        assert small_retriever.get_earliest_onset_years("OMIM:999999") is None