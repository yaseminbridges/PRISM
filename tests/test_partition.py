"""Tests for C1 Partition (partition.py).

Uses the small HPO fixture and hand-constructed DiseaseProfiles so tests
are fast and fully deterministic (MockLLMClient).
"""
from __future__ import annotations
import pytest

from prism.models.phenopacket import HpoTerm, PatientPhenotype
from prism.models.candidate import DiseaseFeature, FitEvidence
from prism.knowledge.hpoa.tool import DiseaseProfile
from prism.components.partition import partition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patient(*term_ids: str, excluded: tuple[str, ...] = ()) -> PatientPhenotype:
    """Build a minimal PatientPhenotype from HPO IDs using small-fixture labels."""
    _labels = {
        "HP:0000399": "Prelingual sensorineural hearing impairment",
        "HP:0000407": "Sensorineural hearing impairment",
        "HP:0000598": "Abnormality of the ear",
        "HP:0001363": "Craniosynostosis",
        "HP:0011304": "Broad thumb",
        "HP:0001161": "Camptodactyly",
        "HP:0000238": "Hydrocephalus",
        "HP:0000316": "Hypertelorism",
        "HP:0001249": "Intellectual disability",
    }
    observed = [HpoTerm(id=t, label=_labels.get(t, t)) for t in term_ids]
    excl = [HpoTerm(id=t, label=_labels.get(t, t), excluded=True) for t in excluded]
    return PatientPhenotype(
        subject_id="test", sex="unknown",
        observed_terms=observed, excluded_terms=excl,
    )


def _feat(hpo_id: str, label: str, freq: str = "Frequent") -> DiseaseFeature:
    return DiseaseFeature(hpo_id=hpo_id, hpo_label=label, frequency_class=freq, ic=1.0, source="HPOA")


def _profile(disease_id: str, features, excluded_features=None) -> DiseaseProfile:
    return DiseaseProfile(
        disease_id=disease_id,
        features=list(features),
        excluded_features=list(excluded_features or []),
    )


IC_MAP: dict[str, float] = {}  # test features use ic=1.0; gap always ≤ MAX_IC_GAP


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_exact_goes_to_matched(self, small_graph):
        patient = _patient("HP:0000407")
        profile = _profile("OMIM:999", [_feat("HP:0000407", "Sensorineural hearing impairment")])
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.matched) == 1
        assert fit.matched[0].relation == "exact"
        assert fit.matched[0].patient_term.id == "HP:0000407"

    def test_exact_not_in_unexplained(self, small_graph):
        patient = _patient("HP:0000407")
        profile = _profile("OMIM:999", [_feat("HP:0000407", "Sensorineural hearing impairment")])
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.unexplained) == 0


# ---------------------------------------------------------------------------
# Subsumed match (patient more specific than disease feature)
# ---------------------------------------------------------------------------

class TestSubsumedMatch:
    def test_patient_specific_subsumed(self, small_graph):
        # Disease expects HP:0000407 (SHL); patient has HP:0000399 (Prelingual SHL — child)
        patient = _patient("HP:0000399")
        profile = _profile("OMIM:999", [_feat("HP:0000407", "Sensorineural hearing impairment")])
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.matched) == 1
        assert fit.matched[0].relation == "subsumed"

    def test_subsumed_not_unexplained(self, small_graph):
        patient = _patient("HP:0000399")
        profile = _profile("OMIM:999", [_feat("HP:0000407", "Sensorineural hearing impairment")])
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.unexplained) == 0


# ---------------------------------------------------------------------------
# Partial match (patient broader than disease feature) — goes through MockLLM
# ---------------------------------------------------------------------------

class TestPartialMatch:
    def test_patient_broader_goes_to_partial(self, small_graph):
        # Disease expects HP:0000399 (Prelingual SHL); patient has HP:0000407 (SHL — parent)
        patient = _patient("HP:0000407")
        profile = _profile("OMIM:999", [_feat("HP:0000399", "Prelingual sensorineural hearing impairment")])
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.partial) == 1
        assert fit.partial[0].relation == "partial"

    def test_partial_not_unexplained(self, small_graph):
        patient = _patient("HP:0000407")
        profile = _profile("OMIM:999", [_feat("HP:0000399", "Prelingual sensorineural hearing impairment")])
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.unexplained) == 0


# ---------------------------------------------------------------------------
# Unexplained
# ---------------------------------------------------------------------------

class TestUnexplained:
    def test_unrelated_term_unexplained(self, small_graph):
        # Patient has craniosynostosis; disease only has hearing terms
        patient = _patient("HP:0001363")
        profile = _profile("OMIM:999", [_feat("HP:0000407", "Sensorineural hearing impairment")])
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.unexplained) == 1
        assert fit.unexplained[0].id == "HP:0001363"

    def test_multiple_terms_split_correctly(self, small_graph):
        patient = _patient("HP:0000407", "HP:0001363")  # hearing + cranio
        profile = _profile("OMIM:999", [_feat("HP:0000407", "Sensorineural hearing impairment")])
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.matched) == 1
        assert len(fit.unexplained) == 1


# ---------------------------------------------------------------------------
# Expected absent
# ---------------------------------------------------------------------------

class TestExpectedAbsent:
    def test_excluded_disease_feature_expected_absent(self, small_graph):
        # Patient explicitly excluded HP:0000407; disease has it as Obligate
        patient = _patient(excluded=("HP:0000407",))
        profile = _profile("OMIM:999", [_feat("HP:0000407", "Sensorineural hearing impairment", "Obligate")])
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.expected_absent) == 1
        assert fit.expected_absent[0].hpo_id == "HP:0000407"

    def test_ancestor_exclusion_covers_descendant_feature(self, small_graph):
        # Patient excluded HP:0000598 (Ear abnormality); disease has HP:0000407 (SHL — child)
        patient = _patient(excluded=("HP:0000598",))
        profile = _profile("OMIM:999", [_feat("HP:0000407", "Sensorineural hearing impairment", "Obligate")])
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.expected_absent) == 1

    def test_unassessed_absent_feature_not_expected_absent(self, small_graph):
        # Patient just doesn't mention HP:0000407 — it is unassessed, not expected_absent
        patient = _patient("HP:0001363")  # only cranio, no hearing terms at all
        profile = _profile("OMIM:999", [
            _feat("HP:0000407", "Sensorineural hearing impairment", "Obligate"),
            _feat("HP:0001363", "Craniosynostosis"),
        ])
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.expected_absent) == 0  # HP:0000407 not assessed, not excluded


# ---------------------------------------------------------------------------
# Contradictions
# ---------------------------------------------------------------------------

class TestContradictions:
    def test_disease_excluded_feature_patient_has(self, small_graph):
        # Disease says HP:0001161 (Camptodactyly) should be absent; patient has it
        patient = _patient("HP:0001161")
        excl_feat = _feat("HP:0001161", "Camptodactyly", "Excluded")
        profile = _profile(
            "OMIM:101600",
            [_feat("HP:0001363", "Craniosynostosis", "Obligate")],
            excluded_features=[excl_feat],
        )
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.contradictions) == 1
        assert fit.contradictions[0].hpo_id == "HP:0001161"

    def test_patient_descendant_triggers_contradiction(self, small_graph):
        # Disease excludes HP:0000598 (Ear abnormality); patient has HP:0000407 (SHL — child)
        patient = _patient("HP:0000407")
        excl_feat = _feat("HP:0000598", "Abnormality of the ear", "Excluded")
        profile = _profile(
            "OMIM:999",
            [_feat("HP:0001363", "Craniosynostosis")],
            excluded_features=[excl_feat],
        )
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.contradictions) == 1

    def test_no_contradiction_when_patient_lacks_excluded_feature(self, small_graph):
        excl_feat = _feat("HP:0001161", "Camptodactyly", "Excluded")
        patient = _patient("HP:0001363")  # no camptodactyly
        profile = _profile(
            "OMIM:101600",
            [_feat("HP:0001363", "Craniosynostosis")],
            excluded_features=[excl_feat],
        )
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert len(fit.contradictions) == 0


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_exact_match_provenance_contains_disease_id(self, small_graph):
        patient = _patient("HP:0000407")
        profile = _profile("OMIM:601869", [_feat("HP:0000407", "Sensorineural hearing impairment")])
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert "OMIM:601869" in fit.matched[0].provenance

    def test_partial_match_provenance_contains_ic_gap(self, small_graph):
        patient = _patient("HP:0000407")
        profile = _profile("OMIM:999", [_feat("HP:0000399", "Prelingual SHL")])
        fit = partition(patient, profile, small_graph, IC_MAP)
        assert "IC gap" in fit.partial[0].provenance