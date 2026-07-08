from __future__ import annotations
import pytest
from pydantic import ValidationError
from prism.models.phenopacket import HpoTerm, PatientPhenotype


def _make(observed=None, excluded=None):
    return PatientPhenotype(
        subject_id="x",
        sex="male",
        age_years=None,
        observed_terms=observed or [],
        excluded_terms=excluded or [],
    )


def test_valid_patient():
    p = _make(
        observed=[HpoTerm(id="HP:0000407", label="Sensorineural hearing impairment")],
        excluded=[HpoTerm(id="HP:0000316", label="Hypertelorism", excluded=True)],
    )
    assert len(p.observed_terms) == 1
    assert len(p.excluded_terms) == 1


def test_observed_term_must_not_be_excluded():
    with pytest.raises(ValidationError, match="observed_terms must not contain excluded"):
        _make(observed=[HpoTerm(id="HP:0000407", label="X", excluded=True)])


def test_excluded_term_must_have_excluded_true():
    with pytest.raises(ValidationError, match="excluded_terms must all have excluded=True"):
        _make(excluded=[HpoTerm(id="HP:0000407", label="X", excluded=False)])


def test_no_overlap_between_observed_and_excluded():
    with pytest.raises(ValidationError, match="both observed and excluded"):
        _make(
            observed=[HpoTerm(id="HP:0000407", label="X")],
            excluded=[HpoTerm(id="HP:0000407", label="X", excluded=True)],
        )


def test_empty_patient_is_valid():
    p = _make()
    assert p.observed_terms == []
    assert p.excluded_terms == []