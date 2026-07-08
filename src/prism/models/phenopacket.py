"""Patient input data models.

This file defines the two core structures that describe a patient:

  HpoTerm         — a single HPO term, flagged as either observed or explicitly excluded.
  PatientPhenotype — the full patient record: HPO terms, sex, and age.

IMPORTANT DISTINCTION — absent vs excluded:
  A term that is not listed is "not assessed" (we simply don't know).
  A term in excluded_terms means a clinician actively checked for it and it was NOT present.
  This matters downstream: only excluded terms generate the expected_absent signal in C1.
  Never conflate "missing from the phenopacket" with "absent in the patient".
"""
from typing import Self
from pydantic import BaseModel, model_validator


class HpoTerm(BaseModel):
    id: str
    label: str
    onset_age_years: float | None = None
    excluded: bool = False


class PatientPhenotype(BaseModel):
    subject_id: str
    sex: str
    age_years: float | None = None
    observed_terms: list[HpoTerm]
    excluded_terms: list[HpoTerm]

    @model_validator(mode="after")
    def _check_absence_invariant(self) -> Self:
        for t in self.observed_terms:
            if t.excluded:
                raise ValueError(
                    f"observed_terms must not contain excluded terms: {t.id}"
                )
        for t in self.excluded_terms:
            if not t.excluded:
                raise ValueError(
                    f"excluded_terms must all have excluded=True: {t.id}"
                )
        observed_ids = {t.id for t in self.observed_terms}
        overlap = observed_ids & {t.id for t in self.excluded_terms}
        if overlap:
            raise ValueError(
                f"Terms cannot be both observed and excluded: {overlap}"
            )
        return self
