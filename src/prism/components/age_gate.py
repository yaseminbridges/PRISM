"""C3 — Age gate: excuse late-onset absences from the missing-hallmark penalty.

A cardinal feature absent in a patient younger than the disease's typical onset
is not a red flag — it simply hasn't manifested yet.  Moving it to `age_excused`
zeroes out the C2 penalty for that feature while flagging it for monitoring.

Usage in pipeline:
  onset_years = _resolve_onset(disease_id, hpoa, orpha)
  fit = apply_age_gate(fit, patient.age_years, onset_years)
  fit = rescore(fit, ...)   # expected_absent now only contains genuine absences
"""

from prism.models.candidate import FitEvidence


def apply_age_gate(
    fit: FitEvidence,
    patient_age_years: float | None,
    disease_onset_years: float | None,
) -> FitEvidence:
    """Move expected_absent features to age_excused when disease onset > patient age.

    If either age is unknown the gate is a no-op — we never excuse on missing data.

    disease_onset_years is the EARLIEST typical onset for the disease (from HPOA
    or Orphanet).  If the disease should have manifested by the patient's age
    (onset <= patient age), expected_absent features stay and contribute the C2
    penalty.  If onset is still in the future, those absences are developmentally
    expected and are excused.
    """
    if patient_age_years is None or disease_onset_years is None:
        return fit

    if disease_onset_years <= patient_age_years:
        return fit  # disease should have manifested — absences are genuine

    # Onset is still in the patient's future — excuse all expected_absent
    return fit.model_copy(update={
        "age_excused": fit.age_excused + fit.expected_absent,
        "expected_absent": [],
    })