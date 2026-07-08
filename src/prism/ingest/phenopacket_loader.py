"""Phenopacket ingestion: parse a GA4GH phenopacket JSON into a PatientPhenotype.

A phenopacket is the standard format for encoding a patient's clinical data.
This loader handles both individual Phenopackets and Family phenopackets (extracting
the proband from the family).

Key parsing decisions:
- Sex is mapped from the protobuf enum integer to a string ("male"/"female"/"unknown").
- Phenotypic features with `excluded=True` are split into a separate `excluded_terms`
  list — these are terms a clinician actively checked for and did not find.
- If a term appears as both observed and excluded (a data quality issue), it is treated
  as observed (presence is stronger evidence than recorded absence) and a warning is printed.
- Age is read from subject.time_at_last_encounter.age.iso8601duration (e.g. "P10Y", "P6M",
  "P2Y3M"). Returns None if the field is absent — age gate will simply be skipped.
"""
import re
import sys
from pathlib import Path
from phenopackets import Family
from prism.models.phenopacket import HpoTerm, PatientPhenotype
from pheval.utils.phenopacket_utils import phenopacket_reader, PhenopacketUtil

_SEX_MAP = {0: "unknown", 1: "female", 2: "male", 3: "unknown"}

_ISO8601_RE = re.compile(
    r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?",
    re.IGNORECASE,
)


def _parse_iso8601_years(duration: str) -> float | None:
    """Convert an ISO 8601 duration string to a float number of years.

    Handles years, months, weeks, and days. Time components (hours, minutes,
    seconds) are ignored — not relevant for age-gate comparisons.
    Returns None if the string doesn't match the expected format.
    """
    m = _ISO8601_RE.match(duration)
    if not m:
        return None
    years  = int(m.group(1) or 0)
    months = int(m.group(2) or 0)
    weeks  = int(m.group(3) or 0)
    days   = int(m.group(4) or 0)
    return years + months / 12 + weeks * 7 / 365.25 + days / 365.25


def _extract_age_years(subject) -> float | None:
    """Pull age_years from subject.time_at_last_encounter, or return None."""
    try:
        iso = subject.time_at_last_encounter.age.iso8601duration
        if iso:
            return _parse_iso8601_years(iso)
    except AttributeError:
        pass
    return None


def load_phenopacket(phenopacket_path: Path) -> PatientPhenotype:
    pp = phenopacket_reader(phenopacket_path)
    if isinstance(pp, Family):
        pp = pp.proband
    subject = pp.subject
    sex = _SEX_MAP.get(subject.sex, "unknown")  # type: ignore[arg-type]

    observed = [HpoTerm(id=pf.type.id, label=pf.type.label, excluded=False) for pf in
                PhenopacketUtil(pp).observed_phenotypic_features()]
    excluded = [HpoTerm(id=pf.type.id, label=pf.type.label, excluded=True) for pf in
                PhenopacketUtil(pp).negated_phenotypic_features()]

    # Resolve conflicts: a term listed as both observed and excluded is treated
    # as observed (presence is stronger evidence than recorded absence).
    observed_ids = {t.id for t in observed}
    conflicts = {t.id for t in excluded} & observed_ids
    if conflicts:
        print(
            f"[PRISM] Warning: {Path(phenopacket_path).name} lists {conflicts} as "
            f"both observed and excluded — treating as observed.",
            file=sys.stderr,
        )
        excluded = [t for t in excluded if t.id not in conflicts]

    return PatientPhenotype(
        subject_id=subject.id,
        sex=sex,
        age_years=_extract_age_years(subject),
        observed_terms=observed,
        excluded_terms=excluded,
    )
