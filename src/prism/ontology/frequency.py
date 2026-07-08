"""Frequency class weights for disease phenotype features.

HPOA and Orphanet annotate each disease-HPO association with a frequency — how often
that feature appears in patients with the disease. This file maps those qualitative
frequency classes to numeric weights used in the C2 fit score formula.

The classes follow the HPO frequency ontology (HP:0040279 and children):
  Obligate    (100%)       → weight 1.0  — always present; absence is a strong red flag
  VeryFrequent (80–99%)   → weight 0.9
  Frequent    (30–79%)    → weight 0.5
  Occasional  (5–29%)     → weight 0.2
  VeryRare    (1–4%)      → weight 0.05 — almost never present; matching barely helps
  Excluded    (0%)        → weight 0.0  — disease says this feature should NOT be present

An absent Obligate feature carries much more penalty than an absent Occasional one —
that is the point of weighting by frequency in the scoring formula.

`parse_hpoa_frequency` handles the three raw formats found in phenotype.hpoa:
  HPO term ID  e.g. "HP:0040281"
  Ratio        e.g. "5/30"
  Percentage   e.g. "16.67%"
"""

_CLASS_WEIGHTS: dict[str, float] = {
    "Obligate": 1.0,      # HP:0040280 — 100%
    "VeryFrequent": 0.9,  # HP:0040281 — 80-99%
    "Frequent": 0.5,      # HP:0040282 — 30-79%
    "Occasional": 0.2,    # HP:0040283 — 5-29%
    "VeryRare": 0.05,     # HP:0040284 — 1-4%
    "Excluded": 0.0,      # HP:0040285 — 0% (contradiction signal)
}

# fallback weight when frequency_class is None
FALLBACK_WEIGHT: float = 0.2  # treat unknown as Occasional

_HPO_TERM_TO_CLASS: dict[str, str] = {
    "HP:0040280": "Obligate",
    "HP:0040281": "VeryFrequent",
    "HP:0040282": "Frequent",
    "HP:0040283": "Occasional",
    "HP:0040284": "VeryRare",
    "HP:0040285": "Excluded",
}

# Percentage midpoints for each class boundary (5-29% → Occasional, etc.)
_PCT_BOUNDARIES: list[tuple[float, str]] = [
    (100.0, "Obligate"),
    (79.5, "VeryFrequent"),
    (29.5, "Frequent"),
    (4.5, "Occasional"),
    (0.5, "VeryRare"),
    (0.0, "Excluded"),
]


def frequency_class_weight(frequency_class: str | None) -> float:
    if frequency_class is None:
        return FALLBACK_WEIGHT
    return _CLASS_WEIGHTS.get(frequency_class, FALLBACK_WEIGHT)


def _pct_to_class(pct: float) -> str:
    for threshold, cls in _PCT_BOUNDARIES:
        if pct >= threshold:
            return cls
    return "Excluded"


def parse_hpoa_frequency(raw: str) -> tuple[str | None, float]:
    """Parse HPOA frequency field into (class_name, weight).

    Accepts: HPO term ID ("HP:0040281"), ratio ("5/30"), percentage ("16.67%"),
    or empty string.
    """
    raw = (raw or "").strip()
    if not raw:
        return None, FALLBACK_WEIGHT

    # HPO frequency term
    if raw.startswith("HP:"):
        cls = _HPO_TERM_TO_CLASS.get(raw)
        return cls, frequency_class_weight(cls)

    # Percentage
    if raw.endswith("%"):
        try:
            pct = float(raw[:-1])
            cls = _pct_to_class(pct)
            return cls, frequency_class_weight(cls)
        except ValueError:
            return None, FALLBACK_WEIGHT

    # Ratio n/m
    if "/" in raw:
        parts = raw.split("/", 1)
        try:
            numerator, denominator = float(parts[0]), float(parts[1])
            pct = (numerator / denominator * 100) if denominator else 0.0
            cls = _pct_to_class(pct)
            return cls, frequency_class_weight(cls)
        except (ValueError, ZeroDivisionError):
            return None, FALLBACK_WEIGHT

    return None, FALLBACK_WEIGHT