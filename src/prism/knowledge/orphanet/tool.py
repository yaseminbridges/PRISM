"""Orphanet disease profile retrieval from local XML dumps.

Orphanet is a European rare disease database. This file supplements HPOA with phenotype
profiles from two Orphanet XML files:

  en_product4.xml      — disease-HPO associations with frequency labels
  en_product9_ages.xml — average age of onset per disease

`OrphanetRetriever` indexes both files into memory at construction. Lookup is by
ORPHA:{code} identifier — OMIM-only candidates will not find a profile here.

Why use Orphanet alongside HPOA?
- Orphanet often has richer onset data for rare diseases.
- Some diseases are ORPHA-only (no OMIM entry) and would have no profile from HPOA alone.
- HPOA and Orphanet can have complementary HPO annotations for the same disease.

In the pipeline (`pipeline.py`), HPOA is the primary source. Orphanet features are merged
in only where HPOA has no annotation for a given HPO term (deduplication by HPO ID).
IC values (from HPOGraph.compute_ic) are injected at retrieval time via `ic_map`.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

from prism.models.candidate import DiseaseFeature
from prism.knowledge.hpoa.tool import DiseaseProfile

_FREQ_MAP: dict[str, str] = {
    "Obligate (100%)": "Obligate",
    "Very frequent (99-80%)": "VeryFrequent",
    "Frequent (79-30%)": "Frequent",
    "Occasional (29-5%)": "Occasional",
    "Very rare (<4-1%)": "VeryRare",
    "Excluded (0%)": "Excluded",
}

# Orphanet onset label → earliest age in years (lower bound of Orphanet's defined window).
# Age gate excuses a missing feature if the patient is younger than this threshold.
# Orphanet windows: Antenatal=before birth, Neonatal=0–28d, Infancy=28d–2yr,
# Childhood=2–12yr, Adolescent=12–18yr, Adult=18+yr, Elderly=65+yr.
ONSET_YEARS: dict[str, float | None] = {
    "Antenatal": 0.0,   # before birth
    "Neonatal":  0.0,   # birth to 28 days
    "Infancy":   0.08,  # 28 days to 2 years (~0.077 yr)
    "Childhood": 2.0,   # 2 to 12 years
    "Adolescent": 12.0, # 12 to 18 years
    "Adult":     18.0,  # 18 years or later
    "Elderly":   65.0,  # 65 years or later
    "All ages":  None,  # no constraint — gate skipped
    "No data available": None,
}


class OrphanetRetriever:
    """Deterministic retrieval from local Orphanet XML dumps.

    Indexes are built once at construction from:
      en_product4.xml      — disease-HPO phenotype associations + frequency
      en_product9_ages.xml — average age of onset per disease
      en_product1.xml      — OMIM ↔ ORPHA cross-references (optional)

    Primary lookup is by "ORPHA:{code}". When en_product1.xml is loaded,
    OMIM IDs are also resolved via the cross-reference index so that OMIM
    candidates can benefit from Orphanet phenotype profiles.

    Only "E" (exact) and "NTBT" (ORPHA is a subtype of the OMIM entry)
    cross-references are used. "BTNT" mappings (ORPHA is broader than OMIM)
    are skipped to avoid mixing in features from unrelated diseases.
    """

    def __init__(
        self,
        product4_path: Path | str | None = None,
        ages_path: Path | str | None = None,
        product1_path: Path | str | None = None,
    ) -> None:
        self._profiles: dict[str, DiseaseProfile] = {}
        self._onsets: dict[str, list[str]] = {}
        self._omim_to_orpha: dict[str, str] = {}  # "OMIM:XXXXXX" -> "ORPHA:YYYYYY"

        if product4_path:
            self._load_product4(Path(product4_path))
        if ages_path:
            self._load_ages(Path(ages_path))
        if product1_path:
            self._load_product1(Path(product1_path))

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_product4(self, path: Path) -> None:
        root = ET.parse(path).getroot()
        for disorder in root.iter("Disorder"):
            orpha_code = disorder.findtext("OrphaCode")
            if not orpha_code:
                continue
            orpha_id = f"ORPHA:{orpha_code}"
            features: list[DiseaseFeature] = []
            excluded: list[DiseaseFeature] = []

            for assoc in disorder.findall(".//HPODisorderAssociation"):
                hpo_id = assoc.findtext(".//HPOId")
                if not hpo_id:
                    continue
                hpo_label = assoc.findtext(".//HPOTerm") or hpo_id
                freq_label = assoc.findtext(".//HPOFrequency/Name") or ""
                freq_class = _FREQ_MAP.get(freq_label)

                feat = DiseaseFeature(
                    hpo_id=hpo_id,
                    hpo_label=hpo_label,
                    frequency_class=freq_class,
                    ic=0.0,
                    source="Orphanet",
                )
                if freq_class == "Excluded":
                    excluded.append(feat)
                else:
                    features.append(feat)

            self._profiles[orpha_id] = DiseaseProfile(
                disease_id=orpha_id,
                features=features,
                excluded_features=excluded,
            )

    def _load_ages(self, path: Path) -> None:
        root = ET.parse(path).getroot()
        for disorder in root.iter("Disorder"):
            orpha_code = disorder.findtext("OrphaCode")
            if not orpha_code:
                continue
            orpha_id = f"ORPHA:{orpha_code}"
            onsets = [
                name
                for onset in disorder.findall(".//AverageAgeOfOnset")
                if (name := onset.findtext("Name")) and name != "No data available"
            ]
            self._onsets[orpha_id] = onsets

    def _load_product1(self, path: Path) -> None:
        """Build OMIM → ORPHA cross-reference index from en_product1.xml."""
        root = ET.parse(path).getroot()
        _USABLE_RELATIONS = {"E", "NTBT"}
        for disorder in root.iter("Disorder"):
            orpha_code = disorder.findtext("OrphaCode")
            if not orpha_code:
                continue
            orpha_id = f"ORPHA:{orpha_code}"
            for ref in disorder.findall(".//ExternalReference"):
                if ref.findtext("Source") != "OMIM":
                    continue
                omim_num = ref.findtext("Reference")
                if not omim_num:
                    continue
                rel_name = ref.findtext(".//DisorderMappingRelation/Name") or ""
                rel_code = rel_name.split("(")[0].strip()
                if rel_code not in _USABLE_RELATIONS:
                    continue
                omim_id = f"OMIM:{omim_num}"
                # First mapping wins; avoids overwriting an exact match with an NTBT one
                if omim_id not in self._omim_to_orpha:
                    self._omim_to_orpha[omim_id] = orpha_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def xref_orpha(self, omim_id: str) -> str | None:
        """Return the ORPHA ID that this OMIM ID maps to, or None if no xref exists."""
        return self._omim_to_orpha.get(omim_id)

    def get_disease_profile(
        self,
        disease_id: str,
        graph=None,
        ic_map: dict[str, float] | None = None,
    ) -> DiseaseProfile | None:
        """Return phenotype profile for disease_id (ORPHA:* or OMIM:*).

        For OMIM IDs, falls back to the cross-reference index (en_product1.xml)
        to find the corresponding ORPHA profile. Returns None if no profile found.
        """
        lookup_id = disease_id
        if disease_id not in self._profiles and disease_id in self._omim_to_orpha:
            lookup_id = self._omim_to_orpha[disease_id]
        profile = self._profiles.get(lookup_id)
        if profile is None:
            return None
        if graph is None and not ic_map:
            return profile

        def _enrich(feats: list[DiseaseFeature]) -> list[DiseaseFeature]:
            return [
                f.model_copy(update={
                    "hpo_label": graph.name(f.hpo_id) if graph else f.hpo_label,
                    "ic": (ic_map or {}).get(f.hpo_id, 0.0),
                })
                for f in feats
            ]

        return DiseaseProfile(
            disease_id=disease_id,
            features=_enrich(profile.features),
            excluded_features=_enrich(profile.excluded_features),
        )

    def get_age_of_onset(self, disease_id: str) -> list[str]:
        """Return onset category labels for disease_id (e.g. ['Neonatal', 'Infancy'])."""
        return self._onsets.get(disease_id, [])

    def earliest_onset_years(self, disease_id: str) -> float | None:
        """Return the earliest typical onset in years, or None if unknown."""
        candidates = [
            ONSET_YEARS[o]
            for o in self._onsets.get(disease_id, [])
            if ONSET_YEARS.get(o) is not None
        ]
        return min(candidates) if candidates else None