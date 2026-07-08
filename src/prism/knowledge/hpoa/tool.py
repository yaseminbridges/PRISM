"""HPOA disease profile retrieval.

This file provides two things:

1. `DiseaseProfile` — a dataclass holding the phenotype profile of a disease: its
   positively annotated HPO features and its disease-excluded features (features that
   should NOT be present in patients with this disease). Used throughout the pipeline
   as the reference to compare a patient against.

2. `HpoaRetriever` — loads the HPO Annotation database (`phenotype.hpoa`, a TSV file)
   into memory at construction, then efficiently returns DiseaseProfiles on demand.
   The HPOA file links disease IDs (OMIM/ORPHA) to HPO terms with frequency data.

   Key methods:
   - `get_disease_profile(disease_id)` — returns positive + excluded features for a disease.
     Features with qualifier=NOT or frequency=Excluded go into excluded_features.
   - `disease_to_terms()` — returns {disease_id -> set of HPO IDs} for IC computation.
   - `get_earliest_onset_years(disease_id)` — returns the earliest typical onset in years
     from HPOA onset annotations, used by C3 (age gate).

HPOA is the primary knowledge source. Orphanet supplements where HPOA has no data.
"""
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from prism.models.candidate import DiseaseFeature
from prism.ontology.frequency import parse_hpoa_frequency
from prism.ontology.hpo import HPOGraph

# HPO onset term → earliest age in years at which the disease can first manifest.
# Age gate excuses a missing feature if the patient is younger than this threshold
# (i.e. the onset window hasn't started yet for this patient).
# Values are the lower bound of each HPO-defined onset window.
_ONSET_TERM_YEARS: dict[str, float] = {
    "HP:0030674": 0.0,   # Antenatal onset         — prior to birth
    "HP:0034199": 0.0,   # Late first trimester     — 11–13 weeks gestation
    "HP:0034198": 0.0,   # Second trimester         — 14–27 weeks gestation
    "HP:0034197": 0.0,   # Third trimester          — 28 weeks to birth
    "HP:0011461": 0.0,   # Fetal onset              — after 8 weeks gestation
    "HP:0003577": 0.0,   # Congenital onset         — present at birth
    "HP:0003623": 0.0,   # Neonatal onset           — birth to 28 days
    "HP:0003593": 0.08,  # Infantile onset          — 28 days to 1 year (~0.077 yr)
    "HP:0011463": 1.0,   # Childhood onset          — 1 to 5 years
    "HP:0003621": 5.0,   # Juvenile onset           — 5 to 15 years
    "HP:0011462": 16.0,  # Young adult onset        — 16 to 40 years
    "HP:0025708": 16.0,  # Early young adult onset  — 16 to 19 years
    "HP:0025709": 19.0,  # Intermediate young adult — 19 to 25 years
    "HP:0025710": 25.0,  # Late young adult onset   — 25 to 40 years
    "HP:0003581": 16.0,  # Adult onset              — ≥16 years
    "HP:0003596": 40.0,  # Middle age onset         — 40 to 60 years
    "HP:0003584": 60.0,  # Late onset               — after 60 years
}


@dataclass
class DiseaseProfile:
    disease_id: str
    features: list[DiseaseFeature] = field(default_factory=list)
    excluded_features: list[DiseaseFeature] = field(default_factory=list)


class HpoaRetriever:
    """Deterministic retrieval of disease phenotype profiles from a local
    phenotype.hpoa file (HPOA v2 tab-separated format).

    Loads the full file into memory at construction; each get_disease_profile
    call is a filter operation — no network access.
    """

    def __init__(self, path: Path | str) -> None:
        self._df = pl.read_csv(
            path,
            separator="\t",
            comment_prefix="#",
            has_header=True,
            infer_schema_length=0,  # all strings — avoids type-inference surprises
        )

    def disease_ids(self) -> list[str]:
        return self._df["database_id"].unique().to_list()

    def disease_to_terms(self) -> dict[str, set[str]]:
        """Return {disease_id -> set of annotated HPO term IDs} for IC computation.

        Only phenotypic features (aspect == 'P') without NOT qualifier.
        """
        phenotype_rows = self._df.filter(
            (pl.col("aspect") == "P")
            & pl.col("qualifier").fill_null("").ne("NOT")
        )
        result: dict[str, set[str]] = {}
        for row in phenotype_rows.select(["database_id", "hpo_id"]).iter_rows():
            result.setdefault(row[0], set()).add(row[1])
        return result

    def get_earliest_onset_years(self, disease_id: str) -> float | None:
        """Return earliest typical onset in years from HPOA onset annotations.

        The onset column contains HPO onset term IDs (e.g. HP:0003593 = Infantile).
        Returns None if the disease has no onset annotations in HPOA.
        """
        rows = self._df.filter(
            (pl.col("database_id") == disease_id)
            & pl.col("onset").fill_null("").ne("")
        )
        years = [
            _ONSET_TERM_YEARS[oid]
            for oid in rows["onset"].to_list()
            if oid in _ONSET_TERM_YEARS
        ]
        return min(years) if years else None

    def get_disease_profile(
        self,
        disease_id: str,
        graph: HPOGraph | None = None,
        ic_map: dict[str, float] | None = None,
    ) -> DiseaseProfile:
        """Retrieve phenotypic features and disease-excluded features for disease_id.

        Args:
            disease_id: OMIM:xxxxxx or ORPHA:xxxxxx identifier.
            graph: Optional HPOGraph for resolving HPO term labels.
            ic_map: Optional precomputed IC values (from HPOGraph.compute_ic).

        Returns:
            DiseaseProfile with .features (positive annotations) and
            .excluded_features (qualifier=NOT or frequency=HP:0040285).
        """
        rows = self._df.filter(
            (pl.col("database_id") == disease_id) & (pl.col("aspect") == "P")
        )

        features: list[DiseaseFeature] = []
        excluded: list[DiseaseFeature] = []

        for row in rows.iter_rows(named=True):
            hpo_id = row["hpo_id"]
            freq_raw = row.get("frequency") or ""
            qualifier = (row.get("qualifier") or "").strip().upper()

            freq_class, _ = parse_hpoa_frequency(freq_raw)
            label = graph.name(hpo_id) if graph else hpo_id
            ic = (ic_map or {}).get(hpo_id, 0.0)

            feat = DiseaseFeature(
                hpo_id=hpo_id,
                hpo_label=label,
                frequency_class=freq_class,
                ic=ic,
                source="HPOA",
            )

            if qualifier == "NOT" or freq_class == "Excluded":
                excluded.append(feat)
            else:
                features.append(feat)

        return DiseaseProfile(
            disease_id=disease_id,
            features=features,
            excluded_features=excluded,
        )