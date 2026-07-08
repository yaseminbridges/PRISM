"""C1 — Partition: build the per-candidate FitEvidence fit sheet."""

from prism.models.phenopacket import HpoTerm, PatientPhenotype
from prism.models.candidate import DiseaseFeature, FeatureMatch, FitEvidence
from prism.knowledge.hpoa.tool import DiseaseProfile
from prism.ontology.hpo import HPOGraph

# Maximum IC gap allowed for a partial match (patient broader than disease feature).
# IC gap = disease_feature.ic - patient_term.ic.
# A gap > MAX_IC_GAP means the patient's term is too vague to be informative:
# e.g. "Abnormality of nervous system" (IC≈0.5) vs "Focal clonic seizure" (IC≈6) → gap 5.5 → reject.
# "Seizure" (IC≈2.5) vs "Focal clonic seizure" (IC≈6) → gap 3.5 → reject.
# "Focal seizure" (IC≈4) vs "Focal clonic seizure" (IC≈6) → gap 2 → accept.
MAX_IC_GAP: float = 3.0


def partition(
    patient: PatientPhenotype,
    profile: DiseaseProfile,
    graph: HPOGraph,
    ic_map: dict[str, float],
) -> FitEvidence:
    """Partition patient phenotype against a disease profile into FitEvidence buckets.

    Matching tiers (deterministic first, LLM only for ambiguous cases):
      exact     — patient term ID == disease feature ID
      subsumed  — patient term is a descendant of disease feature (patient is more specific)
      partial   — patient term is an ancestor of disease feature (patient is broader) → LLM

    Disease-level checks:
      expected_absent  — unmatched disease feature explicitly excluded by the patient
      contradictions   — disease-excluded feature (Excluded/qualifier=NOT) the patient has
    """
    fit = FitEvidence(disease_id=profile.disease_id)
    matched_feature_ids: set[str] = set()

    for p_term in patient.observed_terms:
        fm = _best_match(p_term, profile.features, graph, ic_map, profile.disease_id)
        if fm is not None:
            matched_feature_ids.add(fm.disease_feature.hpo_id)
            # partial matches (LLM-resolved broader terms) are kept separate from
            # exact/subsumed matches because C2 weights them differently (w_partial < w_match)
            if fm.relation == "partial":
                fit.partial.append(fm)
            else:
                fit.matched.append(fm)
        else:
            # no relationship found at any tier — this patient term is unexplained by this disease
            fit.unexplained.append(p_term)

    # Expected absent: disease feature not matched AND patient explicitly excluded it
    # (or excluded an ancestor of it, covering it by subsumption)
    excluded_ids = {t.id for t in patient.excluded_terms}
    for feat in profile.features:
        if feat.hpo_id not in matched_feature_ids:
            if feat.hpo_id in excluded_ids or any(
                graph.subsumes(eid, feat.hpo_id) for eid in excluded_ids
            ):
                fit.expected_absent.append(feat)

    # Contradictions: disease says feature should not be present, patient has it
    # (the disease's excluded_features list = features annotated with qualifier=NOT)
    observed_ids = {t.id for t in patient.observed_terms}
    for excl_feat in profile.excluded_features:
        if excl_feat.hpo_id in observed_ids or any(
            graph.subsumes(excl_feat.hpo_id, oid) for oid in observed_ids
        ):
            fit.contradictions.append(excl_feat)

    return fit


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _best_match(
    p_term: HpoTerm,
    features: list[DiseaseFeature],
    graph: HPOGraph,
    ic_map: dict[str, float],
    disease_id: str,
) -> FeatureMatch | None:
    """Return the best FeatureMatch for p_term against the disease feature list.

    Priority: exact > subsumed > partial (LLM-resolved).
    Returns None if no relationship found.
    """
    # We keep the best candidate per tier and return the highest-priority one at the end.
    # Priority: exact > subsumed > partial (LLM). Only one match per patient term is returned.
    subsumed: FeatureMatch | None = None
    partial: FeatureMatch | None = None

    for feat in features:
        if p_term.id == feat.hpo_id:
            # Exact match — patient and disease use the identical HPO term. Return immediately.
            return FeatureMatch(
                patient_term=p_term,
                disease_feature=feat,
                relation="exact",
                provenance=(
                    f"{feat.source}: {disease_id} annotated with "
                    f"{feat.hpo_id} ({feat.frequency_class})"
                ),
            )

        if graph.subsumes(feat.hpo_id, p_term.id):
            # Subsumed: disease feature is an ancestor of the patient term.
            # The patient has a more specific variant of what the disease expects — still counts.
            if subsumed is None:
                subsumed = FeatureMatch(
                    patient_term=p_term,
                    disease_feature=feat,
                    relation="subsumed",
                    provenance=(
                        f"{feat.source}: {disease_id} annotated with "
                        f"{feat.hpo_label} ({feat.hpo_id}, {feat.frequency_class}); "
                        f"patient has more specific term {p_term.label} ({p_term.id})"
                    ),
                )

        elif graph.subsumes(p_term.id, feat.hpo_id) and partial is None:
            # Partial: patient has a broader (ancestor) term than the disease expects.
            # Accept only if the IC gap is within MAX_IC_GAP — i.e. the patient's term
            # is specific enough to be informative. A large gap (e.g. "Abnormality of
            # nervous system" vs "Focal clonic seizure") means the match is too vague.
            patient_ic = ic_map.get(p_term.id, 0.0)
            if feat.ic - patient_ic <= MAX_IC_GAP:
                partial = FeatureMatch(
                    patient_term=p_term,
                    disease_feature=feat,
                    relation="partial",
                    provenance=(
                        f"{feat.source}: {disease_id} expects {feat.hpo_label} "
                        f"({feat.hpo_id}, IC={feat.ic:.1f}); patient has broader term "
                        f"{p_term.label} ({p_term.id}, IC={patient_ic:.1f}); "
                        f"IC gap {feat.ic - patient_ic:.1f} ≤ {MAX_IC_GAP}"
                    ),
                )

    return subsumed or partial