"""C4 — Disambiguator: discriminating-feature analysis for overlapping candidates.

Triggered when ≥2 top candidates have high disease-profile similarity (Jaccard ≥
threshold).  Fully deterministic — the LLM is only called when the patient is
silent on a key discriminator (the one place an agentic loop is justified per §6).

Algorithm:
  1. Compute pairwise Jaccard similarity between candidate disease profiles.
  2. Collect the set of candidates involved in any near-similar pair.
  3. For each candidate, discriminating features = features unique to that
     disease (not shared with any other candidate in the set).
  4. Check patient observed + excluded terms against each discriminator.
  5. Score candidates: +1 per supported discriminator, -1 per contradicted one.
     Disease-excluded features that the patient HAS also score -1 for that disease.
  6. Report best-supported candidate and list of features to check.
"""
from typing import Literal

from prism.models.phenopacket import PatientPhenotype
from prism.models.candidate import DiseaseFeature
from prism.models.report import DisambiguationReport, ReRankedCandidate
from prism.knowledge.hpoa.tool import DiseaseProfile
from prism.ontology.hpo import HPOGraph
from prism.reasoning.llm import LLMClient


def disambiguate(
    candidates: list[ReRankedCandidate],
    profiles: dict[str, DiseaseProfile],
    patient: PatientPhenotype,
    graph: HPOGraph,
    llm: LLMClient,
    similarity_threshold: float = 0.3,
    top_n: int = 5,
) -> DisambiguationReport | None:
    """Run C4 disambiguation. Returns None if no near-similar pair is found."""

    # Step 1 — find overlapping candidate pairs among top-N positively-scored hits
    top = [
        rc for rc in candidates[:top_n]
        if (rc.fit.fit_score or 0.0) > 0 and rc.candidate.disease_id in profiles
    ]

    overlapping_ids: set[str] = set()
    for i, rc_a in enumerate(top):
        for rc_b in top[i + 1:]:
            did_a = rc_a.candidate.disease_id
            did_b = rc_b.candidate.disease_id
            terms_a = {f.hpo_id for f in profiles[did_a].features}
            terms_b = {f.hpo_id for f in profiles[did_b].features}
            if _jaccard(terms_a, terms_b) >= similarity_threshold:
                overlapping_ids.add(did_a)
                overlapping_ids.add(did_b)

    if not overlapping_ids:
        return None

    overlap_profiles = {did: profiles[did] for did in overlapping_ids}

    # Step 2 — discriminating features (unique to each disease in the overlap set)
    discriminators = _unique_features(overlap_profiles)

    # Step 3 — score each candidate against patient
    scores: dict[str, int] = {did: 0 for did in overlapping_ids}
    all_discriminating: set[str] = set()
    silent_discriminators: list[tuple[str, str]] = []   # (hpo_id, disease_id)

    for did, disc_ids in discriminators.items():
        all_discriminating |= disc_ids
        for hpo_id in disc_ids:
            stance = _patient_stance(hpo_id, patient, graph)
            if stance == "supports":
                scores[did] += 1
            elif stance == "against":
                scores[did] -= 1
            else:
                silent_discriminators.append((hpo_id, did))

    # Disease-excluded features that the patient HAS argue against that disease
    for did, profile in overlap_profiles.items():
        for excl_feat in profile.excluded_features:
            stance = _patient_stance(excl_feat.hpo_id, patient, graph)
            if stance == "supports":
                scores[did] -= 1  # disease says no, patient has it
            elif stance == "against":
                scores[did] += 1  # disease says no, patient doesn't — consistent

    # Step 4 — LLM consulted for silent discriminators; update scores based on response.
    # "matched"/"partial" → LLM infers feature likely present → +1 for that disease.
    # "absent_expected"/"contradiction" → LLM infers feature likely absent → -1.
    # "unexplained" → uncertain → neutral.
    llm_verdicts: dict[tuple[str, str], str] = {}  # (hpo_id, did) -> label
    for hpo_id, did in silent_discriminators:
        profile = overlap_profiles[did]
        feat = next((f for f in profile.features if f.hpo_id == hpo_id), None)
        if feat:
            result = llm.resolve_match(
                patient_term_id="",
                patient_term_label="[silent]",
                disease_feature_id=hpo_id,
                disease_feature_label=feat.hpo_label,
                context=(
                    f"Discriminating feature for {did}: {feat.hpo_label} ({hpo_id}). "
                    f"Patient phenopacket is silent on this term. "
                    f"Is there clinical context suggesting presence or absence?"
                ),
            )
            llm_verdicts[(hpo_id, did)] = result.label
            if result.label in ("matched", "partial"):
                scores[did] += 1
            elif result.label in ("absent_expected", "contradiction"):
                scores[did] -= 1
            # "unexplained" → neutral, no score change

    # Step 5 — pick best supported
    best_supported: str | None = None
    if scores:
        best_did = max(scores, key=lambda d: scores[d])
        best_score = scores[best_did]
        other_scores = [s for d, s in scores.items() if d != best_did]
        if best_score > 0 and (not other_scores or best_score > max(other_scores)):
            best_supported = best_did

    # Step 6 — rationale
    rationale = _build_rationale(
        candidates, discriminators, scores, patient, graph,
        overlap_profiles, silent_discriminators, llm_verdicts, best_supported,
    )

    return DisambiguationReport(
        candidate_disease_ids=sorted(overlapping_ids),
        discriminating_features=sorted(all_discriminating),
        best_supported=best_supported,
        rationale=rationale,
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _unique_features(profiles: dict[str, DiseaseProfile]) -> dict[str, set[str]]:
    """Return, per disease, HPO IDs present in that disease but NO other in the set."""
    all_terms = {did: {f.hpo_id for f in p.features} for did, p in profiles.items()}
    return {
        did: terms - set().union(*(t for d, t in all_terms.items() if d != did))
        for did, terms in all_terms.items()
    }


def _patient_stance(
    hpo_id: str,
    patient: PatientPhenotype,
    graph: HPOGraph,
) -> Literal["supports", "against", "silent"]:
    """Does the patient support, argue against, or not mention this HPO term?"""
    observed_ids = {t.id for t in patient.observed_terms}
    excluded_ids = {t.id for t in patient.excluded_terms}

    # Has the term or a descendant of it
    if hpo_id in observed_ids or any(
        graph.subsumes(hpo_id, oid) for oid in observed_ids
    ):
        return "supports"

    # Excluded the term or an ancestor (covering it by subsumption)
    if hpo_id in excluded_ids or any(
        graph.subsumes(eid, hpo_id) for eid in excluded_ids
    ):
        return "against"

    return "silent"


def _build_rationale(
    candidates: list[ReRankedCandidate],
    discriminators: dict[str, set[str]],
    scores: dict[str, int],
    patient: PatientPhenotype,
    graph: HPOGraph,
    profiles: dict[str, DiseaseProfile],
    silent: list[tuple[str, str]],
    llm_verdicts: dict[tuple[str, str], str],
    best_supported: str | None,
) -> str:
    name_map = {
        rc.candidate.disease_id: rc.candidate.disease_name or rc.candidate.disease_id
        for rc in candidates
        if rc.candidate.disease_id in profiles
    }

    parts: list[str] = []
    parts.append(
        "Overlapping candidates: "
        + ", ".join(f"{name_map.get(d, d)} ({d})" for d in sorted(profiles))
        + "."
    )

    for did, disc_ids in discriminators.items():
        if not disc_ids:
            continue
        feats = profiles[did].features
        labels = [
            next((f.hpo_label for f in feats if f.hpo_id == hid), hid)
            for hid in disc_ids
        ]
        stances = [_patient_stance(hid, patient, graph) for hid in disc_ids]
        detail = "; ".join(
            f"{lbl} ({hid}) → patient {st}"
            for hid, lbl, st in zip(disc_ids, labels, stances)
        )
        parts.append(f"{name_map.get(did, did)} discriminators: {detail}.")

    if silent:
        llm_parts: list[str] = []
        no_verdict: list[str] = []
        for hpo_id, did in silent:
            feat_label = next(
                (f.hpo_label for f in profiles[did].features if f.hpo_id == hpo_id),
                hpo_id,
            )
            verdict = llm_verdicts.get((hpo_id, did))
            disease_name = name_map.get(did, did)
            if verdict in ("matched", "partial"):
                llm_parts.append(f"{feat_label} ({hpo_id}) likely present → +1 {disease_name}")
            elif verdict in ("absent_expected", "contradiction"):
                llm_parts.append(f"{feat_label} ({hpo_id}) likely absent → -1 {disease_name}")
            elif verdict == "unexplained":
                llm_parts.append(f"{feat_label} ({hpo_id}) uncertain → neutral")
            else:
                no_verdict.append(f"{feat_label} ({hpo_id})")
        if llm_parts:
            parts.append("LLM silent-discriminator inference: " + "; ".join(llm_parts) + ".")
        if no_verdict:
            parts.append(f"Check for: {', '.join(no_verdict)}.")

    if best_supported:
        parts.append(f"Best supported: {name_map.get(best_supported, best_supported)} ({best_supported}).")
    else:
        parts.append("Insufficient evidence to distinguish candidates.")

    return " ".join(parts)