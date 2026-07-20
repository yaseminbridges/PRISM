"""Top-level pipeline: orchestrates a single (phenopacket, exomiser) → RankedReport run.

This is the best file to read to understand how all of PRISM's components connect.
Call `run()` for one-shot use; use `PRISMAgent` (reasoning/agent.py) for batch runs
where loading HPO/HPOA repeatedly would be too slow.

The pipeline runs in this order for each case:
  1. Ingest — parse the phenopacket and Exomiser parquet into typed objects.
  2. Ontology — load HPOGraph and compute IC map (how specific each HPO term is).
  3. Knowledge — build disease profiles from HPOA + Orphanet for each candidate disease.
  4. C1 Partition — for each candidate, compare patient terms to disease features,
     producing a FitEvidence (matched / partial / unexplained / expected_absent / contradictions).
  5. C3 Age gate — excuse expected_absent features if disease onset is later than patient age.
  6. C2 Rescore — compute fit_score = tanh(weighted sum) and re-rank candidates.
  7. C5 Mechanism — check whether variant evidence (ACMG class, MOI) supports each disease.
  8. C4 Disambiguate — if top candidates overlap phenotypically, find discriminating features
     and use the LLM to resolve silent cases.
  9. Callable diagnosis — apply strict thresholds to decide whether to call a diagnosis.
  10. Narrative — LLM generates plain-English summaries for the top N candidates.
"""
from pathlib import Path
from pydantic import BaseModel

from prism._paths import HPO_OBO, HPOA, ORPHANET_P4, ORPHANET_AGES, ORPHANET_XREF
from prism.ingest.phenopacket_loader import load_phenopacket
from prism.ingest.exomiser_loader import load_exomiser
from prism.ontology.hpo import HPOGraph
from prism.knowledge.hpoa.tool import DiseaseProfile, HpoaRetriever
from prism.knowledge.orphanet.tool import OrphanetRetriever
from prism.models.report import RankedReport, ReRankedCandidate
from prism.components.partition import partition
from prism.components.age_gate import apply_age_gate
from prism.components.rescore import rerank
from prism.components.disambiguate import disambiguate
from prism.components.mechanism import infer_mechanism_congruence
from prism.reasoning.llm import LLMClient, MockLLMClient


class PRISMConfig(BaseModel):
    hpo_path: Path = HPO_OBO
    hpoa_path: Path = HPOA
    orphanet_product4_path: Path | None = ORPHANET_P4
    orphanet_ages_path: Path | None = ORPHANET_AGES
    orphanet_xref_path: Path | None = ORPHANET_XREF
    top_n: int = 20
    rescore_mode: str = "blended"
    narrative_top_n: int = 3   # generate LLM narrative for this many top candidates
    online_enrichment: bool = False  # off by default — never on inside GEL


def _resolve_onset(
    disease_id: str | None,
    hpoa: HpoaRetriever,
    orpha: OrphanetRetriever | None,
) -> float | None:
    """Return earliest typical onset in years for a disease, or None if unknown.

    Checks Orphanet first (richer onset data for ORPHA IDs), then HPOA onset
    annotations.  Takes the minimum when both sources have values.
    """
    if not disease_id:
        return None
    years: list[float] = []
    if orpha:
        oy = orpha.earliest_onset_years(disease_id)
        if oy is not None:
            years.append(oy)
    hy = hpoa.get_earliest_onset_years(disease_id)
    if hy is not None:
        years.append(hy)
    return min(years) if years else None


def _dedup_orpha_omim(
    candidates: list,
    orpha: "OrphanetRetriever | None",
) -> list:
    """Drop ORPHA candidates that are xrefs of an OMIM candidate in the same (gene, moi) group.

    Exomiser's associatedDiseases can list both OMIM:101600 and ORPHA:710 for the same
    disease. Keeping both wastes a slot and confuses ranking. We prefer the OMIM entry
    because HPOA has richer frequency annotations for OMIM IDs.
    """
    if not orpha:
        return candidates

    # Collect OMIM IDs present per (gene, moi) group
    omim_by_group: dict[tuple, set] = {}
    for c in candidates:
        if c.disease_id and c.disease_id.startswith("OMIM:"):
            key = (c.gene_symbol, c.moi)
            omim_by_group.setdefault(key, set()).add(c.disease_id)

    result = []
    for c in candidates:
        if c.disease_id and c.disease_id.startswith("ORPHA:"):
            key = (c.gene_symbol, c.moi)
            omim_ids = omim_by_group.get(key, set())
            if any(orpha.xref_orpha(oid) == c.disease_id for oid in omim_ids):
                continue  # covered by OMIM xref — drop the ORPHA duplicate
        result.append(c)
    return result


def _merge_profiles(
    hpoa_profile: DiseaseProfile | None,
    orpha_profile: DiseaseProfile | None,
    disease_id: str,
) -> DiseaseProfile:
    """Merge HPOA and Orphanet profiles, deduplicating by HPO ID.

    HPOA is primary (frequency data is richer); Orphanet supplements where HPOA
    has no annotation for the disease (common for ORPHA-only IDs).
    """
    if hpoa_profile is None and orpha_profile is None:
        return DiseaseProfile(disease_id=disease_id)
    if orpha_profile is None:
        return hpoa_profile  # type: ignore[return-value]
    if hpoa_profile is None:
        return orpha_profile

    seen = {f.hpo_id for f in hpoa_profile.features}
    extra_features = [f for f in orpha_profile.features if f.hpo_id not in seen]

    seen_excl = {f.hpo_id for f in hpoa_profile.excluded_features}
    extra_excluded = [f for f in orpha_profile.excluded_features if f.hpo_id not in seen_excl]

    return DiseaseProfile(
        disease_id=disease_id,
        features=hpoa_profile.features + extra_features,
        excluded_features=hpoa_profile.excluded_features + extra_excluded,
    )


class PRISMResources:
    """Pre-loaded ontology and knowledge resources shared across cases in a batch run."""

    def __init__(self, config: PRISMConfig) -> None:
        self.config = config
        self.graph = HPOGraph.from_obo(config.hpo_path)
        self.hpoa = HpoaRetriever(config.hpoa_path)
        self.ic_map = self.graph.compute_ic(self.hpoa.disease_to_terms())

        self.orpha: OrphanetRetriever | None = None
        p4 = config.orphanet_product4_path
        pa = config.orphanet_ages_path
        px = config.orphanet_xref_path
        if (p4 and p4.exists()) or (pa and pa.exists()) or (px and px.exists()):
            self.orpha = OrphanetRetriever(
                product4_path=p4 if p4 and p4.exists() else None,
                ages_path=pa if pa and pa.exists() else None,
                product1_path=px if px and px.exists() else None,
            )


def run_with_resources(
    phenopacket_path: Path | str,
    exomiser_path: Path | str,
    resources: PRISMResources,
    llm: LLMClient | None = None,
) -> RankedReport:
    """Per-case pipeline using pre-loaded shared resources.

    Called by the batch command so that HPO/HPOA/Orphanet are loaded only once.
    """
    if llm is None:
        llm = MockLLMClient()

    config = resources.config
    graph = resources.graph
    hpoa = resources.hpoa
    ic_map = resources.ic_map
    orpha = resources.orpha

    # Ingest
    patient = load_phenopacket(phenopacket_path)
    # top_n is passed to load_exomiser so Polars filters to top-N ranks before the
    # Python loop — avoids building thousands of Pydantic objects for phenotype-only
    # parquets (16k+ genes) that would all be discarded after the rank filter anyway.
    all_candidates = load_exomiser(exomiser_path, top_n=config.top_n)
    candidates = _dedup_orpha_omim(all_candidates, orpha)

    if not candidates:
        return RankedReport(case_id=patient.subject_id, reranked=[], disambiguation=None,
                            callable_diagnoses=[], narratives={})

    # C1 Partition — build FitEvidence per candidate; accumulate profiles for C4
    partitioned: list[ReRankedCandidate] = []
    profiles: dict[str, DiseaseProfile] = {}
    for i, candidate in enumerate(candidates):
        disease_id = candidate.disease_id

        hpoa_profile = (
            hpoa.get_disease_profile(disease_id, graph=graph, ic_map=ic_map)
            if disease_id else None
        )
        orpha_profile = (
            orpha.get_disease_profile(disease_id, graph=graph, ic_map=ic_map)
            if (orpha and disease_id) else None
        )
        profile = _merge_profiles(hpoa_profile, orpha_profile, disease_id or candidate.gene_symbol)

        if disease_id:
            profiles[disease_id] = profile

        fit = partition(patient, profile, graph, ic_map)

        # C3 Age gate — excuse expected_absent features if disease onset > patient age
        onset_years = _resolve_onset(disease_id, hpoa, orpha)
        fit = apply_age_gate(fit, patient.age_years, onset_years)

        partitioned.append(ReRankedCandidate(
            candidate=candidate,
            fit=fit,
            old_rank=candidate.exomiser_rank,
            new_rank=i + 1,  # placeholder; C2 will update
            rationale="pending C2",
        ))

    # C2 Re-score and conservative re-rank
    reranked = rerank(partitioned, ic_map, mode=config.rescore_mode)

    # C5 Mechanism congruence — annotate each candidate with variant evidence verdict
    reranked = [
        rc.model_copy(update={"mechanism": infer_mechanism_congruence(rc.candidate)})
        for rc in reranked
    ]

    # C4 Disambiguate — detect near-similar candidates and find discriminating features
    disambiguation = disambiguate(reranked, profiles, patient, graph, llm)

    # callable_diagnoses: ≤2 high-confidence calls; empty is valid and preferred.
    # Two paths to a call:
    #   Path A (C4 ran): C4 identified a best_supported candidate → call it if fit_score > 0.5
    #                    and mechanism is not incongruent.
    #   Path B (no C4):  No overlap found → call the top candidate only if it dominates
    #                    (fit_score > 0.6, leads runner-up by > 0.15, mechanism OK).
    # The thresholds are intentionally conservative — a missed diagnosis is less harmful
    # than a confident wrong call.
    callable_diagnoses: list[ReRankedCandidate] = []
    if disambiguation and disambiguation.best_supported:
        best = next(
            (rc for rc in reranked if rc.candidate.disease_id == disambiguation.best_supported),
            None,
        )
        mech_ok = best and (best.mechanism is None or best.mechanism.verdict != "incongruent")
        if best and (best.fit.fit_score or 0.0) > 0.5 and mech_ok:
            callable_diagnoses = [best]
    elif not disambiguation and reranked:
        top = reranked[0]
        runner_up = reranked[1] if len(reranked) > 1 else None
        top_score = top.fit.fit_score or 0.0
        runner_up_score = runner_up.fit.fit_score or 0.0 if runner_up else 0.0
        mech_ok = top.mechanism is None or top.mechanism.verdict != "incongruent"
        if top_score > 0.6 and (top_score - runner_up_score) > 0.15 and mech_ok:
            callable_diagnoses = [top]

    # Narrative synthesis — LLM generates a narrative for the top N candidates
    # (not just callables, so the LLM's contribution is always visible in output)
    narratives: dict[str, str] = {}
    for rc in reranked[: config.narrative_top_n]:
        did = rc.candidate.disease_id or rc.candidate.gene_symbol
        fit_summary = {
            "matched": [
                {"term": fm.disease_feature.hpo_label, "freq": fm.disease_feature.frequency_class}
                for fm in rc.fit.matched
            ],
            "partial": [
                {
                    "patient_term": fm.patient_term.label,
                    "disease_feature": fm.disease_feature.hpo_label,
                    "provenance": fm.provenance,
                }
                for fm in rc.fit.partial
            ],
            "unexplained": [t.label for t in rc.fit.unexplained],
            "expected_absent": [f.hpo_label for f in rc.fit.expected_absent],
            "contradictions": [f.hpo_label for f in rc.fit.contradictions],
            "fit_score": rc.fit.fit_score,
            "mechanism": rc.mechanism.verdict if rc.mechanism else "unknown",
        }
        narratives[did] = llm.synthesise_narrative(
            disease_name=rc.candidate.disease_name or did,
            fit_summary=fit_summary,
        )

    return RankedReport(
        case_id=patient.subject_id,
        reranked=reranked,
        disambiguation=disambiguation,
        callable_diagnoses=callable_diagnoses,
        narratives=narratives,
    )


def run(
    phenopacket_path: Path | str,
    exomiser_path: Path | str,
    config: PRISMConfig | None = None,
    llm: LLMClient | None = None,
) -> RankedReport:
    """Convenience wrapper for one-shot use: loads all resources then runs the pipeline.

    For batch processing use PRISMResources + run_with_resources so that HPO/HPOA/Orphanet
    are loaded only once.
    """
    if config is None:
        config = PRISMConfig()
    return run_with_resources(phenopacket_path, exomiser_path, PRISMResources(config), llm)