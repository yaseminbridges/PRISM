"""Stateful orchestrator — initialises tools once, exposes a clean run interface.

Distinction from `pipeline.run()`:
  - `pipeline.run()` is functional: loads all tools on every call (fine for one-shot use).
  - `PRISMAgent` is stateful: tools (HPOGraph, HpoaRetriever, IC map, Orphanet index) are
    loaded once at construction and reused across multiple `run()` calls.  This matters for
    batch evaluation (e.g. PhEval benchmark) where the same knowledge bases are queried for
    thousands of cases.

The agent also exposes `explain()` for generating clinical narratives outside the pipeline
and `discriminating_features()` for interactive disambiguation.
"""
from __future__ import annotations
from pathlib import Path

from prism.ontology.hpo import HPOGraph
from prism.knowledge.hpoa.tool import HpoaRetriever
from prism.knowledge.orphanet.tool import OrphanetRetriever
from prism.models.report import RankedReport, ReRankedCandidate
from prism.reasoning.llm import LLMClient, MockLLMClient
from prism.pipeline import PRISMConfig, _merge_profiles, _resolve_onset
from prism.components.partition import partition
from prism.components.age_gate import apply_age_gate
from prism.components.rescore import rerank
from prism.components.disambiguate import disambiguate
from prism.components.mechanism import infer_mechanism_congruence
from prism.ingest.phenopacket_loader import load_phenopacket
from prism.ingest.exomiser_loader import load_exomiser


class PRISMAgent:
    """Stateful PRISM orchestrator.  Reuse across cases to amortise tool initialisation.

    Usage::

        agent = PRISMAgent(config)
        for pp, exo in case_pairs:
            report = agent.run(pp, exo)
    """

    def __init__(
        self,
        config: PRISMConfig | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.config = config or PRISMConfig()
        self.llm = llm or MockLLMClient()

        # Lazy-initialised knowledge bases — populated on first run()
        self._graph: HPOGraph | None = None
        self._hpoa: HpoaRetriever | None = None
        self._ic_map: dict[str, float] | None = None
        self._orpha: OrphanetRetriever | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        phenopacket_path: Path | str,
        exomiser_path: Path | str,
    ) -> RankedReport:
        """Run the full C1→C5 pipeline, reusing already-loaded tools."""
        self._ensure_loaded()

        graph = self._graph
        hpoa = self._hpoa
        ic_map = self._ic_map
        orpha = self._orpha
        llm = self.llm
        config = self.config

        patient = load_phenopacket(phenopacket_path)
        candidates = load_exomiser(exomiser_path)[: config.top_n]

        # C1 + C3
        from prism.models.report import ReRankedCandidate as RRC
        from prism.knowledge.hpoa.tool import DiseaseProfile

        partitioned: list[RRC] = []
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

            fit = partition(patient, profile, graph, llm)
            onset_years = _resolve_onset(disease_id, hpoa, orpha)
            fit = apply_age_gate(fit, patient.age_years, onset_years)

            partitioned.append(RRC(
                candidate=candidate,
                fit=fit,
                old_rank=candidate.exomiser_rank,
                new_rank=i + 1,
                rationale="pending C2",
            ))

        # C2
        reranked = rerank(partitioned, ic_map, mode=config.rescore_mode)

        # C5
        reranked = [
            rc.model_copy(update={"mechanism": infer_mechanism_congruence(rc.candidate)})
            for rc in reranked
        ]

        # C4
        disambiguation_result = disambiguate(reranked, profiles, patient, graph, llm)

        # callable_diagnoses
        callable_diagnoses: list[RRC] = []
        if disambiguation_result and disambiguation_result.best_supported:
            best = next(
                (rc for rc in reranked
                 if rc.candidate.disease_id == disambiguation_result.best_supported),
                None,
            )
            mech_ok = best and (best.mechanism is None or best.mechanism.verdict != "incongruent")
            if best and (best.fit.fit_score or 0.0) > 0.5 and mech_ok:
                callable_diagnoses = [best]
        elif not disambiguation_result and reranked:
            top = reranked[0]
            runner_up = reranked[1] if len(reranked) > 1 else None
            top_score = top.fit.fit_score or 0.0
            runner_up_score = runner_up.fit.fit_score or 0.0 if runner_up else 0.0
            mech_ok = top.mechanism is None or top.mechanism.verdict != "incongruent"
            if top_score > 0.6 and (top_score - runner_up_score) > 0.15 and mech_ok:
                callable_diagnoses = [top]

        # Narrative
        narratives: dict[str, str] = {}
        for rc in callable_diagnoses:
            did = rc.candidate.disease_id or rc.candidate.gene_symbol
            fit_summary = {
                "matched": [
                    {"term": fm.disease_feature.hpo_label, "freq": fm.disease_feature.frequency_class}
                    for fm in rc.fit.matched
                ],
                "partial": [fm.disease_feature.hpo_label for fm in rc.fit.partial],
                "unexplained": [t.label for t in rc.fit.unexplained],
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
            disambiguation=disambiguation_result,
            callable_diagnoses=callable_diagnoses,
            narratives=narratives,
        )

    def explain(self, rc: ReRankedCandidate) -> str:
        """Generate a clinical narrative for a single candidate outside the pipeline."""
        fit_summary = {
            "matched": [fm.disease_feature.hpo_label for fm in rc.fit.matched],
            "partial": [fm.disease_feature.hpo_label for fm in rc.fit.partial],
            "unexplained": [t.label for t in rc.fit.unexplained],
            "fit_score": rc.fit.fit_score,
        }
        return self.llm.synthesise_narrative(
            disease_name=rc.candidate.disease_name or rc.candidate.disease_id or "?",
            fit_summary=fit_summary,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Initialise knowledge bases on first call."""
        if self._graph is None:
            self._graph = HPOGraph.from_obo(self.config.hpo_path)
        if self._hpoa is None:
            self._hpoa = HpoaRetriever(self.config.hpoa_path)
        if self._ic_map is None:
            self._ic_map = self._graph.compute_ic(self._hpoa.disease_to_terms())
        if self._orpha is None:
            p4 = self.config.orphanet_product4_path
            pa = self.config.orphanet_ages_path
            if (p4 and p4.exists()) or (pa and pa.exists()):
                self._orpha = OrphanetRetriever(
                    product4_path=p4 if p4 and p4.exists() else None,
                    ages_path=pa if pa and pa.exists() else None,
                )