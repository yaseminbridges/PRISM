from __future__ import annotations
from pathlib import Path
import pytest
from prism.ingest.phenopacket_loader import load_phenopacket
from prism.ingest.exomiser_loader import load_exomiser


class TestPhenopacketLoader:
    def test_gipc3_subject(self, gipc3_phenopacket):
        p = load_phenopacket(gipc3_phenopacket)
        assert p.subject_id == "proband-gipc3"
        assert p.sex == "male"
        assert p.age_years is None  # age parsing deferred

    def test_gipc3_observed_terms(self, gipc3_phenopacket):
        p = load_phenopacket(gipc3_phenopacket)
        assert len(p.observed_terms) == 2
        assert len(p.excluded_terms) == 0
        ids = {t.id for t in p.observed_terms}
        assert "HP:0000407" in ids
        assert "HP:0000399" in ids

    def test_pfeiffer_excluded_term(self, pfeiffer_phenopacket):
        p = load_phenopacket(pfeiffer_phenopacket)
        assert len(p.observed_terms) == 3
        assert len(p.excluded_terms) == 1
        assert p.excluded_terms[0].id == "HP:0001161"
        assert p.excluded_terms[0].excluded is True

    def test_pfeiffer_sex(self, pfeiffer_phenopacket):
        p = load_phenopacket(pfeiffer_phenopacket)
        assert p.sex == "female"

    def test_crouzon_case(self, crouzon_phenopacket):
        p = load_phenopacket(crouzon_phenopacket)
        assert p.subject_id == "proband-crouzon"
        assert len(p.observed_terms) == 2
        assert len(p.excluded_terms) == 0


class TestExomiserLoader:
    def test_loads_candidates(self, real_exomiser_parquet):
        candidates = load_exomiser(real_exomiser_parquet)
        assert len(candidates) > 0

    def test_sorted_by_rank(self, real_exomiser_parquet):
        candidates = load_exomiser(real_exomiser_parquet)
        ranks = [c.exomiser_rank for c in candidates]
        assert ranks == sorted(ranks)

    def test_gipc3_candidate(self, real_exomiser_parquet):
        candidates = load_exomiser(real_exomiser_parquet)
        gipc3 = next((c for c in candidates if c.gene_symbol == "GIPC3"), None)
        assert gipc3 is not None
        assert gipc3.exomiser_rank == 2
        assert gipc3.disease_id == "OMIM:601869"
        assert gipc3.moi == "AR"

    def test_gipc3_variant(self, real_exomiser_parquet):
        candidates = load_exomiser(real_exomiser_parquet)
        gipc3 = next(c for c in candidates if c.gene_symbol == "GIPC3")
        assert len(gipc3.variants) == 1
        v = gipc3.variants[0]
        assert v.consequence == "MISSENSE_VARIANT"
        assert v.variant_id == "19-3586872-G-A"
        assert v.acmg == "UNCERTAIN_SIGNIFICANCE"

    def test_all_candidates_have_gene_symbol(self, real_exomiser_parquet):
        candidates = load_exomiser(real_exomiser_parquet)
        assert all(c.gene_symbol for c in candidates)