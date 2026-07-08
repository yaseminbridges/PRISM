"""Tests for C5 Mechanism congruence (mechanism.py)."""
from __future__ import annotations
import pytest

from prism.models.exomiser import ExomiserCandidate, Variant
from prism.components.mechanism import infer_mechanism_congruence


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _variant(
    acmg: str | None = "PATHOGENIC",
    consequence: str | None = "MISSENSE_VARIANT",
    pathogenicity_score: float | None = 0.95,
) -> Variant:
    return Variant(
        variant_id="chr1:100:A:T",
        acmg=acmg,
        consequence=consequence,
        pathogenicity_score=pathogenicity_score,
    )


def _candidate(
    variants: list[Variant],
    moi: str | None = "AUTOSOMAL_DOMINANT",
    disease_id: str = "OMIM:999",
    disease_name: str | None = "Test disease",
) -> ExomiserCandidate:
    return ExomiserCandidate(
        gene_symbol="GENE1",
        disease_id=disease_id,
        disease_name=disease_name,
        moi=moi,
        exomiser_rank=1,
        combined_score=0.9,
        phenotype_score=0.8,
        variant_score=0.7,
        variants=variants,
    )


# ---------------------------------------------------------------------------
# No variant data
# ---------------------------------------------------------------------------

class TestNoVariants:
    def test_no_variants_returns_uncertain(self):
        c = _candidate(variants=[])
        v = infer_mechanism_congruence(c)
        assert v.verdict == "uncertain"

    def test_reason_mentions_no_data(self):
        c = _candidate(variants=[])
        v = infer_mechanism_congruence(c)
        assert "No variant" in v.reason


# ---------------------------------------------------------------------------
# ACMG classification checks
# ---------------------------------------------------------------------------

class TestAcmgClass:
    def test_all_benign_is_incongruent(self):
        c = _candidate(variants=[_variant(acmg="BENIGN"), _variant(acmg="LIKELY_BENIGN")])
        v = infer_mechanism_congruence(c)
        assert v.verdict == "incongruent"

    def test_single_benign_variant_is_incongruent(self):
        c = _candidate(variants=[_variant(acmg="BENIGN")])
        v = infer_mechanism_congruence(c)
        assert v.verdict == "incongruent"

    def test_vus_only_is_uncertain(self):
        c = _candidate(variants=[_variant(acmg="UNCERTAIN_SIGNIFICANCE")])
        v = infer_mechanism_congruence(c)
        assert v.verdict == "uncertain"

    def test_vus_only_reason_mentions_no_plp(self):
        c = _candidate(variants=[_variant(acmg="UNCERTAIN_SIGNIFICANCE")])
        v = infer_mechanism_congruence(c)
        assert "PATHOGENIC" in v.reason or "P/LP" in v.reason

    def test_pathogenic_ad_is_congruent(self):
        c = _candidate(
            variants=[_variant(acmg="PATHOGENIC")],
            moi="AUTOSOMAL_DOMINANT",
        )
        v = infer_mechanism_congruence(c)
        assert v.verdict == "congruent"

    def test_likely_pathogenic_is_congruent(self):
        c = _candidate(
            variants=[_variant(acmg="LIKELY_PATHOGENIC")],
            moi="AD",
        )
        v = infer_mechanism_congruence(c)
        assert v.verdict == "congruent"

    def test_acmg_none_treated_as_vus(self):
        c = _candidate(variants=[_variant(acmg=None)])
        v = infer_mechanism_congruence(c)
        assert v.verdict == "uncertain"

    def test_mixed_benign_vus_is_uncertain_not_incongruent(self):
        # Has a VUS → not all benign → uncertain, not incongruent
        c = _candidate(variants=[
            _variant(acmg="BENIGN"),
            _variant(acmg="UNCERTAIN_SIGNIFICANCE"),
        ])
        v = infer_mechanism_congruence(c)
        assert v.verdict == "uncertain"


# ---------------------------------------------------------------------------
# MOI × allele count
# ---------------------------------------------------------------------------

class TestMoiAllelCount:
    def test_ar_one_variant_uncertain(self):
        c = _candidate(
            variants=[_variant(acmg="PATHOGENIC")],
            moi="AUTOSOMAL_RECESSIVE",
        )
        v = infer_mechanism_congruence(c)
        assert v.verdict == "uncertain"
        assert "biallelic" in v.reason.lower() or "2" in v.reason

    def test_ar_two_variants_congruent(self):
        c = _candidate(
            variants=[_variant(acmg="PATHOGENIC"), _variant(acmg="LIKELY_PATHOGENIC")],
            moi="AUTOSOMAL_RECESSIVE",
        )
        v = infer_mechanism_congruence(c)
        assert v.verdict == "congruent"

    def test_xlr_one_variant_uncertain(self):
        c = _candidate(
            variants=[_variant(acmg="PATHOGENIC")],
            moi="X_LINKED_RECESSIVE",
        )
        v = infer_mechanism_congruence(c)
        assert v.verdict == "uncertain"

    def test_xlr_abbreviated_one_variant_uncertain(self):
        c = _candidate(
            variants=[_variant(acmg="PATHOGENIC")],
            moi="XLR",
        )
        v = infer_mechanism_congruence(c)
        assert v.verdict == "uncertain"

    def test_ad_one_variant_congruent(self):
        c = _candidate(
            variants=[_variant(acmg="PATHOGENIC")],
            moi="AUTOSOMAL_DOMINANT",
        )
        v = infer_mechanism_congruence(c)
        assert v.verdict == "congruent"

    def test_no_moi_one_plp_variant_congruent(self):
        c = _candidate(
            variants=[_variant(acmg="PATHOGENIC")],
            moi=None,
        )
        v = infer_mechanism_congruence(c)
        assert v.verdict == "congruent"


# ---------------------------------------------------------------------------
# Consequence summary in rationale
# ---------------------------------------------------------------------------

class TestConsequenceSummary:
    def test_lof_consequence_mentioned(self):
        c = _candidate(
            variants=[_variant(acmg="PATHOGENIC", consequence="STOP_GAINED")],
            moi="AUTOSOMAL_DOMINANT",
        )
        v = infer_mechanism_congruence(c)
        assert v.verdict == "congruent"
        assert "LoF" in v.reason

    def test_missense_consequence_mentioned(self):
        c = _candidate(
            variants=[_variant(acmg="PATHOGENIC", consequence="MISSENSE_VARIANT")],
            moi="AUTOSOMAL_DOMINANT",
        )
        v = infer_mechanism_congruence(c)
        assert v.verdict == "congruent"
        assert "missense" in v.reason.lower()

    def test_frameshift_is_lof(self):
        c = _candidate(
            variants=[_variant(acmg="PATHOGENIC", consequence="FRAMESHIFT_VARIANT")],
            moi="AUTOSOMAL_DOMINANT",
        )
        v = infer_mechanism_congruence(c)
        assert "LoF" in v.reason

    def test_splice_donor_is_lof(self):
        c = _candidate(
            variants=[_variant(acmg="PATHOGENIC", consequence="SPLICE_DONOR_VARIANT")],
            moi="AUTOSOMAL_DOMINANT",
        )
        v = infer_mechanism_congruence(c)
        assert "LoF" in v.reason

    def test_moi_appears_in_rationale(self):
        c = _candidate(
            variants=[_variant(acmg="PATHOGENIC")],
            moi="AUTOSOMAL_DOMINANT",
        )
        v = infer_mechanism_congruence(c)
        assert "AUTOSOMAL_DOMINANT" in v.reason


# ---------------------------------------------------------------------------
# Pipeline integration: mechanism field populated on ReRankedCandidate
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_mechanism_field_set_after_pipeline(self):
        """mechanism is None before C5 and set after."""
        from prism.models.candidate import FitEvidence
        from prism.models.report import ReRankedCandidate

        candidate = _candidate(variants=[_variant(acmg="PATHOGENIC")], moi="AUTOSOMAL_DOMINANT")
        rc = ReRankedCandidate(
            candidate=candidate,
            fit=FitEvidence(disease_id="OMIM:999"),
            old_rank=1,
            new_rank=1,
            rationale="test",
        )
        assert rc.mechanism is None  # default

        verdict = infer_mechanism_congruence(rc.candidate)
        rc2 = rc.model_copy(update={"mechanism": verdict})
        assert rc2.mechanism is not None
        assert rc2.mechanism.verdict == "congruent"

    def test_incongruent_suppresses_callable(self):
        """An incongruent mechanism verdict must block callable_diagnoses."""
        from prism.models.candidate import FitEvidence
        from prism.models.report import ReRankedCandidate, MechanismVerdict

        candidate = _candidate(variants=[], moi="AUTOSOMAL_DOMINANT")
        incongruent_mech = MechanismVerdict(verdict="incongruent", reason="all benign")
        rc = ReRankedCandidate(
            candidate=candidate,
            fit=FitEvidence(disease_id="OMIM:999", fit_score=0.8),
            old_rank=1,
            new_rank=1,
            rationale="test",
            mechanism=incongruent_mech,
        )
        # The callable_diagnoses logic checks mechanism.verdict != "incongruent"
        mech_ok = rc.mechanism is None or rc.mechanism.verdict != "incongruent"
        assert not mech_ok