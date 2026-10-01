"""Tests for writing PRISM scores back into an Exomiser parquet."""
from __future__ import annotations

import polars as pl
import pytest

from prism.exomiser_export import write_exomiser_with_prism_scores
from prism.models.candidate import FitEvidence
from prism.models.exomiser import ExomiserCandidate
from prism.models.report import RankedReport, ReRankedCandidate


def _rc(gene: str, moi: str, disease: str, new_rank: int, fit: float) -> ReRankedCandidate:
    return ReRankedCandidate(
        candidate=ExomiserCandidate(
            gene_symbol=gene, disease_id=disease, disease_name=disease, moi=moi,
            exomiser_rank=1, combined_score=0.9, phenotype_score=0.9, variant_score=0.9,
            variants=[],
        ),
        fit=FitEvidence(disease_id=disease, fit_score=fit),
        old_rank=1, new_rank=new_rank, rationale="",
    )


def test_scores_joined_per_gene_and_moi(tmp_path):
    exomiser = tmp_path / "case-exomiser.parquet"
    pl.DataFrame({
        "rank": [1, 1, 1, 2, 3],
        "geneSymbol": ["FBN1", "FBN1", "FBN1", "GENE2", "UNSCORED"],
        "moi": ["AD", "AD", "AR", "AD", "AD"],
        "diseasePhenotypeScore": [0.8, 0.8, 0.7, 0.5, 0.1],
    }).write_parquet(exomiser)
    report = RankedReport(
        case_id="case",
        reranked=[
            _rc("FBN1", "AD", "OMIM:154700", new_rank=1, fit=0.95),
            _rc("FBN1", "AD", "OMIM:604308", new_rank=3, fit=0.99),  # worse rank → ignored
            _rc("GENE2", "AD", "OMIM:2", new_rank=2, fit=0.4),
            _rc("FBN1", "AR", "OMIM:3", new_rank=4, fit=0.2),
        ],
        disambiguation=None,
        callable_diagnoses=[],
    )
    out = tmp_path / "out" / "case-exomiser.parquet"
    write_exomiser_with_prism_scores(exomiser, report, out)

    df = pl.read_parquet(out)
    assert df.height == 5  # one output row per input row
    assert df["diseasePhenotypeScore"].to_list() == [0.8, 0.8, 0.7, 0.5, 0.1]  # untouched
    assert df["prismDiseasePhenotypeScore"].to_list() == pytest.approx([0.95, 0.95, 0.2, 0.4, None])
    assert df["prismDiseaseId"].to_list() == ["OMIM:154700", "OMIM:154700", "OMIM:3", "OMIM:2", None]
    assert df["prismRank"].to_list() == [1, 1, 4, 2, None]
