"""Write PRISM scores back into an Exomiser results parquet as extra columns.

Exomiser parquets have one row per (gene, MOI, variant); PRISM scores one candidate per
(gene, MOI, disease). For each (gene, MOI) we take PRISM's best-ranked disease and add:

  prismDiseasePhenotypeScore — that disease's PRISM fit_score (0–1)
  prismDiseaseId             — the disease the score refers to
  prismRank                  — PRISM's new_rank for that candidate

Rows for genes PRISM did not score (e.g. outside --top-n) get nulls. All original
columns, including Exomiser's own diseasePhenotypeScore, are left untouched.
"""
from pathlib import Path

import polars as pl

from prism.models.report import RankedReport


def prism_score_table(report: RankedReport) -> pl.DataFrame:
    """One row per (geneSymbol, moi): PRISM's best-ranked disease and its fit score."""
    best: dict[tuple[str, str], tuple[int, float | None, str | None]] = {}
    for rc in report.reranked:
        key = (rc.candidate.gene_symbol, rc.candidate.moi)
        if key not in best or rc.new_rank < best[key][0]:
            best[key] = (rc.new_rank, rc.fit.fit_score, rc.candidate.disease_id)
    return pl.DataFrame(
        {
            "geneSymbol": [k[0] for k in best],
            "moi": [k[1] for k in best],
            "prismRank": [v[0] for v in best.values()],
            "prismDiseasePhenotypeScore": [v[1] for v in best.values()],
            "prismDiseaseId": [v[2] for v in best.values()],
        },
        schema={
            "geneSymbol": pl.Utf8,
            "moi": pl.Utf8,
            "prismRank": pl.Int64,
            "prismDiseasePhenotypeScore": pl.Float64,
            "prismDiseaseId": pl.Utf8,
        },
    )


def write_exomiser_with_prism_scores(
    exomiser_path: Path, report: RankedReport, output_path: Path
) -> None:
    """Copy the Exomiser parquet to output_path with the PRISM score columns appended."""
    df = pl.read_parquet(exomiser_path)
    scores = prism_score_table(report).with_columns(
        pl.col("geneSymbol").cast(df.schema["geneSymbol"]),
        pl.col("moi").cast(df.schema["moi"]),
    )
    out = df.join(scores, on=["geneSymbol", "moi"], how="left", maintain_order="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)
