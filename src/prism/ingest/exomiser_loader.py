"""Exomiser parquet ingestion: parse Exomiser output into a list of ExomiserCandidates.

Exomiser stores its results in a parquet file where each row is one variant contributing
to a (gene, mode-of-inheritance) candidate. This loader aggregates those rows into one
ExomiserCandidate per (rank, gene, MOI) group.

Key parsing decisions:

1. DISEASE SELECTION — `_pick_disease()`
   Exomiser stores all known diseases for a gene in the `associatedDiseases` column,
   regardless of the candidate's MOI. We pick the one whose inheritanceMode matches the
   candidate's MOI. For example, FGFR2 appears as both an AD and AR candidate; we want
   "Pfeiffer syndrome" for the AD candidate, not a recessive FGFR2 condition.
   Fall-through: if no exact MOI match, try AD+AR combined, then fall back to first disease.

2. VARIANT AGGREGATION
   Only rows where `isContributingVariant=True` are collected as variants. These are the
   variants Exomiser considers most relevant to the candidate's pathogenicity score.

3. ACMG CLASSIFICATION
   "NOT_AVAILABLE" is normalised to None so downstream code can treat None as "unclassified".
"""
from pathlib import Path

import polars as pl

from prism.models.exomiser import ExomiserCandidate, Variant

# Maps the parquet moi string to the inheritanceMode bytes found in associatedDiseases
_MOI_TO_INHERITANCE: dict[str, bytes] = {
    "AD": b"AUTOSOMAL_DOMINANT",
    "AR": b"AUTOSOMAL_RECESSIVE",
    "XD": b"X_DOMINANT",
    "XR": b"X_RECESSIVE",
    "MT": b"MITOCHONDRIAL",
}


def _pick_diseases(
    diseases: list[dict], moi: str
) -> list[tuple[str | None, str | None]]:
    """Return ALL diseases matching this candidate's MOI, so each gets its own candidate.

    Context: a gene like ACTG1 can be associated with multiple diseases under the
    same MOI (e.g. Deafness AD and Baraitser-Winter syndrome AD). Returning only the
    first would silently drop the correct diagnosis. Expanding to one candidate per
    disease lets PRISM's phenotype fit score pick the right one.

    Fall-through logic:
    1. "ANY" (phenotype-only mode) — return all diseases; no MOI constraint is known.
    2. Exact MOI match — collect all diseases with matching inheritanceMode.
    3. AUTOSOMAL_DOMINANT_AND_RECESSIVE — counts for both AD and AR candidates.
    4. Fallback to first disease only when no MOI match at all.
    """
    if moi == "ANY":
        return [(d["diseaseId"], d["diseaseName"]) for d in diseases] or [(None, None)]

    target = _MOI_TO_INHERITANCE.get(moi)
    matched = [
        (d["diseaseId"], d["diseaseName"])
        for d in diseases
        if d["inheritanceMode"] == target
    ]
    if not matched and moi in ("AD", "AR"):
        matched = [
            (d["diseaseId"], d["diseaseName"])
            for d in diseases
            if d["inheritanceMode"] == b"AUTOSOMAL_DOMINANT_AND_RECESSIVE"
        ]
    if not matched and diseases:
        matched = [(diseases[0]["diseaseId"], diseases[0]["diseaseName"])]
    return matched or [(None, None)]


def _decode(value: bytes | str | None) -> str | None:
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    return value or None


def _build_variant(row: dict) -> Variant:
    chrom = row.get("contigName") or ""
    pos = row.get("start") or ""
    ref = row.get("ref") or ""
    alt = row.get("alt") or ""
    variant_id = f"{chrom}-{pos}-{ref}-{alt}" if all([chrom, pos, ref, alt]) else None
    acmg = _decode(row.get("acmgClassification"))
    if acmg == "NOT_AVAILABLE":
        acmg = None
    return Variant(
        variant_id=variant_id,
        acmg=acmg,
        pathogenicity_score=row.get("maxPathScore"),
        consequence=_decode(row.get("functionalClass")),
    )


def load_exomiser(exomiser_parquet_path: Path, top_n: int | None = None) -> list[ExomiserCandidate]:
    df = pl.read_parquet(exomiser_parquet_path)

    # Gene-level scores are identical across all rows sharing (rank, geneSymbol, moi)
    gene_agg = df.group_by(["rank", "geneSymbol", "moi"], maintain_order=True).agg(
        pl.col("geneCombinedScore").first(),
        pl.col("genePhenotypeScore").first(),
        pl.col("geneVariantScore").first(),
        pl.col("associatedDiseases").first(),
    ).sort("rank")

    # Phenotype-only runs use moi="ANY" and penalise geneCombinedScore because there
    # are no variants. Use genePhenotypeScore as the ranking score in that case.
    gene_agg = gene_agg.with_columns(
        pl.when(pl.col("moi") == "ANY")
        .then(pl.col("genePhenotypeScore"))
        .otherwise(pl.col("geneCombinedScore"))
        .alias("_score")
    )

    # Filter to top-N genes before the Python loop so we don't build Pydantic objects
    # for thousands of genes that will be discarded. Critical for phenotype-only parquets
    # which can have 16k+ genes all with moi="ANY".
    #
    # Standard mode: genes have unique ranks — filter by top-N unique rank values so all
    #   MOIs of the same gene are kept (FGFR2-AD and FGFR2-AR share the same rank).
    # Phenotype-only (moi="ANY"): Exomiser assigns the same rank to many tied genes so
    #   unique-rank filtering keeps everything. Filter by top-N gene symbols by score instead.
    if top_n is not None:
        phenotype_only = (gene_agg["moi"] == "ANY").all()
        if phenotype_only:
            top_genes = (
                gene_agg.sort("_score", descending=True)
                .head(top_n)["geneSymbol"]
                .to_list()
            )
            gene_agg = gene_agg.filter(pl.col("geneSymbol").is_in(top_genes))
            df = df.filter(pl.col("geneSymbol").is_in(top_genes))
        else:
            top_ranks = gene_agg["rank"].unique().sort().head(top_n).to_list()
            gene_agg = gene_agg.filter(pl.col("rank").is_in(top_ranks))
            df = df.filter(pl.col("rank").is_in(top_ranks))

    # Collect contributing variant fields per candidate
    contrib_agg = (
        df.filter(pl.col("isContributingVariant"))
        .group_by(["rank", "geneSymbol", "moi"], maintain_order=True)
        .agg(
            pl.struct(["contigName", "start", "ref", "alt", "acmgClassification", "maxPathScore", "functionalClass"])
            .alias("contributing_variants")
        )
    )

    combined = (
        gene_agg
        .join(contrib_agg, on=["rank", "geneSymbol", "moi"], how="left")
        .sort("rank")
    )

    candidates: list[ExomiserCandidate] = []
    for row in combined.iter_rows(named=True):
        diseases = row["associatedDiseases"] or []
        matched_diseases = _pick_diseases(diseases, row["moi"])
        variants = [_build_variant(v) for v in (row["contributing_variants"] or [])]
        for disease_id, disease_name in matched_diseases:
            candidates.append(
                ExomiserCandidate(
                    gene_symbol=row["geneSymbol"],
                    disease_id=disease_id,
                    disease_name=disease_name,
                    moi=row["moi"],
                    exomiser_rank=row["rank"],
                    combined_score=row["_score"],
                    phenotype_score=row["genePhenotypeScore"],
                    variant_score=row["geneVariantScore"],
                    variants=variants,
                )
            )
    return candidates