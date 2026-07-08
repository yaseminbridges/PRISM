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


def _pick_disease(
    diseases: list[dict], moi: str
) -> tuple[str | None, str | None]:
    """Return the disease ID/name that best matches this candidate's MOI.

    Context: Exomiser scores each gene once per mode of inheritance it could
    plausibly act under (e.g. FGFR2 gets separate AD and AR candidates).
    The parquet stores ALL known diseases for the gene in `associatedDiseases`
    regardless of MOI, so we have to select the one whose inheritanceMode
    matches the candidate's MOI to get the right disease label (e.g. Pfeiffer
    syndrome for FGFR2-AD rather than a recessive FGFR2 condition).

    Fall-through logic:
    1. Exact MOI match — normal case.
    2. AUTOSOMAL_DOMINANT_AND_RECESSIVE — some OMIM entries list a disease
       under both inheritance modes as a single value; counts for AD or AR.
    3. Fallback to first disease — handles edge cases where a gene appears
       under an unexpected MOI (e.g. PAH ranked AD despite all its diseases
       being AR); preserves a label rather than returning None.
    """
    target = _MOI_TO_INHERITANCE.get(moi)
    for d in diseases:
        if d["inheritanceMode"] == target:
            return d["diseaseId"], d["diseaseName"]
    # Some OMIM entries cover both AD and AR under a single inheritanceMode value
    if moi in ("AD", "AR"):
        for d in diseases:
            if d["inheritanceMode"] == b"AUTOSOMAL_DOMINANT_AND_RECESSIVE":
                return d["diseaseId"], d["diseaseName"]
    # Fallback: gene appears under an MOI with no matching disease entry
    if diseases:
        return diseases[0]["diseaseId"], diseases[0]["diseaseName"]
    return None, None


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


def load_exomiser(exomiser_parquet_path: Path) -> list[ExomiserCandidate]:
    df = pl.read_parquet(exomiser_parquet_path)

    # Gene-level scores are identical across all rows sharing (rank, geneSymbol, moi)
    gene_agg = df.group_by(["rank", "geneSymbol", "moi"], maintain_order=True).agg(
        pl.col("geneCombinedScore").first(),
        pl.col("genePhenotypeScore").first(),
        pl.col("geneVariantScore").first(),
        pl.col("associatedDiseases").first(),
    )

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
        disease_id, disease_name = _pick_disease(diseases, row["moi"])
        variants = [_build_variant(v) for v in (row["contributing_variants"] or [])]
        candidates.append(
            ExomiserCandidate(
                gene_symbol=row["geneSymbol"],
                disease_id=disease_id,
                disease_name=disease_name,
                moi=row["moi"],
                exomiser_rank=row["rank"],
                combined_score=row["geneCombinedScore"],
                phenotype_score=row["genePhenotypeScore"],
                variant_score=row["geneVariantScore"],
                variants=variants,
            )
        )
    return candidates