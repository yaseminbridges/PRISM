"""Exomiser output data models.

This file defines the structures that hold Exomiser's ranked output after it has been
parsed from its parquet format by ingest/exomiser_loader.py.

  Variant           — a single contributing genetic variant for a candidate gene.
  ExomiserCandidate — one ranked gene/disease/MOI combination from Exomiser, with
                      combined score, phenotype score, variant score, and variant list.

Each row in the Exomiser parquet corresponds to a (gene, mode-of-inheritance) pair —
a gene can appear multiple times if it is plausible under both AD and AR inheritance.
The loader collapses these into one ExomiserCandidate per pair.
"""
from pydantic import BaseModel


class Variant(BaseModel):
    variant_id: str
    acmg: str | None
    pathogenicity_score: float | None
    consequence: str | None


class ExomiserCandidate(BaseModel):
    gene_symbol: str
    disease_id: str | None
    disease_name: str | None
    moi: str | None
    exomiser_rank: int
    combined_score: float
    phenotype_score: float
    variant_score: float
    variants: list[Variant]