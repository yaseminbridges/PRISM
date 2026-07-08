"""Intermediate data models produced during phenotype matching (C1).

This file defines the structures that C1 (partition.py) fills in and C2 (rescore.py)
scores. These are the core intermediate representations — everything between the raw
Exomiser output and the final RankedReport.

  DiseaseFeature — one HPO feature from a disease profile, with frequency class and IC.
  FeatureMatch   — a pairing of a patient term with a disease feature, with a relation
                   label (exact / subsumed / partial) and provenance text.
  FitEvidence    — the full match sheet for one candidate disease, grouped into buckets:
                     matched        (patient term aligns to disease feature)
                     partial        (LLM-resolved broader patient term)
                     unexplained    (patient terms the disease doesn't account for)
                     expected_absent (disease features the patient explicitly lacks)
                     age_excused    (expected_absent excused by developmental timing — C3)
                     contradictions (disease says absent, patient has it)
                     fit_score      (computed by C2, None until C2 runs)
"""
from typing import Literal
from pydantic import BaseModel
from prism.models.phenopacket import HpoTerm


class DiseaseFeature(BaseModel):
    hpo_id: str
    hpo_label: str
    frequency_class: str | None
    ic: float
    source: str


class FeatureMatch(BaseModel):
    patient_term: HpoTerm
    disease_feature: DiseaseFeature
    relation: Literal["exact", "subsumed", "partial"]
    provenance: str  # retrieved KB sentence the LLM grounded on


class FitEvidence(BaseModel):
    disease_id: str
    matched: list[FeatureMatch] = []
    partial: list[FeatureMatch] = []
    unexplained: list[HpoTerm] = []             # patient terms this disease does not cover
    expected_absent: list[DiseaseFeature] = []  # cardinal feats assessed-absent in patient
    age_excused: list[DiseaseFeature] = []      # absent but onset > patient age -> monitor
    contradictions: list[DiseaseFeature] = []   # disease-EXCLUDED feature present in patient
    fit_score: float | None = None