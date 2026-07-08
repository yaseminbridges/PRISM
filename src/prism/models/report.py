"""Final output data models returned by the PRISM pipeline.

  MechanismVerdict    — C5 result: does the variant evidence support the disease?
                        verdict is one of: congruent / incongruent / uncertain.
  ReRankedCandidate   — one candidate after PRISM processing: wraps the original
                        ExomiserCandidate with PRISM's FitEvidence, new rank,
                        rationale string, and mechanism verdict.
  DisambiguationReport — C4 result: which of the overlapping candidates is best
                         supported and why (discriminating features + rationale).
  RankedReport        — the top-level output for one patient case, containing the
                        full re-ranked list, optional disambiguation, callable
                        diagnoses, and LLM narratives.
"""
from typing import Literal
from pydantic import BaseModel
from prism.models.exomiser import ExomiserCandidate
from prism.models.candidate import FitEvidence


class MechanismVerdict(BaseModel):
    """C5 output: does the variant mechanism support this disease candidate?"""
    verdict: Literal["congruent", "incongruent", "uncertain"]
    reason: str  # human-readable explanation for the scientist


class ReRankedCandidate(BaseModel):
    candidate: ExomiserCandidate
    fit: FitEvidence
    old_rank: int
    new_rank: int
    rationale: str             # human-readable C2 movement summary
    mechanism: MechanismVerdict | None = None  # C5 verdict; None = not yet assessed


class DisambiguationReport(BaseModel):
    candidate_disease_ids: list[str]
    discriminating_features: list[str]  # HPO IDs that distinguish the candidates
    best_supported: str | None          # disease_id of best-supported candidate, or None
    rationale: str


class RankedReport(BaseModel):
    case_id: str
    reranked: list[ReRankedCandidate]
    disambiguation: DisambiguationReport | None
    callable_diagnoses: list[ReRankedCandidate]  # ≤2, or empty (empty is valid & preferred)
    narratives: dict[str, str] = {}              # disease_id → LLM narrative for callables