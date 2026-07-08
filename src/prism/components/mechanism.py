"""C5 — Mechanism congruence: does the variant evidence support the disease candidate?

Three-tier check, fully deterministic:

  1. Pathogenicity class: are any variants PATHOGENIC or LIKELY_PATHOGENIC?
     No → uncertain; all BENIGN/LIKELY_BENIGN → incongruent.

  2. Mode-of-inheritance × allele count: recessive disease needs ≥2 P/LP hits
     (compound het or homozygous). One P/LP alone → uncertain. One P/LP + VUS
     → uncertain with compound-het note (VUS could be the second hit).

  3. Consequence plausibility (informational, never causes incongruent alone):
     summarise LoF vs missense counts for the rationale string.

The LLM is NOT called here — consequence/GoF vs LoF inference for specific
genes is left to the disambiguator's LLM loop where authorised.
"""

from prism.models.exomiser import ExomiserCandidate, Variant
from prism.models.report import MechanismVerdict

# Variant Sequence Ontology consequences that predict loss-of-function
_LOF_SO = frozenset({
    "STOP_GAINED",
    "FRAMESHIFT_VARIANT",
    "SPLICE_DONOR_VARIANT",
    "SPLICE_ACCEPTOR_VARIANT",
    "START_LOST",
    "EXON_LOSS_VARIANT",
    "TRANSCRIPT_ABLATION",
})

_PATHOGENIC_ACMG = frozenset({"PATHOGENIC", "LIKELY_PATHOGENIC"})
_BENIGN_ACMG     = frozenset({"BENIGN", "LIKELY_BENIGN"})
_VUS_ACMG        = frozenset({"UNCERTAIN_SIGNIFICANCE", "VUS"})

# MOI strings Exomiser may emit — normalised to the canonical form after upper()
_RECESSIVE_MOI = frozenset({
    "AUTOSOMAL_RECESSIVE", "AR",
    "X_LINKED_RECESSIVE",  "XR",
})


def _fmt(v: Variant) -> str:
    """Short variant label: id (ACMG consequence)."""
    parts = [v.variant_id]
    if v.acmg:
        parts.append(f"[{v.acmg}]")
    if v.consequence:
        parts.append(v.consequence.replace("_", " ").lower())
    return " ".join(parts)


def infer_mechanism_congruence(candidate: ExomiserCandidate) -> MechanismVerdict:
    """Return a MechanismVerdict for a single re-ranked candidate.

    Args:
        candidate: ExomiserCandidate from Exomiser output (may have 0..N variants).

    Returns:
        MechanismVerdict with verdict in {congruent, incongruent, uncertain}
        and a human-readable reason string.
    """
    if not candidate.variants:
        return MechanismVerdict(
            verdict="uncertain",
            reason="No variant data attached to this candidate.",
        )

    acmg_classes = [(v.acmg or "").upper() for v in candidate.variants]
    plp  = [v for v, cls in zip(candidate.variants, acmg_classes) if cls in _PATHOGENIC_ACMG]
    vus  = [v for v, cls in zip(candidate.variants, acmg_classes) if cls in _VUS_ACMG]

    if not plp:
        named_classes = [cls for cls in acmg_classes if cls]
        if named_classes and all(cls in _BENIGN_ACMG for cls in named_classes):
            return MechanismVerdict(
                verdict="incongruent",
                reason=(
                    f"All {len(candidate.variants)} variant(s) classified "
                    f"BENIGN or LIKELY_BENIGN — this gene change does not support "
                    f"{candidate.disease_name or candidate.disease_id or 'this disease'}."
                ),
            )
        vus_note = (
            f"{len(vus)} VUS: {', '.join(_fmt(v) for v in vus)}" if vus else "no VUS"
        )
        return MechanismVerdict(
            verdict="uncertain",
            reason=(
                f"No PATHOGENIC or LIKELY_PATHOGENIC variant found "
                f"({len(candidate.variants)} variant(s) assessed; {vus_note})."
            ),
        )

    # MOI × allele-count check
    moi_raw = (candidate.moi or "").upper().replace("-", "_").replace(" ", "_")
    if moi_raw in _RECESSIVE_MOI and len(plp) < 2:
        # Check for possible compound het: 1 P/LP + ≥1 VUS could be the second hit
        if vus:
            return MechanismVerdict(
                verdict="uncertain",
                reason=(
                    f"Recessive disease (MOI={candidate.moi}): "
                    f"1 P/LP ({_fmt(plp[0])}) + "
                    f"{len(vus)} VUS ({', '.join(_fmt(v) for v in vus)}) — "
                    f"possible compound het, pathogenicity of VUS unresolved."
                ),
            )
        return MechanismVerdict(
            verdict="uncertain",
            reason=(
                f"Recessive disease (MOI={candidate.moi}) but only "
                f"{len(plp)} P/LP allele identified ({_fmt(plp[0])}); "
                f"biallelic hits expected."
            ),
        )

    # Consequence summary (informational)
    lof_n = sum(1 for v in plp if (v.consequence or "").upper() in _LOF_SO)
    mis_n = len(plp) - lof_n
    plp_ids = ", ".join(_fmt(v) for v in plp)
    parts = [f"{len(plp)} P/LP variant(s): {plp_ids}"]
    if lof_n:
        parts.append(f"{lof_n} LoF")
    if mis_n:
        parts.append(f"{mis_n} missense/other")
    if vus:
        parts.append(f"{len(vus)} VUS also present: {', '.join(_fmt(v) for v in vus)}")
    if moi_raw:
        parts.append(f"MOI={candidate.moi}")

    return MechanismVerdict(verdict="congruent", reason=", ".join(parts) + ".")