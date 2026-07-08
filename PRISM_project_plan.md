# PRISM - Phenotype Re-ranking via Interpretable Semantic Matching

**A phenotype-fit reasoning layer on top of Exomiser for rare-disease diagnosis.**

---

## 0. Thesis & non-goals

### The pivot (why this replaces the previous multi-agent build)

The earlier system re-derived gene–disease association strength using OMIM/ClinGen/PubMed and classified links as stronger/weaker. That is **redundant with Exomiser**, which already folds gene–disease evidence into prioritisation (ClinVar whitelisting, ACMG since v14, the variant score itself). Re-deriving it adds cost and no signal.

The leverage is on the **phenotype axis**. Exomiser's phenotype score (hiPHIVE/PHIVE) is a single Resnik/Phenodigm semantic-similarity number over a bag of HPO terms, smoothed across the PPI network. That number structurally discards what a clinical scientist reasons over:

- **Frequency** — obligate vs occasional features of the same disease score about the same.
- **Missing hallmarks** — expected-but-absent cardinal features aren't penalised; excluded terms are barely used.
- **Age/onset** — no model to separate "genuinely absent" from "not manifested yet."
- **Overlap/discrimination** — when two diseases share broad terms (DD, ID, dysmorphism) and the patient only *has* broad terms, similarity scores both high and can't separate them. This is the Pfeiffer / poorly-characterised-DDD problem.

**PRISM does not re-rank genes from scratch and does not replace Exomiser.** It consumes Exomiser's ranked shortlist + the patient phenopacket and produces (a) an auditable per-candidate phenotype-fit sheet, (b) a conservative re-rank of the shortlist using frequency-, specificity-, and age-aware fit, and (c) a disambiguation report for overlapping candidates.

### Non-goals (state these explicitly so scope doesn't creep)

- Not re-deriving gene–disease association strength.
- Not replacing Exomiser's variant/gene prioritisation.
- Not a multi-agent swarm. One orchestrating reasoner + deterministic tools, with a focused agentic loop *only* in the disambiguator (C4) where iterative KB retrieval earns its keep.
- Not a global re-score. Re-rank **within Exomiser's top-N** only.

---

## 1. Central design rule: the deterministic / LLM boundary

This is the most important constraint in the codebase. Get it wrong and PhEval is unreproducible and the method is undefensible.

**The LLM never emits a rank-affecting number.** It emits structured, provenance-tagged judgements; a deterministic function turns those into the score.

| Layer | Owns | Lives in |
|---|---|---|
| **Deterministic (Python)** | All KB retrieval; IC computation; frequency-class → weight mapping; ontology-aware set operations (subsumption, set-difference for discriminators); the re-score arithmetic; age-gating logic; anything touching the final rank number. | `*/tool.py`, `components/`, `ontology/` |
| **LLM (grounded in retrieved text)** | Resolving ambiguous matches Python can't settle (is patient term X *plausibly* a manifestation of disease feature Y when not ontologically identical?); narrative synthesis of the fit sheet; disambiguation reasoning; flagging "too thin to call." | `*/skill.md`, `reasoning/` |

**Contract between them:** the LLM resolves each patient term / disease feature into one of `{matched, partial, absent_expected, unexplained, contradiction}` **with a provenance pointer to the retrieved KB sentence**. The deterministic scorer (C2) consumes those labels and produces the number. Same labels in → same score out, always.

Carry forward the existing domain-first pattern: each knowledge base is a folder with a deterministic `tool.py` (retrieval) and a `skill.md` (reasoning instructions the LLM is bound to).

---

## 2. Environment constraints (GEL Research Environment)

Assume PRISM runs **inside the GEL Research Environment**, which has implications Claude Code must respect from day one:

- **Knowledge bases are offline/local.** No live PubMed / Crossref / Wikipedia / external API calls inside GEL. HPOA, HPO ontology, Orphanet, and PanelApp dumps must be packaged as local files and accessed through retrievers. An optional `online_enrichment` mode may hit external APIs **only** for dev outside GEL — gated behind a config flag, off by default.
- **LLM endpoint is an OPEN DEPENDENCY (see §8).** Confirm what model the data can legally reach inside GEL before building the reasoning layer. Abstract it behind an `LLMClient` interface so the backend (in-environment host, approved API, or local weights) is swappable.
- **Packaging** follows existing practice: Docker built `--platform linux/amd64`, pushed via Docker Hub/Artifactory, run under Singularity; orchestration via Nextflow on LSF/bsub. PRISM's PhEval runner must be invocable as a Nextflow process.

---

## 3. Architecture & data flow

```
phenopacket (HPO observed + EXCLUDED, age, onset, sex)
      +
Exomiser output (ranked genes/diseases, variant + pheno scores, ACMG, MOI)
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  [ingest]  normalise both into pydantic models
        │
        ▼
  [ontology]  HPO IC, ancestors, onset terms; HPOA frequency → weights
        │
        ▼
  [knowledge]  per candidate disease, retrieve profile (offline):
               HPOA (freq classes, disease-excluded feats),
               Orphanet (age-of-onset, clinical summary), PanelApp (gene-panel context)
        │
        ▼
  [reasoning]  LLM resolves ambiguous matches → labelled FitEvidence
        │
        ▼
  [components]
     C1 partition   explained / unexplained / expected-but-absent / contradiction
     C2 rescore     freq × specificity, age-gated penalty  → fit_score
     C3 age-gate    onset model; excuse late-onset absences
     C4 disambiguate discriminating-feature analysis over overlapping candidates
     C5 mechanism   (optional) variant-mechanism ↔ phenotype congruence
        │
        ▼
  [output]  per-candidate fit sheet + conservative re-rank + disambiguation report
            + ≤2 callable diagnoses (or none — empty is a valid, preferred answer)
        │
        ▼
  [eval]  PhEval plugin: top-1/3/10, MRR vs baseline; overall + broad-phenotype subset
```

---

## 4. Repository layout

Two repos, mirroring the existing dx / pheval split.

### `prism` — core library

```
PRISM/
  pyproject.toml
  README.md
  src/prism/
    __init__.py
    models/
      phenopacket.py      # normalised patient model (observed + excluded + onset)
      exomiser.py         # normalised Exomiser candidate model
      candidate.py        # FitEvidence working object that accumulates evidence
      report.py           # output contracts: fit sheet, ranked report, disambiguation
    ingest/
      phenopacket_loader.py   # GA4GH phenopacket -> models.phenopacket
      exomiser_loader.py      # reuse existing Polars parquet/tsv extractor -> models.exomiser
    ontology/
      hpo.py              # IC, ancestors/descendants, onset terms (pyhpo or local graph)
      frequency.py        # HPOA frequency class -> numeric weight g(freq)
    knowledge/            # domain-first: each KB = tool.py (deterministic) + skill.md (LLM)
      hpoa/      { tool.py, skill.md }   # disease phenotype profiles, freq + disease-excluded feats
      orphanet/  { tool.py, skill.md }   # incl. age-of-onset + clinical summary
      panelapp/  { tool.py, skill.md }   # gene-panel context
    components/
      partition.py        # C1
      rescore.py          # C2  (PURE deterministic — reproducible)
      age_gate.py         # C3
      disambiguate.py     # C4  (agentic loop allowed here)
      mechanism.py        # C5  (optional, later)
    reasoning/
      agent.py            # orchestration: when to call which component
      llm.py              # LLMClient interface (swappable backend)
      prompts/            # the skill.md files the LLM is bound to
    pipeline.py           # top-level: (phenopacket, exomiser_output) -> RankedReport
    cli.py
  tests/
    fixtures/             # synthetic phenopackets + exomiser outputs + KNOWN expected answers
    test_ingest.py
    test_rescore.py       # deterministic -> exact assertions
    test_partition.py
    test_disambiguate.py
```

### `prism-pheval` — PhEval integration

```
prism-pheval/
  src/prism_pheval/
    runner.py             # PhEval runner plugin: case -> Exomiser -> [PRISM] -> standardised results
    post_process.py       # parse PRISM RankedReport into PhEval result format
  config/
    prism_pheval_config.yaml
  nextflow/
    main.nf               # invoke as Nextflow process on LSF
  README.md
```

---

## 5. Data contracts (pydantic — the spine)

```python
# models/phenopacket.py
class HpoTerm(BaseModel):
    id: str                      # "HP:0001250"
    label: str
    onset_age_years: float | None = None
    excluded: bool = False       # explicitly ASSESSED-absent

class PatientPhenotype(BaseModel):
    subject_id: str
    sex: Literal["male", "female", "unknown"]
    age_years: float | None = None        # resolved from ISO8601 age
    observed_terms: list[HpoTerm]
    excluded_terms: list[HpoTerm]
    # INVARIANT: absence != excluded. A term not present is "not assessed",
    # NOT "absent". Only `excluded_terms` are assessed-absent. This distinction
    # is load-bearing for every missing-hallmark penalty downstream.
```

```python
# models/exomiser.py
class Variant(BaseModel):
    hgvs: str | None
    acmg: str | None             # P / LP / VUS / LB / B
    pathogenicity_score: float | None
    consequence: str | None      # for C5 mechanism inference

class ExomiserCandidate(BaseModel):
    gene_symbol: str
    disease_id: str | None       # OMIM:* / ORPHA:*
    disease_name: str | None
    moi: str | None
    exomiser_rank: int
    combined_score: float
    phenotype_score: float
    variant_score: float
    variants: list[Variant]
```

```python
# models/candidate.py — working object PRISM builds up per candidate
class DiseaseFeature(BaseModel):
    hpo_id: str
    label: str
    frequency_class: str | None  # Obligate / VeryFrequent / Frequent / Occasional / VeryRare / Excluded
    ic: float
    source: str                  # "HPOA" | "Orphanet"

class FeatureMatch(BaseModel):
    patient_term: HpoTerm
    disease_feature: DiseaseFeature
    relation: Literal["exact", "subsumed", "partial"]
    provenance: str              # retrieved KB sentence the LLM grounded on

class FitEvidence(BaseModel):
    disease_id: str
    matched: list[FeatureMatch]
    partial: list[FeatureMatch]
    unexplained: list[HpoTerm]           # patient terms this disease does not cover
    expected_absent: list[DiseaseFeature] # cardinal feats assessed-absent in patient
    age_excused: list[DiseaseFeature]     # absent but onset > patient age -> monitor
    contradictions: list[DiseaseFeature]  # disease-EXCLUDED feature present in patient
    fit_score: float | None = None
```

```python
# models/report.py
class ReRankedCandidate(BaseModel):
    candidate: ExomiserCandidate
    fit: FitEvidence
    old_rank: int
    new_rank: int
    rationale: str               # human-readable, for the scientist

class RankedReport(BaseModel):
    case_id: str
    reranked: list[ReRankedCandidate]
    disambiguation: "DisambiguationReport | None"
    callable_diagnoses: list[ReRankedCandidate]   # ≤2, or empty (empty is valid & preferred)
```

---

## 6. Component specs

### C1 — Partition (the fit sheet) · usability deliverable, no rank change

For each Exomiser top-N candidate, retrieve the disease profile and partition evidence into the `FitEvidence` buckets. Output a per-candidate sheet a scientist signs off in seconds:

- **Explains:** matched patient terms, each with frequency class, IC, and the KB sentence grounding the match.
- **Doesn't explain:** unexplained patient terms (salient features this disease ignores).
- **Expected but absent:** cardinal disease features the patient was assessed-absent for.
- **Contradictions:** disease-excluded features (HPOA freq 0%) the patient *has* — strong evidence against.

LLM role here is bounded: resolve non-exact matches and write the narrative, grounded only in retrieved text. No invented features.

### C2 — Re-score · the PhEval-measurable contribution (PURE deterministic)

Re-rank **within Exomiser's top-N**. Default mode is **conservative** (protects the safety metric).

```
g(freq):  Obligate→1.0  VeryFrequent→0.9  Frequent→0.5
          Occasional→0.2  VeryRare→0.05  Excluded→ contradiction signal

For candidate D, patient P:
  fit_raw = 0
  for f in matched(D):          fit_raw += w_match * g(freq_f) * ic_f
  for f in partial(D):          fit_raw += w_partial * g(freq_f) * ic_f
  for f in expected_absent(D)   # only if age-assessed, see C3
     and not age_excused:       fit_raw -= w_miss * g(freq_f) * ic_f
  for t in unexplained(D):      fit_raw -= w_unexp * ic_t
  for f in contradictions(D):   fit_raw -= w_contra            # disease-excluded feature present

  fit_score = normalise(fit_raw)           # ~[-1, 1]

Re-rank key:
  conservative (default): reorder only when |Δfit| > margin AND Exomiser scores near-tie
  aggressive  (config):   key = α * rank_norm(S_exo) + (1-α) * fit_score
```

Weights `w_*` and `α` tuned on a dev split (§7). **Do not throw away the Exomiser score** — blend or use as primary, never discard.

### C3 — Age gate · keeps C2's penalty honest

Pull onset distributions (HPO onset terms, Orphanet age-of-onset). For each expected-but-absent feature: if typical onset > patient age, **excuse** the absence (move to `age_excused`, flag "monitor", zero penalty). Present-but-too-early features become evidence for a severe allele / against the call. This is what makes the missing-hallmark penalty safe rather than noisy.

### C4 — Disambiguator · the Pfeiffer / DDD core (agentic loop allowed)

Trigger when top-N contains ≥2 candidates with high pairwise phenotype similarity. Steps:

1. Compute **discriminating features** = set-difference over the candidates' disease profiles (deterministic).
2. Check the patient — **including excluded terms** — for each discriminator.
3. If the phenopacket is silent on a key discriminator, the agent may iteratively pull more KB evidence (the one place a loop is justified).
4. Report the best-supported candidate **and what to check next**: e.g. syndactyly → Apert; broad thumbs/great toes → Pfeiffer; normal hands → Crouzon.

### C5 — Mechanism ↔ phenotype congruence · optional, phenotype-adjacent

Same gene, different mechanism → different syndrome (e.g. FGFR2 GoF craniosynostosis vs LoF LADD). Exomiser scores variant pathogenicity but never asks whether the *mechanism implied by this variant* matches the *phenotype in front of you*. Infer mechanism from `Variant.consequence` + known gene mechanism, check congruence. Build only after C1–C4 land.

---

## 7. Evaluation design (the PhEval story)

- **Baseline:** stock Exomiser ranking on a corpus (PhenopacketStore and/or a 100KGP subset).
- **Treatment:** Exomiser → PRISM re-rank.
- **Metrics:** top-1, top-3, top-10, MRR (PhEval computes these).
- **Stratify — the honest-science part.** Report overall **and** on the **broad/overlapping-phenotype subset**, defined operationally as cases where either (a) the patient HPO set is small / high in the ontology (low mean IC), or (b) Exomiser top-N contains ≥2 diseases with high pairwise similarity. Claim the lift where the mechanism predicts it; show it's **neutral, not regressive**, elsewhere.
- **Ablation:** C2 / C3 / C4 on-off to attribute the delta. This is what makes it a method, not a vibe.
- **Safety metric (must be ~0):** rate at which PRISM demotes the **true** diagnosis out of top-1. Conservative mode exists to keep this near zero.

---

## 8. Open questions / dependencies (resolve before the dependent phase)

1. **LLM endpoint inside GEL** — what model can the data legally reach? Blocks the reasoning layer (C1+). Until confirmed, build C0–C2-deterministic against a mock `LLMClient`.
2. **HPOA frequency coverage** — frequency is coarse and often missing. Caps C2/C3 coverage; decide the fallback weight when `frequency_class is None`.
3. **Phenopacket absence semantics** — real 100KGP phenopackets rarely distinguish assessed-absent from not-assessed. Until excluded terms are reliable, treat absence as **soft** evidence, not a hard subtractor (don't over-penalise on missing data).

---

## 9. Phasing & decision gates

| Phase | Build                                                                                                       | Deliverable | Gate |
|---|-------------------------------------------------------------------------------------------------------------|---|---|
| **0 — Spine** | models, ingest (reuse Polars extractor), ontology layer, HPOA retriever end-to-end on fixtures. No scoring. | Load phenopacket + Exomiser output, retrieve & dump disease profiles. | Plumbing works on N fixture cases. |
| **1 — C1 Partition** | the fit sheet. LLM resolves matches, grounded.                                                              | Human-reviewable per-candidate sheet. Standalone-valuable. | A scientist can sign off a sheet faster than reading raw Exomiser output. |
| **2 — C2 + PhEval** | deterministic re-score, conservative re-rank, wire `prism-pheval`.                                          | First measurable rank delta. | **KILL-SWITCH: does it move rank on the broad subset WITHOUT regressing overall & with safety-metric ≈ 0? If no, stop and rethink before building more.** |
| **3 — C3 Age gate** | onset model excuses late-onset absences.                                                                    | Fewer false demotions. | Safety metric improves or holds. |
| **4 — C4 Disambiguator** | discriminating-feature analysis + loop.                                                                     | Pfeiffer/DDD disambiguation report. | Lift concentrated in the overlap subset. |
| **5 — C5 + orchestration** | mechanism congruence; optional run-Exomiser-end-to-end + CaseQuerier over PhenopacketStore for demo.        | Polish / demo. | — |

Front-loads de-risking; Phase 2 is the go/no-go.

---

## 10. Phase 0 — first tasks for Claude Code

1. Scaffold `prism` per §4 with `pyproject.toml` (Python 3.11+, pydantic v2, polars, pyhpo or local HPO graph, pytest).
2. Implement `models/` exactly per §5, with the absence-vs-excluded invariant enforced in `PatientPhenotype`.
3. `ingest/phenopacket_loader.py`: GA4GH phenopacket JSON → `PatientPhenotype`, resolving ISO8601 age → `age_years` and capturing excluded terms + per-term onset.
4. `ingest/exomiser_loader.py`: adapt the existing Polars parquet/tsv extractor → `list[ExomiserCandidate]`.
5. `ontology/hpo.py` (IC, ancestors, onset terms) and `ontology/frequency.py` (`g(freq)` map).
6. `knowledge/hpoa/tool.py`: deterministic retrieval of a disease's phenotype profile (features + frequency class + disease-excluded features) from a **local** HPOA file; stub `skill.md`.
7. Build 3–5 `tests/fixtures/` cases (synthetic phenopacket + Exomiser output + known expected partition), including one deliberate **broad-phenotype overlap** case (e.g. Pfeiffer vs Crouzon vs Apert) for C4 later.
8. `pipeline.py` skeleton wiring ingest → retrieval → (empty) components → `RankedReport`, plus `cli.py`.
9. `reasoning/llm.py`: `LLMClient` interface with a **mock** backend so Phases 0–2-deterministic run without a live model.

Defer all LLM-calling code until §8.1 is resolved; everything in Phase 0 is deterministic and unit-testable with exact assertions.
