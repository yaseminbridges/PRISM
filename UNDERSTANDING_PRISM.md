# Understanding PRISM

## What Is PRISM?

PRISM (Phenotype Re-ranking via Interpretable Semantic Matching) is a post-processing
layer that sits on top of **Exomiser**. Exomiser is a clinical genomics tool that takes a
patient's HPO phenotype terms and genetic variants and produces a ranked list of
candidate gene/disease pairs. PRISM takes that ranked list and re-ranks it using richer
phenotype reasoning — matching the patient's symptoms against curated disease profiles,
weighing how specific and informative each match is, and optionally asking an LLM to
resolve ambiguous cases.

---

## Inputs and Outputs

**Inputs (per patient case):**
- A **phenopacket** (JSON) — describes the patient: observed HPO terms, explicitly excluded
  HPO terms, sex, and age.
- An **Exomiser parquet file** — Exomiser's ranked gene/disease candidates, each with a
  combined score, phenotype score, variant score, and contributing variants.

**Output: `RankedReport`**
- `reranked` — all candidates re-ordered with PRISM fit scores
- `callable_diagnoses` — ≤2 high-confidence diagnosis calls (empty is valid)
- `disambiguation` — if two diseases looked similar, which one fits better and why
- `narratives` — LLM-generated plain-English summaries for the top candidates

---

## The End-to-End Pipeline

```
phenopacket.json  ──┐
                    ├──► [Ingest] ──► [C1 Partition] ──► [C3 Age Gate] ──►
exomiser.parquet  ──┘                                                      │
                                                                           ▼
                    ┌──────────────────── [C2 Rescore & Rerank] ◄──────────┘
                    │
                    ▼
              [C5 Mechanism] ──► [C4 Disambiguate] ──► [Callable call] ──► RankedReport
```

### Step 0 — Ingest
- `ingest/phenopacket_loader.py` parses the phenopacket JSON into a `PatientPhenotype`.
- `ingest/exomiser_loader.py` reads the Exomiser parquet into a list of `ExomiserCandidate`.
- The top N candidates (default 20) are kept.

### Step 1 — Knowledge loading
- The HPO ontology (`hp.obo`) is loaded into `HPOGraph`, which supports ancestor/descendant
  lookups and computes **information content (IC)** per term (how rare/specific a term is).
- `HpoaRetriever` loads `phenotype.hpoa` — the HPO Annotation database linking diseases to
  their HPO features with frequency information.
- `OrphanetRetriever` (optional) loads Orphanet XML for supplementary phenotype profiles,
  especially for ORPHA-prefixed disease IDs.

### C1 — Partition (`components/partition.py`)
For **each candidate disease**, the patient's observed HPO terms are compared against the
disease's known phenotype profile. Each patient term is sorted into one of five buckets:

| Bucket | Meaning |
|---|---|
| `matched` | Exact HPO ID match, or patient has a more specific term (descendant) |
| `partial` | Patient has a broader term than the disease expects — LLM decides if it counts |
| `unexplained` | Patient term has no ontological relationship to any disease feature |
| `expected_absent` | Disease feature the patient explicitly does NOT have |
| `contradictions` | Disease says a feature should be absent — patient has it |

The result is a `FitEvidence` object per candidate.

### C3 — Age Gate (`components/age_gate.py`)
If a disease feature is absent in the patient (`expected_absent`) but the disease typically
only manifests **after** the patient's current age, that absence is not a red flag — the
feature simply hasn't appeared yet. Those features are moved to `age_excused`, removing
their penalty in C2. If either age is unknown, the gate is skipped.

### C2 — Rescore (`components/rescore.py`)
Computes a `fit_score` for each candidate from their `FitEvidence`:

```
fit_raw = Σ w_match   × freq_weight × IC   (for matched features)
        + Σ w_partial × freq_weight × IC   (for partial matches)
        − Σ w_miss    × freq_weight × IC   (for expected_absent, not age_excused)
        − Σ w_unexp   × IC                 (for unexplained patient terms)
        − |contradictions| × w_contra

fit_score = tanh(fit_raw)   → bounded in (−1, 1)
```

The score is then used to re-rank candidates under one of two modes:
- **Conservative** (default): only swap two candidates if their Exomiser scores are nearly
  tied (within 0.05) AND PRISM's fit_score differs by more than 0.1.
- **Aggressive**: blend Exomiser score (70%) and PRISM fit_score (30%) to produce a new
  overall ranking.

### C5 — Mechanism Congruence (`components/mechanism.py`)
Checks whether the variant evidence supports the disease's expected mechanism:
- Are any variants PATHOGENIC or LIKELY_PATHOGENIC?
- If the disease is recessive, are there ≥ 2 P/LP alleles?
- Result: `congruent`, `incongruent`, or `uncertain`.

An `incongruent` verdict blocks the candidate from being a callable diagnosis.

### C4 — Disambiguate (`components/disambiguate.py`)
Triggered when ≥ 2 top candidates have highly overlapping phenotype profiles
(Jaccard similarity ≥ 0.3). PRISM:
1. Finds **discriminating features** — HPO terms present in one disease but not the others.
2. Checks whether the patient supports, contradicts, or is silent on each discriminator.
3. For silent discriminators, asks the LLM: is this feature likely present or absent?
4. Scores each disease and picks the best-supported one.

### Callable Diagnosis
A diagnosis is "called" only under strict rules:
- **If C4 ran**: the best-supported candidate must have `fit_score > 0.5` and mechanism ≠ incongruent.
- **If C4 did not run**: the top candidate must have `fit_score > 0.6`, lead the runner-up
  by > 0.15, and mechanism ≠ incongruent.

An empty `callable_diagnoses` is valid and preferred over a low-confidence call.

---

## The LLM's Role

The LLM is used in **two places only**, and never directly emits a number that affects scoring:

1. **C1 Partial match** — when the patient has a broader HPO term than the disease expects,
   the LLM labels the pair as `matched`, `partial`, `absent_expected`, `unexplained`, or
   `contradiction`. Only `matched`/`partial` counts toward the fit score.

2. **C4 Silent discriminators** — when two similar diseases both lack a patient signal on a
   discriminating feature, the LLM is asked whether that feature is likely present or absent.
   Its answer updates the disambiguation score (±1).

All scoring arithmetic lives in C2 (`rescore.py`) and is fully deterministic.

---

## Key Data Models

| Model | File | Purpose |
|---|---|---|
| `HpoTerm` | `models/phenopacket.py` | A single HPO term (observed or excluded) |
| `PatientPhenotype` | `models/phenopacket.py` | Full patient description |
| `ExomiserCandidate` | `models/exomiser.py` | One gene/disease candidate from Exomiser |
| `Variant` | `models/exomiser.py` | A contributing variant for a candidate |
| `DiseaseFeature` | `models/candidate.py` | One feature from a disease profile |
| `FeatureMatch` | `models/candidate.py` | A patient term matched to a disease feature |
| `FitEvidence` | `models/candidate.py` | All match buckets for one candidate (C1 output) |
| `ReRankedCandidate` | `models/report.py` | A candidate with new rank + fit + mechanism |
| `RankedReport` | `models/report.py` | The final output for one patient case |
| `DiseaseProfile` | `knowledge/hpoa/tool.py` | Curated phenotype profile for a disease |
| `MechanismVerdict` | `models/report.py` | C5 output: congruent/incongruent/uncertain |
| `DisambiguationReport` | `models/report.py` | C4 output: which disease fits best |

---

## Files Reading Order

Read the files in this order to build up understanding from the ground up:

1. **`models/phenopacket.py`** — Start here. This is the shape of the patient input: HPO
   terms (observed vs explicitly excluded), sex, and age. The absence/excluded distinction
   is critical — a term not mentioned is "not assessed", not "absent".

2. **`models/exomiser.py`** — The shape of what Exomiser produces: one candidate per
   gene/disease/MOI combination, with variant evidence and scores.

3. **`models/candidate.py`** — The core intermediate data structure. `FitEvidence` is what
   C1 produces and what C2 scores. `DiseaseFeature` is one annotated HPO feature from a
   disease profile.

4. **`models/report.py`** — The output structures: `ReRankedCandidate`, `RankedReport`,
   `DisambiguationReport`, `MechanismVerdict`.

5. **`ontology/frequency.py`** — How disease feature frequency classes (`Obligate`,
   `Frequent`, etc.) map to numeric weights. Obligate features matter much more than
   occasional ones.

6. **`ontology/hpo.py`** — The HPO ontology graph. Supports ancestor/descendant lookups
   (used in C1 for subsumed/partial matching) and computes Resnik information content
   per term (IC — how specific/rare a term is across all diseases).

7. **`ingest/phenopacket_loader.py`** — Parses a GA4GH phenopacket JSON file into a
   `PatientPhenotype`. Also handles Family phenopackets (extracts the proband).

8. **`ingest/exomiser_loader.py`** — Reads the Exomiser parquet file. Aggregates
   gene-level scores and contributing variants, then picks the right disease per candidate
   based on mode-of-inheritance matching.

9. **`knowledge/hpoa/tool.py`** — Loads the HPO Annotation database (`phenotype.hpoa`)
   into memory. For any disease ID (OMIM or ORPHA), returns the list of annotated HPO
   features with frequency and IC enrichment.

10. **`knowledge/orphanet/tool.py`** — Supplements HPOA with Orphanet disease profiles
    from XML files. Especially useful for ORPHA-prefixed disease IDs. The pipeline merges
    HPOA (primary) and Orphanet (supplementary), deduplicating by HPO ID.

11. **`reasoning/llm.py`** — The LLM abstraction layer. Defines `LLMClient` (abstract),
    `MockLLMClient` (deterministic — for testing and runs without a model), and
    `OllamaLLMClient` (real model via a local Ollama server). All LLM calls go through
    the same `resolve_match` / `synthesise_narrative` interface.

12. **`components/partition.py`** — C1. Reads patient terms and a disease profile,
    produces `FitEvidence`. The matching priority is: exact → subsumed → LLM partial.
    Only the "broader term" case (patient has an ancestor of the disease feature) calls
    the LLM.

13. **`components/age_gate.py`** — C3. Simple post-processing on `FitEvidence`: moves
    `expected_absent` features to `age_excused` when the disease onset is later than the
    patient's current age.

14. **`components/rescore.py`** — C2. The fit score formula and re-ranking logic. This is
    the mathematical core of PRISM. No LLM involvement — pure arithmetic on `FitEvidence`.

15. **`components/mechanism.py`** — C5. Checks variant pathogenicity class and
    mode-of-inheritance consistency. Fully deterministic, no LLM.

16. **`components/disambiguate.py`** — C4. The most complex component. Finds overlapping
    candidate pairs, computes discriminating features, scores them against the patient,
    and uses the LLM for features the patient is silent on.

17. **`pipeline.py`** — The top-level orchestrator for a single case. Loads knowledge
    bases, runs C1 → C3 → C2 → C5 → C4, decides on callable diagnoses, generates
    narratives. This is the best file to read to understand how the components connect.

18. **`reasoning/agent.py`** — A stateful wrapper around `pipeline.py`. Loads knowledge
    bases once at construction and reuses them across many cases. Use this for batch
    evaluation (e.g. PhEval benchmarks) to avoid reloading HPO and HPOA thousands of times.

19. **`cli.py`** — The command-line interface. Three subcommands: `run` (single case),
    `batch` (directory of cases with `--skip-existing` for retrying failures), and
    `pheval-gene-result` (export to PhEval benchmark format).

20. **`pheval_export.py`** — Converts PRISM JSON report files into PhEval gene result
    parquet files for standardised benchmarking.
