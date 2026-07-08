# Orphanet Reasoning Skill

## Role
Resolve ambiguous phenotype matches using Orphanet clinical descriptions and
HPO annotations as grounding context.

## Inputs
- `patient_term`: HPO term observed in the patient (id, label)
- `disease_feature`: HPO term from the Orphanet disease profile (id, label, frequency_class)
- `context`: Sentence(s) from Orphanet describing this feature or the disease

## Output (structured JSON)
```json
{
  "label": "matched | partial | absent_expected | unexplained | contradiction",
  "provenance": "verbatim sentence from context that grounds this judgement"
}
```

## Rules
1. Only use information present in `context`. Do not invent features.
2. `matched` — patient term is the same concept or a child of the disease feature,
   or the context confirms they refer to the same clinical finding.
3. `partial` — concepts overlap but are not identical; context supports a plausible
   but uncertain link. Always flag with provenance.
4. `contradiction` — Orphanet lists this feature as Excluded (0%) yet the patient
   has it. Quote the Orphanet exclusion from context.
5. If context is insufficient to resolve, return `"label": "partial"` and note
   "insufficient context" in provenance — never guess.
6. Age-of-onset from Orphanet is evidence, not a verdict. A feature absent in a
   patient younger than typical onset is not a contradiction — flag for C3.