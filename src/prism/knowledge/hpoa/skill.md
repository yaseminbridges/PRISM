# HPOA Reasoning Skill

## Role
Resolve ambiguous matches between a patient HPO term and an HPOA disease feature
when the two terms are not ontologically identical (not exact, not subsumed).

## Inputs
- `patient_term`: HPO term observed in the patient (id, label)
- `disease_feature`: HPO term from the disease profile (id, label, frequency_class)
- `context`: Sentence(s) retrieved from HPOA or Orphanet describing this feature

## Output (structured JSON)
```json
{
  "label": "matched | partial | absent_expected | unexplained | contradiction",
  "provenance": "verbatim sentence from context that grounds this judgement"
}
```

## Rules
1. Only use information present in `context`. Do not invent features.
2. `matched` — patient term is the same concept or a child of the disease feature.
3. `partial` — overlapping but not subsumed; flag with provenance.
4. `contradiction` — disease explicitly excludes this feature (qualifier=NOT) yet
   the patient has it. State the exclusion sentence from context.
5. If context is insufficient to resolve, return `"label": "partial"` and note
   "insufficient context" in provenance — never guess.