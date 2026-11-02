# Data versions

Each subdirectory here is a named data version, selectable with `--data-version <name>`
on `prism run` / `prism batch` / `prism build-db`.

A version directory only needs to contain the files it changes — anything it
doesn't provide falls back to the base files in `data/hpo/`, `data/hpoa/`,
and `data/orphanet/`. Recognised filenames:

- `hp.obo`
- `phenotype.hpoa`
- `en_product4.xml`   (Orphanet disease-HPO associations)
- `en_product9_ages.xml`  (Orphanet onset data)
- `en_product1.xml`   (Orphanet OMIM-ORPHA cross-references)

Example — a version that only overrides the HPOA annotations (e.g. the
literature-augmented file produced by `hpoa_supplement_merge.py`):

```
data/versions/augmented/
  phenotype.hpoa
```

Then run with:

```
prism batch cases/ --output-dir out/ --data-version augmented
```
## Bundled versions

- `2512` — HPO / HPOA / Orphanet from the 2025-11-24 HPO release.
- `2602` — `hp.obo` and `phenotype.hpoa` from the 2026-01-08 HPO release. Orphanet
  falls back to the base files: `en_product4.xml` and `en_product9_ages.xml` from
  2025-12-09, `en_product1.xml` (OMIM↔ORPHA xrefs) from 2026-06-23.
- `2602_cardinality` — the `2602` files (2026-01-08 `hp.obo` and `phenotype.hpoa`, same
  rows, no additions) with the `cardinality` column copied from `cardinality`, matched on
  database_id + hpo_id + qualifier + aspect. 510 rows (421 phenotype) have no match in the
  2026-06-23 release and are left empty, which C2 treats as neutral (×1.0). Orphanet as `2602`.
- `cardinality` — `phenotype.hpoa` from the 2026-06-23 HPOA release with an extra
  `cardinality` column (CARDINAL / SUPPORTIVE / NON_CARDINAL), LLM-assigned per
  disease–phenotype annotation. Uses the base `hp.obo` (2026-06-06), which contains
  every HPO term the file references. Use with `--cardinality match|miss|both`:

  ```
  prism batch cases/ --output-dir out/ --data-version cardinality --cardinality both
  ```

  For a like-for-like baseline, run `--data-version cardinality --cardinality off`.
