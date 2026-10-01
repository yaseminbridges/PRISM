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
- `cardinality` — `phenotype.hpoa` from the 2026-06-23 HPOA release with an extra
  `cardinality` column (CARDINAL / SUPPORTIVE / NON_CARDINAL), LLM-assigned per
  disease–phenotype annotation. Uses the base `hp.obo` (2026-06-06), which contains
  every HPO term the file references. Use with `--cardinality match|miss|both`:

  ```
  prism batch cases/ --output-dir out/ --data-version cardinality --cardinality both
  ```

  For a like-for-like baseline, run `--data-version cardinality --cardinality off`.
