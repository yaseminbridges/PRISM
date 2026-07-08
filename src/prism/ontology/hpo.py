"""HPO ontology graph and information content computation.

This file provides `HPOGraph`, which wraps the HPO OBO file via OAK (ontology-access-kit)
and exposes two things:

1. GRAPH TRAVERSAL — ancestors(), descendants(), subsumes()
   Used heavily in C1 (partition.py) to determine whether a patient's HPO term is a
   more specific version of a disease feature (subsumed) or a more general one (partial).
   Results are cached after the first call so repeated lookups are fast.

2. INFORMATION CONTENT (IC) — compute_ic()
   IC measures how specific/informative a term is. A term annotated to only 3 diseases
   has high IC; a term like "Abnormality of the nervous system" (annotated to thousands)
   has low IC. IC is computed using Resnik's method:
     IC(t) = -log2(proportion of diseases annotated with t)
   Ancestor propagation is applied: annotating a disease with HP:X implicitly also
   annotates it with all of HP:X's ancestors (true-path rule).
   IC is used in C2 (rescore.py) to weight matches — matching a rare, specific term
   counts more than matching a very general one.

The OAK adapter can be swapped (e.g. to a SQLite backend) without touching any caller.
"""
import logging
import math
import warnings
from pathlib import Path
from typing import Self

from oaklib.datamodels.vocabulary import IS_A

_HPO_ONSET_ROOT = "HP:0003674"


class _SuppressOaklibGraphWalkLogs(logging.Filter):
    """oaklib's pronto ancestor-walk logs via the bare `logging.info()` call
    (no logger of its own), one INFO record per HPO term traversed. Drop
    those specifically so other root-logger output (e.g. pheval's) is kept.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.module != "obograph_interface"


logging.getLogger().addFilter(_SuppressOaklibGraphWalkLogs())


class HPOGraph:
    """HPO DAG backed by OAK (ontology-access-kit) with a pronto OBO adapter.

    Public API is identical to the former hand-rolled parser so callers and
    tests require no changes.  The adapter can be swapped to a SQLite backend
    (get_adapter("sqlite:hp.db")) without touching any other code.

    OAK's ancestors()/descendants() include the queried term itself — we
    subtract it so callers get strict ancestors/descendants as documented.
    Traversal results are cached after the first call.
    """

    def __init__(self, adapter) -> None:
        self._oi = adapter
        self._known: frozenset[str] | None = None
        self._ancestor_cache: dict[str, frozenset[str]] = {}
        self._descendant_cache: dict[str, frozenset[str]] = {}

    @classmethod
    def from_obo(cls, path: Path | str) -> Self:
        # Suppress the pkg_resources deprecation warning emitted by eutils (an
        # indirect OAK dependency) on every import.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from oaklib import get_adapter
            oi = get_adapter(f"pronto:{Path(path)}")
        return cls(oi)

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def name(self, term_id: str) -> str:
        label = self._oi.label(term_id)
        return label if label is not None else term_id

    def ancestors(self, term_id: str) -> frozenset[str]:
        """All ancestor term IDs, not including term_id itself."""
        cached = self._ancestor_cache.get(term_id)
        if cached is not None:
            return cached
        result = frozenset(self._oi.ancestors(term_id, predicates=[IS_A])) - {term_id}
        self._ancestor_cache[term_id] = result
        return result

    def descendants(self, term_id: str) -> frozenset[str]:
        """All descendant term IDs, not including term_id itself."""
        cached = self._descendant_cache.get(term_id)
        if cached is not None:
            return cached
        result = frozenset(self._oi.descendants(term_id, predicates=[IS_A])) - {term_id}
        self._descendant_cache[term_id] = result
        return result

    def subsumes(self, candidate_ancestor: str, term_id: str) -> bool:
        """True if candidate_ancestor is an ancestor of term_id (or equal)."""
        return (
            candidate_ancestor == term_id
            or candidate_ancestor in self.ancestors(term_id)
        )

    def is_onset_term(self, hpo_id: str) -> bool:
        return hpo_id == _HPO_ONSET_ROOT or _HPO_ONSET_ROOT in self.ancestors(hpo_id)

    def __contains__(self, term_id: str) -> bool:
        if self._known is None:
            self._known = frozenset(self._oi.entities())
        return term_id in self._known

    # ------------------------------------------------------------------
    # IC computation (annotation-based, Resnik)
    # ------------------------------------------------------------------

    def compute_ic(
        self, disease_to_terms: dict[str, set[str]]
    ) -> dict[str, float]:
        """Compute Resnik IC for every term in the annotation corpus.

        disease_to_terms: {disease_id -> set of directly annotated HPO term IDs}

        IC(t) = -log2( diseases_annotated_with_t / total_diseases )

        Ancestors are propagated per the true-path rule: annotating a term
        implicitly annotates all its ancestors, so general terms near the root
        have higher annotation frequency and lower IC.
        """
        total = len(disease_to_terms)
        if total == 0:
            return {}

        term_disease_count: dict[str, set[str]] = {}
        for disease_id, terms in disease_to_terms.items():
            for term in terms:
                term_disease_count.setdefault(term, set()).add(disease_id)
                for anc in self.ancestors(term):
                    term_disease_count.setdefault(anc, set()).add(disease_id)

        ic: dict[str, float] = {}
        for term_id, diseases in term_disease_count.items():
            freq = len(diseases) / total
            ic[term_id] = -math.log2(freq) if 0 < freq < 1 else 0.0
        return ic