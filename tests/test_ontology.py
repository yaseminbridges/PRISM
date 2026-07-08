from __future__ import annotations
import pytest
from prism.ontology.frequency import (
    frequency_class_weight,
    parse_hpoa_frequency,
    FALLBACK_WEIGHT,
)


class TestHPOGraph:
    def test_ancestors_include_chain(self, small_graph):
        # HP:0000399 -> HP:0000407 -> HP:0000598 -> HP:0000118 -> HP:0000001
        ancs = small_graph.ancestors("HP:0000399")
        assert "HP:0000407" in ancs
        assert "HP:0000598" in ancs
        assert "HP:0000118" in ancs
        assert "HP:0000001" in ancs

    def test_ancestors_exclude_self(self, small_graph):
        assert "HP:0000399" not in small_graph.ancestors("HP:0000399")

    def test_ancestors_exclude_descendants(self, small_graph):
        ancs = small_graph.ancestors("HP:0000407")
        assert "HP:0000399" not in ancs

    def test_descendants(self, small_graph):
        desc = small_graph.descendants("HP:0000407")
        assert "HP:0000399" in desc
        assert "HP:0000407" not in desc

    def test_subsumes_ancestor(self, small_graph):
        assert small_graph.subsumes("HP:0000407", "HP:0000399")

    def test_subsumes_self(self, small_graph):
        assert small_graph.subsumes("HP:0000407", "HP:0000407")

    def test_subsumes_false_for_sibling(self, small_graph):
        # HP:0001363 (Craniosynostosis) does not subsume HP:0000407 (Hearing)
        assert not small_graph.subsumes("HP:0001363", "HP:0000407")

    def test_is_onset_term_true(self, small_graph):
        assert small_graph.is_onset_term("HP:0003577")  # Congenital onset

    def test_is_onset_term_false(self, small_graph):
        assert not small_graph.is_onset_term("HP:0001363")

    def test_name_lookup(self, small_graph):
        assert small_graph.name("HP:0000399") == "Prelingual sensorineural hearing impairment"

    def test_contains(self, small_graph):
        assert "HP:0001363" in small_graph
        assert "HP:9999999" not in small_graph


class TestIC:
    def test_root_has_zero_ic(self, small_graph, small_retriever):
        d2t = small_retriever.disease_to_terms()
        ic = small_graph.compute_ic(d2t)
        # HP:0000001 is ancestor of everything -> freq=1.0 -> IC=0
        assert ic.get("HP:0000001", 0.0) == pytest.approx(0.0)

    def test_specific_term_has_higher_ic_than_general(self, small_graph, small_retriever):
        d2t = small_retriever.disease_to_terms()
        ic = small_graph.compute_ic(d2t)
        # HP:0000399 (Prelingual SHL, 1 disease) > HP:0000407 (SHL, 1 disease) since both in GIPC3 only
        # HP:0001363 (Craniosynostosis, 3 diseases) < HP:0001161 (Camptodactyly, 1 disease = Apert only)
        assert ic.get("HP:0001161", 0.0) > ic.get("HP:0001363", 0.0)

    def test_empty_corpus_returns_empty(self, small_graph):
        assert small_graph.compute_ic({}) == {}


class TestFrequency:
    def test_weights_per_spec(self):
        assert frequency_class_weight("Obligate") == 1.0
        assert frequency_class_weight("VeryFrequent") == 0.9
        assert frequency_class_weight("Frequent") == 0.5
        assert frequency_class_weight("Occasional") == 0.2
        assert frequency_class_weight("VeryRare") == 0.05
        assert frequency_class_weight("Excluded") == 0.0

    def test_none_returns_fallback(self):
        assert frequency_class_weight(None) == FALLBACK_WEIGHT

    def test_parse_hpo_term(self):
        cls, w = parse_hpoa_frequency("HP:0040280")
        assert cls == "Obligate"
        assert w == 1.0

    def test_parse_ratio(self):
        cls, w = parse_hpoa_frequency("5/30")  # 16.7% -> Occasional
        assert cls == "Occasional"

    def test_parse_percentage(self):
        cls, w = parse_hpoa_frequency("85%")  # VeryFrequent
        assert cls == "VeryFrequent"

    def test_parse_empty(self):
        cls, w = parse_hpoa_frequency("")
        assert cls is None
        assert w == FALLBACK_WEIGHT