from __future__ import annotations


class TestHpoaRetriever:
    def test_gipc3_profile(self, small_retriever, small_graph, small_ic_map):
        profile = small_retriever.get_disease_profile(
            "OMIM:601869", graph=small_graph, ic_map=small_ic_map
        )
        assert profile.disease_id == "OMIM:601869"
        assert len(profile.features) == 2
        assert len(profile.excluded_features) == 0
        ids = {f.hpo_id for f in profile.features}
        assert "HP:0000407" in ids
        assert "HP:0000399" in ids

    def test_frequency_classes(self, small_retriever):
        profile = small_retriever.get_disease_profile("OMIM:601869")
        freq_map = {f.hpo_id: f.frequency_class for f in profile.features}
        assert freq_map["HP:0000407"] == "Obligate"
        assert freq_map["HP:0000399"] == "VeryFrequent"

    def test_pfeiffer_excluded_feature(self, small_retriever):
        profile = small_retriever.get_disease_profile("OMIM:101600")
        excl_ids = {f.hpo_id for f in profile.excluded_features}
        assert "HP:0001161" in excl_ids  # Camptodactyly NOT expected in Pfeiffer

    def test_pfeiffer_feature_count(self, small_retriever):
        profile = small_retriever.get_disease_profile("OMIM:101600")
        assert len(profile.features) == 4
        assert len(profile.excluded_features) == 1

    def test_labels_resolved_from_graph(self, small_retriever, small_graph):
        profile = small_retriever.get_disease_profile(
            "OMIM:601869", graph=small_graph
        )
        hp407 = next(f for f in profile.features if f.hpo_id == "HP:0000407")
        assert hp407.hpo_label == "Sensorineural hearing impairment"

    def test_ic_populated_when_ic_map_provided(self, small_retriever, small_graph, small_ic_map):
        profile = small_retriever.get_disease_profile(
            "OMIM:601869", graph=small_graph, ic_map=small_ic_map
        )
        for f in profile.features:
            assert f.ic > 0.0

    def test_disease_to_terms(self, small_retriever):
        d2t = small_retriever.disease_to_terms()
        assert "OMIM:601869" in d2t
        assert "HP:0000407" in d2t["OMIM:601869"]
        # NOT qualifier should not appear in disease_to_terms
        assert "HP:0001161" not in d2t.get("OMIM:101600", set())

    def test_unknown_disease_returns_empty_profile(self, small_retriever):
        profile = small_retriever.get_disease_profile("OMIM:999999")
        assert profile.features == []
        assert profile.excluded_features == []