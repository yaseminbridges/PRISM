from __future__ import annotations
from pathlib import Path
import pytest

from prism.ontology.hpo import HPOGraph
from prism.knowledge.hpoa.tool import HpoaRetriever

FIXTURES = Path(__file__).parent / "fixtures"
PHENOPACKETS = FIXTURES / "phenopackets"
EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="session")
def small_graph() -> HPOGraph:
    return HPOGraph.from_obo(FIXTURES / "hp_small.obo")


@pytest.fixture(scope="session")
def small_retriever() -> HpoaRetriever:
    return HpoaRetriever(FIXTURES / "phenotype_small.hpoa")


@pytest.fixture(scope="session")
def small_ic_map(small_graph: HPOGraph, small_retriever: HpoaRetriever) -> dict[str, float]:
    return small_graph.compute_ic(small_retriever.disease_to_terms())


@pytest.fixture
def gipc3_phenopacket() -> Path:
    return PHENOPACKETS / "gipc3_case.json"


@pytest.fixture
def pfeiffer_phenopacket() -> Path:
    return PHENOPACKETS / "pfeiffer_case.json"


@pytest.fixture
def crouzon_phenopacket() -> Path:
    return PHENOPACKETS / "crouzon_case.json"


@pytest.fixture
def real_exomiser_parquet() -> Path:
    return EXAMPLES / "Asgharzade-2018-GIPC3-Ahv-14_23-exomiser.parquet"