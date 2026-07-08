"""Paths to bundled data files shipped with the package.

Use these instead of hard-coding 'data/...' relative to the working directory.
They resolve correctly whether the package is installed via pip or run from source.
"""
from pathlib import Path

_DATA = Path(__file__).parent / "data"

HPO_OBO        = _DATA / "hpo"  / "hp.obo"
HPOA           = _DATA / "hpoa" / "phenotype.hpoa"
ORPHANET_P4    = _DATA / "orphanet" / "en_product4.xml"
ORPHANET_AGES  = _DATA / "orphanet" / "en_product9_ages.xml"
ORPHANET_XREF  = _DATA / "orphanet" / "en_product1.xml"
