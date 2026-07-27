"""Paths to bundled data files shipped with the package.

Use these instead of hard-coding 'data/...' relative to the working directory.
They resolve correctly whether the package is installed via pip or run from source.
"""
from dataclasses import dataclass
from pathlib import Path

_DATA = Path(__file__).parent / "data"

HPO_OBO        = _DATA / "hpo"  / "hp.obo"
HPOA           = _DATA / "hpoa" / "phenotype.hpoa"
ORPHANET_P4    = _DATA / "orphanet" / "en_product4.xml"
ORPHANET_AGES  = _DATA / "orphanet" / "en_product9_ages.xml"
ORPHANET_XREF  = _DATA / "orphanet" / "en_product1.xml"

# Named data versions live under data/versions/<name>/. A version directory only
# needs to contain the files it overrides (e.g. just phenotype.hpoa for an
# augmented-annotation variant) — anything it doesn't provide falls back to the
# base files above. See data/versions/README.md for how to add one.
_VERSIONS_DIR = _DATA / "versions"


@dataclass(frozen=True)
class DataVersion:
    name: str
    hpo_path: Path
    hpoa_path: Path
    orphanet_p4_path: Path
    orphanet_ages_path: Path
    orphanet_xref_path: Path


def list_data_versions() -> list[str]:
    """Names of data versions available under data/versions/."""
    if not _VERSIONS_DIR.is_dir():
        return []
    return sorted(p.name for p in _VERSIONS_DIR.iterdir() if p.is_dir())


def resolve_data_version(version: str | None) -> DataVersion:
    """Resolve a named data version to concrete file paths.

    version=None returns the base (default) files. Otherwise looks up
    data/versions/<version>/ and uses any files found there, falling back to
    the base files for anything the version directory doesn't override.
    """
    if version is None:
        return DataVersion(
            name="(base)",
            hpo_path=HPO_OBO,
            hpoa_path=HPOA,
            orphanet_p4_path=ORPHANET_P4,
            orphanet_ages_path=ORPHANET_AGES,
            orphanet_xref_path=ORPHANET_XREF,
        )

    vdir = _VERSIONS_DIR / version
    if not vdir.is_dir():
        available = ", ".join(list_data_versions()) or "(none found under data/versions/)"
        raise ValueError(f"Unknown data version {version!r}. Available versions: {available}")

    def _pick(filename: str, base: Path) -> Path:
        candidate = vdir / filename
        return candidate if candidate.exists() else base

    return DataVersion(
        name=version,
        hpo_path=_pick("hp.obo", HPO_OBO),
        hpoa_path=_pick("phenotype.hpoa", HPOA),
        orphanet_p4_path=_pick("en_product4.xml", ORPHANET_P4),
        orphanet_ages_path=_pick("en_product9_ages.xml", ORPHANET_AGES),
        orphanet_xref_path=_pick("en_product1.xml", ORPHANET_XREF),
    )
