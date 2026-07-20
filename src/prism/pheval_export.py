"""Convert PRISM re-ranked JSON reports into PhEval gene result parquet files."""
import sys
from pathlib import Path

import polars as pl
from pheval.post_processing.post_processing import SortOrder, generate_gene_result
from pheval.utils.file_utils import files_with_suffix
from pheval.utils.phenopacket_utils import GeneIdentifierUpdater, create_gene_identifier_map

IDENTIFIER_TYPES = ["ensembl_id", "entrez_id", "hgnc_id", "refseq_accession"]


def export_pheval_gene_results(
    results_dir: Path,
    output_dir: Path,
    phenopacket_dir: Path,
    suffix: str = ".json",
    identifier_type: str = "ensembl_id",
    sort_order: SortOrder = SortOrder.DESCENDING,
) -> None:
    """
    Convert PRISM re-ranked JSON reports in ``results_dir`` into PhEval gene
    result parquet files written under ``output_dir/pheval_gene_results/``.

    Args:
        results_dir (Path): Directory containing PRISM report JSON files (each with a "reranked" field).
        output_dir (Path): PhEval output directory.
        phenopacket_dir (Path): Directory containing the corresponding GA4GH phenopackets.
        suffix (str): File suffix used to find report files in ``results_dir``.
        identifier_type (str): Gene identifier type to resolve, e.g. "ensembl_id", "entrez_id", "hgnc_id", "refseq_accession".
        sort_order (SortOrder): Whether higher or lower scores rank first.
    """
    identifier_map = create_gene_identifier_map()
    gene_identifier_updater = GeneIdentifierUpdater(identifier_type, identifier_map)

    output_dir.joinpath("pheval_gene_results").mkdir(parents=True, exist_ok=True)

    result_files = files_with_suffix(results_dir, suffix)
    if not result_files:
        print(f"No {suffix} files found in {results_dir}", file=sys.stderr)
        return

    for result_file in result_files:
        print(f"  [convert]  {result_file.name}", file=sys.stderr)
        df = pl.read_json(result_file)
        results = []
        for item in df["reranked"][0]:
            gene_symbol = item["candidate"]["gene_symbol"]
            score = item["new_rank"]
            gene_identifier = gene_identifier_updater.find_identifier(gene_symbol)
            results.append(
                {
                    "score": 1 / score,
                    "gene_symbol": gene_symbol,
                    "gene_identifier": gene_identifier,
                }
            )
        if not results:
            completed_df = pl.DataFrame(
                {"score": [], "gene_symbol": [], "gene_identifier": []},
                schema={"score": pl.Float64, "gene_symbol": pl.Utf8, "gene_identifier": pl.Utf8},
            )
        else:
            completed_df = pl.DataFrame(results)
        generate_gene_result(completed_df, sort_order, output_dir, result_file, phenopacket_dir)