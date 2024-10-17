from typing import Literal, get_args
import pandas as pd
from IPython.display import display
import pytest


from scop_constants import SCOP_LINEAGES, FOLDSEEK_SCOP_FIXED
from sourmash_constants import MOLTYPES


class MultisearchParser:

    def __init__(self, pipeline_outdir, moltype, ksize, analysis_outdir, verbose=False):
        self.pipeline_outdir = pipeline_outdir
        self.moltype = moltype
        self.ksize = ksize
        self.analysis_outdir = analysis_outdir
        self.verbose = self.verbose

    def read_multisearch_csv(self):
        if self.verbose:
            print(f"\n\n--- moltype: {self.moltype}, ksize: {self.ksize} --")
        csv = make_multisearch_csv(self.pipeline_outdir, self.moltype, self.ksize)
        if self.verbose:
            print(f"\nReading {csv} ...")
        multisearch = pd.read_csv(csv)
        if self.verbose:
            print("\tDone")
        return multisearch

    def process_multisearch_scop_results(self):

        self.multisearch = self.read_multisearch_csv()

        query_metadata = extract_scop_info_from_name(
            multisearch.query_name, FOLDSEEK_SCOP_FIXED, "query", verbose=False
        )

        match_metadata = extract_scop_info_from_name(
            multisearch.match_name, FOLDSEEK_SCOP_FIXED, "match", verbose=False
        )

        multisearch_metadata = multisearch.join(query_metadata, on="query_name").join(
            match_metadata, on="match_name"
        )

        for lineage_col in get_args(SCOP_LINEAGES):
            query = f"query_{lineage_col}"
            match = f"match_{lineage_col}"
            same = f"same_{lineage_col}"

            multisearch_metadata[same] = (
                multisearch_metadata[query] == multisearch_metadata[match]
            )

            add_categories(multisearch_metadata, query, match)

        write_parquet(
            multisearch_metadata,
            analysis_outdir,
            ksize,
            moltype,
            filtered=False,
            verbose=verbose,
        )

        # Remove self matches and likely spurious matches
        multisearch_metadata_filtered = multisearch_metadata.query(
            "query_md5 != match_md5 and intersect_hashes > 1"
        )

        write_parquet(
            multisearch_metadata_filtered,
            analysis_outdir,
            ksize,
            moltype,
            filtered=True,
            verbose=verbose,
        )
        return multisearch_metadata_filtered


def get_intersecting_scop_fixed(
    scop_id_to_lineage_original: pd.Series,
    scop_id_to_lineage_fixed: pd.Series,
    verbose: str = False,
):
    scop_id_to_lineage_original.name = "scop_original"
    scop_id_to_lineage_fixed.name = "scop_fixed"

    scop_df = scop_id_to_lineage_original.to_frame().join(scop_id_to_lineage_fixed)

    scop_df["scop_merged"] = (
        scop_df["scop_fixed"].copy().fillna(scop_df["scop_original"])
    )

    if verbose:
        print(f"scop_df.shape: {scop_df.shape}")
        display(scop_df.head())

    if verbose:
        print("--- Different SCOP lineages ---")
        different = scop_df.query("scop_fixed != scop_original").dropna(
            subset="scop_fixed"
        )
        print(f"different.shape: {different.shape}")
        display(different)

    scop_merged = scop_df["scop_merged"]
    scop_merged = scop_merged[~scop_merged.index.duplicated()]

    return scop_merged


def subset_scop_name(
    name_series: pd.Series, index: int, new_colname: str, verbose: bool
) -> pd.Series:
    subset = name_series.str.split().str[index]
    subset.name = new_colname
    if verbose:
        print(f"--- {new_colname} ---")
        print(subset.value_counts())

    subset.index = name_series.values
    return subset


def make_scop_metadata(name_series: pd.Series, verbose: bool = False) -> pd.DataFrame:
    """Extract SCOP metadata (scop ID and lineage) from a full query name

    Args:
        name_series (pd.Series): Pandas string series of query name
            e.g. 'd4j42a_ a.25.3.0 (A:) automated matches {Bacillus anthracis [TaxId: 260799]}'
            d4j42a_ -> "scop_id"
            a.25.3.0 -> "lineage"
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: 2-column dataframe with "scop_id" as the index and
            the columns of name (from name_series), , and "scop_lineage"
    """
    scop_id = subset_scop_name(name_series, 0, "scop_id", verbose)
    scop_lineage = subset_scop_name(name_series, 1, "scop_lineage", verbose)
    scop_metadata = scop_id.to_frame()
    scop_metadata = scop_metadata.join(scop_lineage)
    scop_metadata.index.name = "name"
    scop_metadata = scop_metadata.reset_index()
    scop_metadata = scop_metadata.set_index("scop_id")

    return scop_metadata


def extract_scop_info_from_name(
    name_series: pd.Series,
    scop_fixed: pd.Series,
    name_prefix: str,
    verbose: bool = False,
) -> pd.DataFrame:

    name_series_no_dups = name_series.drop_duplicates()
    scop_metadata = make_scop_metadata(name_series_no_dups, verbose)

    scop_lineage_fixed = get_intersecting_scop_fixed(
        scop_metadata["scop_lineage"], scop_fixed, verbose=verbose
    )

    # Use the fixed lineage first
    scop_metadata["scop_lineage_fixed"] = scop_metadata.index.map(scop_lineage_fixed)

    lineages_extracted = extract_scop_lineages(scop_metadata["scop_lineage_fixed"])
    lineages_extracted.index = name_series_no_dups

    scop_metadata_with_extracted_lineages = scop_metadata.join(
        lineages_extracted, on="name"
    )
    scop_metadata_with_extracted_lineages = (
        scop_metadata_with_extracted_lineages.reset_index()
    )
    scop_metadata_with_extracted_lineages.columns = (
        f"{name_prefix}_" + scop_metadata_with_extracted_lineages.columns
    )

    scop_metadata_with_extracted_lineages = (
        scop_metadata_with_extracted_lineages.set_index(f"{name_prefix}_name")
    )
    return scop_metadata_with_extracted_lineages


def extract_scop_lineages(scop_lineage_series: pd.Series) -> pd.DataFrame:
    """Use regex to extract SCOP lineage information"""
    # Pattern from: https://regex101.com/r/gKBJAr/1
    pattern = r"(?P<family>(?P<superfamily>(?P<fold>(?P<class>[a-z])\.\d+)\.\d+)\.\d+)"
    lineages_extracted = scop_lineage_series.str.extractall(pattern)
    lineages_extracted = lineages_extracted.droplevel(-1)
    return lineages_extracted


def make_multisearch_csv(
    outdir,
    moltype,
    ksize,
    query="astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.part_001.fa",
    against="astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.part_001.fa",
):
    basename = f"{query}--in--{against}.{moltype}.{ksize}.multisearch.csv"
    csv = f"{outdir}/sourmash/multisearch/{basename}"
    return csv


def make_output_pq(analysis_outdir: str, ksize: int, moltype: MOLTYPES, filtered: bool):
    pq = f"{analysis_outdir}/00_cleaned_multisearch_results/scope40.multisearch.{moltype}.k{ksize}"
    if filtered:
        pq += ".filtered.pq"
    else:
        pq += ".pq"

    return pq


def write_parquet(
    df: pd.DataFrame,
    analysis_outdir: str,
    ksize: int,
    moltype: MOLTYPES,
    filtered: bool = False,
    verbose: bool = False,
):
    pq = make_output_pq(analysis_outdir, ksize, moltype, filtered)
    if verbose:
        print(f"\nWriting {len(df)} rows and {len(df.columns)} columns to {pq} ...")
    df.to_parquet(pq)
    if verbose:
        print(f"\tDone.")


def add_categories(df: pd.DataFrame, query: str, match: str):
    """Set query and match SCOP lineages as categorical, so we always
    have all options for computing classification scores

    Args:
        df (_type_): _description_
        query (_type_): column in df
        match (_type_): column in df
    """

    # Get all possible categories using the query, which is all possible
    categories = sorted(list(set(df[query])))
    df[query] = pd.Categorical(df[query], categories=categories, ordered=True)
    df[match] = pd.Categorical(df[match], categories=categories, ordered=True)


def process_multisearch_scop_results(
    pipeline_outdir, moltype, ksize, analysis_outdir, verbose=False
):
    if verbose:
        print(f"\n\n--- ksize: {ksize} --")
    csv = make_multisearch_csv(pipeline_outdir, moltype, ksize)
    if verbose:
        print(f"\nReading {csv} ...")
    multisearch = pd.read_csv(csv)
    if verbose:
        print("\tDone")

    query_metadata = extract_scop_info_from_name(
        multisearch.query_name, FOLDSEEK_SCOP_FIXED, "query", verbose=False
    )

    match_metadata = extract_scop_info_from_name(
        multisearch.match_name, FOLDSEEK_SCOP_FIXED, "match", verbose=False
    )

    multisearch_metadata = multisearch.join(query_metadata, on="query_name").join(
        match_metadata, on="match_name"
    )

    for lineage_col in get_args(SCOP_LINEAGES):
        query = f"query_{lineage_col}"
        match = f"match_{lineage_col}"
        same = f"same_{lineage_col}"

        multisearch_metadata[same] = (
            multisearch_metadata[query] == multisearch_metadata[match]
        )

        add_categories(multisearch_metadata, query, match)

    write_parquet(
        multisearch_metadata,
        analysis_outdir,
        ksize,
        moltype,
        filtered=False,
        verbose=verbose,
    )

    # Remove self matches and likely spurious matches
    multisearch_metadata_filtered = multisearch_metadata.query(
        "query_md5 != match_md5 and intersect_hashes > 1"
    )

    write_parquet(
        multisearch_metadata_filtered,
        analysis_outdir,
        ksize,
        moltype,
        filtered=True,
        verbose=verbose,
    )
    return multisearch_metadata_filtered


@pytest.fixture
def test_name_series():
    return pd.read_csv("scop_utils_test_name_series.csv").squeeze()


@pytest.fixture
def test_scop_fixed():
    return pd.read_csv(
        "scop_utils_test_scop_fixed.csv", header=None, index_col=0
    ).squeeze()


@pytest.fixture
def true_scop_metadata():
    return pd.read_csv("scop_utils_true_scop_metadata.csv", index_col=0)


def test_extract_scop_info_from_name(
    test_name_series, test_scop_fixed, true_scop_metadata
):
    test_scop_metadata = extract_scop_info_from_name(
        test_name_series, test_scop_fixed, "query"
    )

    # lineage_cols = ["query_family", "query_superfamily", "query_fold", "query_class"]
    assert test_scop_metadata["query_family"].value_counts().head().to_dict() == {
        "a.4.5.0": 75,
        "a.104.1.0": 61,
        "a.45.1.0": 44,
        "a.121.1.1": 34,
        "a.4.1.9": 33,
    }
    assert test_scop_metadata["query_superfamily"].value_counts().head().to_dict() == {
        "a.4.5": 252,
        "a.4.1": 113,
        "a.39.1": 73,
        "a.104.1": 73,
        "a.45.1": 63,
    }
    assert test_scop_metadata["query_fold"].value_counts().head().to_dict() == {
        "a.4": 425,
        "a.118": 179,
        "a.60": 91,
        "a.39": 89,
        "a.24": 82,
    }
    # All the test data are class "a" -> don't need to test "query_class"

    # Test for overall equality
    pd.testing.assert_frame_equal(test_scop_metadata, true_scop_metadata)
