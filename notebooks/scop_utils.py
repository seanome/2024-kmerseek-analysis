import pandas as pd
from IPython.display import display
import pytest


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
