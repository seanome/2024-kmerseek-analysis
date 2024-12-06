import polars as pl
import pytest


def get_intersecting_scop_fixed(
    scop_id_to_lineage_original: pl.Series,
    scop_id_to_lineage_fixed: pl.Series,
    verbose: bool = False,
):
    scop_df = pl.DataFrame(
        {
            "scop_original": scop_id_to_lineage_original,
            "scop_fixed": scop_id_to_lineage_fixed,
        }
    )

    scop_df = scop_df.with_column(
        pl.when(pl.col("scop_fixed").is_null())
        .then(pl.col("scop_original"))
        .otherwise(pl.col("scop_fixed"))
        .alias("scop_merged")
    )

    if verbose:
        print(f"scop_df.shape: {scop_df.shape}")
        print(scop_df.head())

    if verbose:
        print("--- Different SCOP lineages ---")
        different = scop_df.filter(
            (pl.col("scop_fixed") != pl.col("scop_original"))
            & pl.col("scop_fixed").is_not_null()
        )
        print(f"different.shape: {different.shape}")
        print(different)

    # Simple assertion statement for an ID that is not in scop_fixed but is in the original
    assert "d4j42a_" in scop_df["scop_original"].to_list()

    scop_merged = scop_df.select("scop_merged").to_series()
    scop_merged = scop_merged.drop_duplicates()

    return scop_merged


def subset_scop_name(
    name_series: pl.Series, index: int, new_colname: str, verbose: bool
) -> pl.Series:
    subset = name_series.str.split(by=" ").list.get(index).alias(new_colname)
    if verbose:
        print(f"--- {new_colname} ---")
        print(subset.value_counts())

    return subset


def make_scop_metadata(name_series: pl.Series, verbose: bool = False) -> pl.DataFrame:
    scop_id = subset_scop_name(name_series, 0, "scop_id", verbose)
    scop_lineage = subset_scop_name(name_series, 1, "scop_lineage", verbose)
    scop_metadata = pl.DataFrame(
        {"name": name_series, "scop_id": scop_id, "scop_lineage": scop_lineage}
    )

    return scop_metadata


def extract_scop_info_from_name(
    name_series: pl.Series,
    scop_fixed: pl.Series,
    name_prefix: str,
    verbose: bool = False,
) -> pl.DataFrame:

    name_series_no_dups = name_series.unique()
    scop_metadata = make_scop_metadata(name_series_no_dups, verbose)

    scop_lineage_fixed = get_intersecting_scop_fixed(
        scop_metadata["scop_lineage"], scop_fixed, verbose=verbose
    )

    scop_metadata = scop_metadata.with_column(
        pl.col("scop_id")
        .map_dict(dict(zip(scop_lineage_fixed["scop_id"], scop_lineage_fixed)))
        .alias("scop_lineage_fixed")
    )

    lineages_extracted = extract_scop_lineages(scop_metadata["scop_lineage_fixed"])
    scop_metadata_with_extracted_lineages = scop_metadata.hstack(lineages_extracted)

    scop_metadata_with_extracted_lineages = (
        scop_metadata_with_extracted_lineages.select(pl.all().prefix(f"{name_prefix}_"))
    )
    scop_metadata_with_extracted_lineages = (
        scop_metadata_with_extracted_lineages.set_index(f"{name_prefix}_name")
    )

    return scop_metadata_with_extracted_lineages


def extract_scop_lineages(scop_lineage_series: pl.Series) -> pl.DataFrame:
    pattern = r"(?P<family>(?P<superfamily>(?P<fold>(?P<class>[a-z])\.\d+)\.\d+)\.\d+)"
    lineages_extracted = scop_lineage_series.str.extract(pattern)
    return lineages_extracted


@pytest.fixture
def test_name_series():
    return pl.read_csv("scop_utils_test_name_series.csv").to_series()


@pytest.fixture
def test_scop_fixed():
    return pl.read_csv("scop_utils_test_scop_fixed.csv", has_header=False).to_series()


@pytest.fixture
def true_scop_metadata():
    df = pl.read_csv("scop_utils_true_scop_metadata.csv")
    # Move the 'query_name' column to the front to act as an index
    return df.select(pl.col("query_name"), pl.all().exclude("query_name"))


def test_extract_scop_info_from_name(
    test_name_series, test_scop_fixed, true_scop_metadata
):
    test_scop_metadata = extract_scop_info_from_name(
        test_name_series, test_scop_fixed, "query"
    )

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

    # Test for overall equality
    assert test_scop_metadata.frame_equal(true_scop_metadata)
