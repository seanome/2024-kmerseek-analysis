import os
from typing import Literal, get_args, Union

import polars as pl
from IPython.display import display
import pytest


from scop_constants import SCOP_LINEAGES, FOLDSEEK_SCOP_FIXED
from sourmash_constants import MOLTYPES


class MultisearchParser:

    def __init__(
        self,
        pipeline_outdir: str,
        moltype: MOLTYPES,
        ksize: int,
        analysis_outdir: str,
        verbose: bool = False,
    ):
        self.pipeline_outdir = pipeline_outdir
        self.moltype = moltype
        self.ksize = ksize
        self.analysis_outdir = analysis_outdir
        self.verbose = verbose

    def _read_multisearch_csv(self) -> pl.DataFrame:
        if self.verbose:
            print(f"\n\n--- moltype: {self.moltype}, ksize: {self.ksize} --")
        csv = self._make_multisearch_csv()
        if self.verbose:
            print(f"\nReading {csv} ...")
        multisearch = pl.scan_csv(csv)
        if self.verbose:
            print("\tDone")
        return multisearch

    def _add_query_match_scop_metadata(self, multisearch: pl.DataFrame) -> pl.DataFrame:
        query_metadata = ScopMetadataExtractor(
            multisearch.select("query_name"),
            FOLDSEEK_SCOP_FIXED,
            "query",
            verbose=False,
        ).extract_scop_info_from_name()

        match_metadata = ScopMetadataExtractor(
            multisearch.select("match_name"),
            FOLDSEEK_SCOP_FIXED,
            "match",
            verbose=False,
        ).extract_scop_info_from_name()

        multisearch_metadata = multisearch.join(
            query_metadata, left_on="query_name", right_on="query_name"
        ).join(match_metadata, left_on="match_name", right_on="match_name")
        return multisearch_metadata

    def _add_if_query_match_lineages_are_same(
        self, multisearch_metadata: pl.DataFrame
    ) -> pl.DataFrame:
        for lineage_col in get_args(SCOP_LINEAGES):
            query = f"query_{lineage_col}"
            match = f"match_{lineage_col}"
            same = f"same_{lineage_col}"

            multisearch_metadata = multisearch_metadata.with_columns(
                (pl.col(query) == pl.col(match)).alias(same)
            )

            multisearch_metadata = (
                ScopMetadataExtractor.convert_query_match_cols_to_categories(
                    multisearch_metadata, query, match
                )
            )

        return multisearch_metadata

    def process_multisearch_scop_results(self):
        multisearch = self._read_multisearch_csv()
        multisearch_metadata = self._add_query_match_scop_metadata(multisearch)
        multisearch_metadata = self._add_if_query_match_lineages_are_same(
            multisearch_metadata
        )

        self._write_parquet(
            multisearch_metadata,
            filtered=False,
        )

        # Remove self matches and likely spurious matches
        multisearch_metadata_filtered = multisearch_metadata.filter(
            (pl.col("query_md5") != pl.col("match_md5"))
            & (pl.col("intersect_hashes") > 1)
        )

        self._write_parquet(
            multisearch_metadata_filtered,
            filtered=True,
        )

        self.multisearch = multisearch_metadata
        self.multisearch_filtered = multisearch_metadata_filtered

        return multisearch_metadata_filtered

    def _make_multisearch_csv(
        self,
        query="astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.part_001.fa",
        against="astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.part_001.fa",
    ):
        basename = f"{query}--in--{against}.{self.moltype}.{self.ksize}.multisearch.csv"
        csv = f"{self.pipeline_outdir}/sourmash/multisearch/{basename}"
        return csv

    def _make_output_pq(self, filtered: bool):
        basename = f"scope40.multisearch.{self.moltype}.k{self.ksize}"
        pq = f"{self.analysis_outdir}/00_cleaned_multisearch_results/{basename}"
        if filtered:
            pq += ".filtered.pq"
        else:
            pq += ".pq"

        return pq

    def _write_parquet(
        self,
        df: pl.DataFrame,
        filtered: bool = False,
    ):
        pq = self._make_output_pq(filtered)
        if self.verbose:
            print(f"\nWriting {df.shape[0]} rows and {df.shape[1]} columns to {pq} ...")

        if pq.startswith("s3://"):
            # Writing Parquet files with arrow can be troublesome, need to use s3fs explicitly
            import s3fs

            fs = s3fs.S3FileSystem()
            with fs.open(pq, mode="wb") as f:
                df.sink_parquet(f)
        else:
            df.sink_parquet(pq)

        if self.verbose:
            print(f"\tDone.")

        df = pl.read_parquet(pq)
        return df


class ScopMetadataExtractor:

    def __init__(
        self,
        scop_name_series: Union[pl.Series, pl.DataFrame],
        scop_id_to_lineage_fixed: pl.Series = FOLDSEEK_SCOP_FIXED,
        name_prefix: str = "query",
        verbose: bool = False,
    ):
        self.scop_name_series = (
            scop_name_series.clone()
            if isinstance(scop_name_series, pl.Series)
            else scop_name_series.select(pl.col("*")).clone()
        )
        self.scop_fixed = scop_id_to_lineage_fixed.rename("scop_fixed")
        self.name_prefix = name_prefix
        self.verbose = verbose

    def _get_intersecting_scop_fixed(
        self, scop_id_to_lineage_original: pl.Series
    ) -> pl.Series:
        scop_df = pl.DataFrame(
            {
                "scop_original": scop_id_to_lineage_original,
                "scop_fixed": self.scop_fixed,
            }
        )

        scop_df = scop_df.with_columns(
            pl.when(pl.col("scop_fixed").is_null())
            .then(pl.col("scop_original"))
            .otherwise(pl.col("scop_fixed"))
            .alias("scop_merged")
        )

        if self.verbose:
            print(f"scop_df.shape: {scop_df.shape}")
            print(scop_df.head())

        if self.verbose:
            print("--- Different SCOP lineages ---")
            different = scop_df.filter(
                (pl.col("scop_fixed") != pl.col("scop_original"))
                & pl.col("scop_fixed").is_not_null()
            )
            print(f"different.shape: {different.shape}")
            print(different)

        scop_merged = scop_df.select("scop_merged").drop_nulls()
        scop_merged = scop_merged.unique(subset="scop_merged", keep="first")

        return scop_merged.to_series()

    def _subset_scop_name(
        self, name_series: pl.Series, index: int, new_colname: str
    ) -> pl.Series:
        subset = name_series.str.splitn(" ", 1).list.get(index)
        if self.verbose:
            print(f"--- {new_colname} ---")
            print(subset.value_counts())

        return subset.alias(new_colname)

    def _extract_scop_metadata_from_name(self, name_series: pl.Series) -> pl.DataFrame:
        name_series_no_dups = name_series.unique()
        scop_id = self._subset_scop_name(name_series_no_dups, 0, "scop_id")
        scop_lineage = self._subset_scop_name(name_series_no_dups, 1, "scop_lineage")

        scop_metadata = pl.DataFrame(
            {
                "name": name_series_no_dups,
                "scop_id": scop_id,
                "scop_lineage": scop_lineage,
            }
        )

        return scop_metadata.sort("scop_id")

    def _add_fixed_scop_lineage(self, scop_metadata: pl.DataFrame) -> pl.DataFrame:
        scop_lineage_fixed = self._get_intersecting_scop_fixed(
            scop_metadata["scop_lineage"],
        )

        return scop_metadata.with_columns(
            pl.col("scop_id")
            .map_dict(scop_lineage_fixed.to_dict())
            .alias("scop_lineage_fixed")
        )

    def _make_name_to_scop_lineages(self, scop_metadata: pl.DataFrame) -> pl.DataFrame:
        lineages_extracted = self._regex_extract_scop_lineages(
            scop_metadata["scop_lineage_fixed"]
        )
        return lineages_extracted.with_columns(pl.col("name").cast(pl.Utf8))

    def _make_scop_metadata(self) -> pl.DataFrame:
        scop_metadata = self._extract_scop_metadata_from_name(self.scop_name_series)
        return self._add_fixed_scop_lineage(scop_metadata)

    def _join_scop_metadata_with_extracted_lineages(
        self, scop_metadata: pl.DataFrame, lineages_extracted: pl.DataFrame
    ) -> pl.DataFrame:
        scop_metadata_with_extracted_lineages = scop_metadata.join(
            lineages_extracted, on="name"
        )

        new_column_names = {
            col: f"{self.name_prefix}_{col}"
            for col in scop_metadata_with_extracted_lineages.columns
        }
        scop_metadata_with_extracted_lineages = (
            scop_metadata_with_extracted_lineages.rename(new_column_names)
        )

        return scop_metadata_with_extracted_lineages.sort(f"{self.name_prefix}_name")

    def extract_scop_info_from_name(self) -> pl.DataFrame:
        scop_metadata = self._make_scop_metadata()
        lineages_extracted = self._make_name_to_scop_lineages(scop_metadata)
        return self._join_scop_metadata_with_extracted_lineages(
            scop_metadata, lineages_extracted
        )

    @staticmethod
    def _regex_extract_scop_lineages(scop_lineage_series: pl.Series) -> pl.DataFrame:
        pattern = (
            r"(?P<family>(?P<superfamily>(?P<fold>(?P<class>[a-z])\.\d+)\.\d+)\.\d+)"
        )
        return scop_lineage_series.str.extract(pattern)

    @staticmethod
    def convert_query_match_cols_to_categories(
        df: pl.DataFrame, query: str, match: str
    ) -> pl.DataFrame:
        # Polars doesn't have a direct equivalent to Pandas' Categorical type
        # Instead, we can use the Enum type to achieve similar functionality
        categories = sorted(df[query].unique().to_list())
        category_dict = {val: i for i, val in enumerate(categories)}

        return df.with_columns(
            [
                pl.col(query)
                .map_dict(category_dict)
                .cast(pl.UInt32)
                .alias(f"{query}_cat"),
                pl.col(match)
                .map_dict(category_dict)
                .cast(pl.UInt32)
                .alias(f"{match}_cat"),
            ]
        )


# --- Tests! --- #


@pytest.fixture
def testdata_folder():

    this_folder = os.path.join(os.path.dirname(__file__))
    data_folder = os.path.join(
        this_folder, "test-data", "process_scop_sourmash_multisearch"
    )
    return data_folder


@pytest.fixture
def input_name_series(testdata_folder):
    csv = os.path.join(testdata_folder, "scop_input_name_series.csv")
    return pl.read_csv(csv).to_series()


@pytest.fixture
def input_scop_fixed(testdata_folder):
    csv = os.path.join(testdata_folder, "scop_input_scop_fixed.csv")

    return pl.read_csv(
        csv,
        has_header=False,
    ).to_series()


@pytest.fixture
def true_scop_metadata(testdata_folder):
    csv = os.path.join(testdata_folder, "scop_output_scop_metadata.csv")
    return pl.read_csv(csv)


def test_extract_scop_info_from_name(
    input_name_series, input_scop_fixed, true_scop_metadata
):
    sme = ScopMetadataExtractor(input_name_series, input_scop_fixed, "query")
    test_scop_metadata = sme.extract_scop_info_from_name()

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
    pl.testing.assert_frame_equal(test_scop_metadata, true_scop_metadata)


@pytest.fixture
def true_multisearch_processed_filtered(testdata_folder):
    pq = os.path.join(
        testdata_folder, "multisearch_output_multisearch_results_processed_filtered.pq"
    )
    df = pl.read_parquet(pq)
    return df


def test_parse_scop_multisearch_results(
    true_multisearch_processed_filtered, testdata_folder
):
    # This parser reads:
    # 2024-kmerseek-analysis/notebooks/test-data/process_scop_sourmash_multisearch/multisearch_output_multisearch_results_processed_filtered.pq
    parser = MultisearchParser(
        pipeline_outdir=testdata_folder,
        moltype="protein",
        ksize=10,
        analysis_outdir=testdata_folder,
    )
    test_multisearch_processed_filtered = parser.process_multisearch_scop_results()

    assert os.path.exists(
        os.path.join(
            testdata_folder,
            "00_cleaned_multisearch_results",
            "scope40.multisearch.protein.k10.pq",
        )
    )
    assert os.path.exists(
        os.path.join(
            testdata_folder,
            "00_cleaned_multisearch_results",
            "scope40.multisearch.protein.k10.filtered.pq",
        )
    )

    # Change the index to be a range to match the known output data
    test_multisearch_processed_filtered.index = range(
        len(test_multisearch_processed_filtered)
    )

    # Test that output SCOP lineage counts are correct
    assert test_multisearch_processed_filtered[
        "query_family"
    ].value_counts().head().to_dict() == {
        "a.104.1.0": 38,
        "a.128.1.0": 17,
        "a.211.1.1": 13,
        "a.211.1.2": 5,
        "a.39.1.5": 4,
    }
    assert test_multisearch_processed_filtered[
        "query_superfamily"
    ].value_counts().head().to_dict() == {
        "a.104.1": 42,
        "a.128.1": 22,
        "a.211.1": 22,
        "a.102.1": 6,
        "a.39.1": 6,
    }

    assert test_multisearch_processed_filtered[
        "query_fold"
    ].value_counts().head().to_dict() == {
        "a.104": 42,
        "a.128": 22,
        "a.211": 22,
        "a.102": 8,
        "a.39": 6,
    }

    # Test for overall equality
    pl.testing.assert_frame_equal(
        test_multisearch_processed_filtered,
        true_multisearch_processed_filtered,
    )
