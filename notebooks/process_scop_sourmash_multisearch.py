from typing import Literal, get_args
import pandas as pd
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

    def _read_multisearch_csv(self) -> pd.DataFrame:
        if self.verbose:
            print(f"\n\n--- moltype: {self.moltype}, ksize: {self.ksize} --")
        csv = self._make_multisearch_csv(self.pipeline_outdir, self.moltype, self.ksize)
        if self.verbose:
            print(f"\nReading {csv} ...")
        multisearch = pd.read_csv(csv)
        if self.verbose:
            print("\tDone")
        return multisearch

    def _add_query_match_scop_metadata(self, multisearch: pd.DataFrame) -> pd.DataFrame:
        """Add SCOP lineage information: class, fold, superfamily, family, for each SCOP entry

        Args:
            multisearch (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        query_metadata = extract_scop_info_from_name(
            multisearch.query_name, FOLDSEEK_SCOP_FIXED, "query", verbose=False
        )

        match_metadata = extract_scop_info_from_name(
            multisearch.match_name, FOLDSEEK_SCOP_FIXED, "match", verbose=False
        )

        multisearch_metadata = multisearch.join(query_metadata, on="query_name").join(
            match_metadata, on="match_name"
        )
        return multisearch_metadata

    def _add_if_query_match_lineages_are_same(self, multisearch_metadata: pd.DataFrame):
        """Adds a column to say whether the query and match have the same SCOP lineages

        Modifies "multisearch_metadata" dataframe in place

        Args:
            multisearch_metadata (_type_): _description_
        """
        for lineage_col in get_args(SCOP_LINEAGES):
            query = f"query_{lineage_col}"
            match = f"match_{lineage_col}"
            same = f"same_{lineage_col}"

            multisearch_metadata[same] = (
                multisearch_metadata[query] == multisearch_metadata[match]
            )

            ScopMetadataExtractor.convert_query_match_cols_to_categories(
                multisearch_metadata, query, match
            )

    def process_multisearch_scop_results(self):
        """This is the main function"""

        multisearch = self._read_multisearch_csv()
        multisearch_metadata = self._add_query_match_scop_metadata(multisearch)
        self._add_if_query_match_lineages_are_same()

        self.write_parquet(
            multisearch_metadata,
            filtered=False,
        )

        # Remove self matches and likely spurious matches
        multisearch_metadata_filtered = multisearch_metadata.query(
            "query_md5 != match_md5 and intersect_hashes > 1"
        )

        self.write_parquet(
            multisearch_metadata_filtered,
            filtered=True,
        )
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
        df: pd.DataFrame,
        filtered: bool = False,
    ):
        pq = self.make_output_pq(filtered)
        if self.verbose:
            print(f"\nWriting {len(df)} rows and {len(df.columns)} columns to {pq} ...")
        df.to_parquet(pq)
        if self.verbose:
            print(f"\tDone.")


class ScopMetadataExtractor:

    def __init__(
        self,
        scop_name_series: pd.Series,
        scop_id_to_lineage_fixed: pd.Series = FOLDSEEK_SCOP_FIXED,
        name_prefix: pd.Series = "query",
        verbose: bool = False,
    ):
        self.scop_name_series = scop_name_series.copy()
        self.scop_fixed = scop_id_to_lineage_fixed
        self.name_prefix = name_prefix
        self.verbose = verbose

        self.scop_fixed.name = "scop_fixed"

    def _get_intersecting_scop_fixed(self, scop_id_to_lineage_original: pd.Series):

        scop_id_to_lineage_original.name = "scop_original"
        scop_df = scop_id_to_lineage_original.to_frame().join(self.scop_fixed)

        scop_df["scop_merged"] = (
            scop_df["scop_fixed"].copy().fillna(scop_df["scop_original"])
        )

        if self.verbose:
            print(f"scop_df.shape: {scop_df.shape}")
            display(scop_df.head())

        if self.verbose:
            print("--- Different SCOP lineages ---")
            different = scop_df.query("scop_fixed != scop_original").dropna(
                subset="scop_fixed"
            )
            print(f"different.shape: {different.shape}")
            display(different)

        scop_merged = scop_df["scop_merged"]
        scop_merged = scop_merged[~scop_merged.index.duplicated()]

        return scop_merged

    def _subset_scop_name(
        self, name_series: pd.Series, index: int, new_colname: str
    ) -> pd.Series:
        subset = name_series.str.split().str[index]
        subset.name = new_colname
        if self.verbose:
            print(f"--- {new_colname} ---")
            print(subset.value_counts())

        subset.index = name_series.values
        return subset

    def _extract_scop_metadata_from_name(self, name_series: pd.Series) -> pd.DataFrame:
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
        name_series_no_dups = name_series.drop_duplicates()
        scop_id = self._subset_scop_name(name_series_no_dups, 0, "scop_id")
        scop_lineage = self._subset_scop_name(name_series_no_dups, 1, "scop_lineage")
        scop_metadata = scop_id.to_frame()
        scop_metadata = scop_metadata.join(scop_lineage)
        scop_metadata.index.name = "name"
        scop_metadata = scop_metadata.reset_index()
        scop_metadata = scop_metadata.set_index("scop_id")

        return scop_metadata

    def _add_fixed_scop_lineage(self, scop_metadata: pd.DataFrame):
        """Modifies scop_metadata in-place"""
        scop_lineage_fixed = self._get_intersecting_scop_fixed(
            scop_metadata["scop_lineage"],
        )

        # Use the fixed lineage first
        scop_metadata["scop_lineage_fixed"] = scop_metadata.index.map(
            scop_lineage_fixed
        )
        return scop_metadata

    def _make_name_to_scop_lineages(self, scop_metadata: pd.DataFrame) -> pd.DataFrame:
        lineages_extracted = self._regex_extract_scop_lineages(
            scop_metadata["scop_lineage_fixed"]
        )
        lineages_extracted.index = scop_metadata["name"].values
        return lineages_extracted

    def _make_scop_metadata(self):
        scop_metadata = self._extract_scop_metadata_from_name(self.scop_name_series)
        self._add_fixed_scop_lineage(scop_metadata)
        return scop_metadata

    def _join_scop_metadata_with_extracted_lineages(
        self, scop_metadata: pd.DataFrame, lineages_extracted: pd.DataFrame
    ) -> pd.DataFrame:
        scop_metadata_with_extracted_lineages = scop_metadata.join(
            lineages_extracted, on="name"
        )
        scop_metadata_with_extracted_lineages = (
            scop_metadata_with_extracted_lineages.reset_index()
        )
        scop_metadata_with_extracted_lineages.columns = (
            f"{self.name_prefix}_" + scop_metadata_with_extracted_lineages.columns
        )

        scop_metadata_with_extracted_lineages = (
            scop_metadata_with_extracted_lineages.set_index(f"{self.name_prefix}_name")
        )
        return scop_metadata_with_extracted_lineages

    def extract_scop_info_from_name(
        self,
    ) -> pd.DataFrame:
        scop_metadata = self._make_scop_metadata()

        lineages_extracted = self._make_name_to_scop_lineages(scop_metadata)

        scop_metadata_with_extracted_lineages = (
            self._join_scop_metadata_with_extracted_lineages(
                scop_metadata, lineages_extracted
            )
        )
        return scop_metadata_with_extracted_lineages

    @staticmethod
    def _regex_extract_scop_lineages(scop_lineage_series: pd.Series) -> pd.DataFrame:
        """Use regex to extract SCOP lineage information"""
        # Pattern from: https://regex101.com/r/gKBJAr/1
        pattern = (
            r"(?P<family>(?P<superfamily>(?P<fold>(?P<class>[a-z])\.\d+)\.\d+)\.\d+)"
        )
        lineages_extracted = scop_lineage_series.str.extractall(pattern)
        lineages_extracted = lineages_extracted.droplevel(-1)
        return lineages_extracted

    @staticmethod
    def convert_query_match_cols_to_categories(
        df: pd.DataFrame, query: str, match: str
    ):
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
    sme = ScopMetadataExtractor(test_name_series, test_scop_fixed, "query")
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
    pd.testing.assert_frame_equal(test_scop_metadata, true_scop_metadata)
