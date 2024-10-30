import logging
import os
from typing import Literal, get_args, Union
import tempfile

import boto3
import polars as pl
from IPython.display import display
import s3fs
import pytest

from notifications import notify, notify_done
from s3_io import download_object_from_s3
from scop_constants import SCOP_LINEAGES, FOLDSEEK_SCOP_FIXED
from sourmash_constants import MOLTYPES

from polars_utils import save_parquet, add_log10_col, csv_pq, load_filename


class MultisearchParser:

    def __init__(
        self,
        query_metadata: pl.LazyFrame,
        match_metadata: pl.LazyFrame,
        pipeline_outdir: str,
        moltype: MOLTYPES,
        ksize: int,
        analysis_outdir: str,
        check_same_cols: list[str] = get_args(SCOP_LINEAGES),
        verbose: bool = False,
        input_filetype: csv_pq = "csv",
        lazy: bool = True,
    ):
        """_summary_

        Args:
            query_metadata (pl.LazyFrame): _description_
            match_metadata (pl.LazyFrame): _description_
            pipeline_outdir (str): _description_
            moltype (MOLTYPES): _description_
            ksize (int): _description_
            analysis_outdir (str): _description_
            check_same_cols (list[str], optional): _description_. Defaults to get_args(SCOP_LINEAGES).
            verbose (bool, optional): _description_. Defaults to False.
            input_filetype (csv_pq, optional): _description_. Defaults to "csv".
            lazy (bool, optional):
                Whether to use lazy loading of files with Polars, e.g. "read_parquet" loads the entire file
                into memory (not lazy), while "scan_parquet" doesn't load anything until it needs to, as
                indicated with .collect() or sink_parquet(). Defaults to True.
        """
        self.query_metadata = query_metadata
        self.match_metadata = match_metadata
        self.pipeline_outdir = pipeline_outdir
        self.moltype = moltype
        self.ksize = ksize
        self.analysis_outdir = analysis_outdir
        self.check_same_cols = check_same_cols
        self.verbose = verbose
        self.tempfile = False
        self.input_filetype = input_filetype
        self.lazy = lazy

        if self.verbose:
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)

            handler = logging.StreamHandler()
            logger.addHandler(handler)
        else:
            logging.basicConfig(level=None)

    def _download_from_s3(self, s3_path):
        # Get a "failed to allocate" error when try to scan big csvs from S3
        # This is a workaround
        fp = tempfile.NamedTemporaryFile(
            delete_on_close=True, prefix="/home/ec2-user/tmp/"
        )
        session = boto3.Session()
        bucket_key = s3_path.split("s3://")[-1]
        bucket, key = bucket_key.split("/", 1)
        download_object_from_s3(session, bucket=bucket, key=key, filename=fp.name)
        return fp

    def _read_multisearch(self) -> pl.DataFrame:
        notify(f"\n--- moltype: {self.moltype}, ksize: {self.ksize} --")
        self._multisearch_filename = self._make_multisearch_input_file()
        notify(f"Reading {self._multisearch_filename} ...")

        if self._multisearch_filename.startswith("s3://"):
            # Download file locally because polars.scan_* doesn't work with cloud files (e.g. objects on S3)
            notify(
                f"Downloading {os.path.basename(self._multisearch_filename)} locally ..."
            )
            self.tempfile = True
            self._input_fp = self._download_from_s3(self._multisearch_filename)
        else:
            self._input_fp = self._multisearch_filename

        multisearch = load_filename(self._input_fp, self.input_filetype, self.lazy)

        notify_done()
        return multisearch

    def _add_query_match_scop_metadata(self, multisearch: pl.DataFrame) -> pl.DataFrame:
        notify("\nJoining multisearch with query and match metadata ...")
        multisearch_metadata = multisearch.join(
            self.query_metadata, left_on="query_name", right_on="query_name"
        ).join(self.match_metadata, left_on="match_name", right_on="match_name")
        notify_done()
        return multisearch_metadata

    def _add_if_query_match_cols_are_same(
        self, multisearch_metadata: pl.DataFrame
    ) -> pl.DataFrame:

        for col in self.check_same_cols:
            query = f"query_{col}"
            match = f"match_{col}"
            same = f"same_{col}"

            notify("\nJoining multisearch with query and match metadata ...")

            multisearch_metadata = multisearch_metadata.with_columns(
                (pl.col(query) == pl.col(match)).alias(same)
            )
            notify_done()

        return multisearch_metadata

    def _add_columns(self, df: pl.DataFrame):
        """Add ksize, moltype, and log10 versions of value columns for comparisons"""
        df = df.with_columns(pl.lit(self.ksize).alias("ksize"))
        df = df.with_columns(pl.lit(self.moltype).alias("moltype"))

        df = add_log10_col(df, "prob_overlap_adjusted")
        df = add_log10_col(df, "containment")
        df = add_log10_col(df, "max_containment")
        df = add_log10_col(df, "tf_idf_score")
        df = add_log10_col(df, "jaccard")
        return df

    def _make_multisearch_input_file(
        self,
        query="astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.part_001.fa",
        against="astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.part_001.fa",
    ):
        basename = f"{query}--in--{against}.{self.moltype}.{self.ksize}.multisearch.{self.input_filetype}"
        filename = f"{self.pipeline_outdir}/sourmash/multisearch/{basename}"
        return filename

    def _make_output_pq(self, filtered: bool):
        basename = f"scope40.multisearch.{self.moltype}.k{self.ksize}"
        pq = f"{self.analysis_outdir}/00_cleaned_multisearch_results/{basename}"
        if filtered:
            pq += ".filtered.pq"
        else:
            pq += ".pq"

        return pq

    def _save_parquet(
        self,
        df: pl.LazyFrame,
        filtered: bool = False,
    ):
        pq = self._make_output_pq(filtered)
        notify(f"Saving multisearch file, filtered: {filtered}")
        save_parquet(df, pq, self.lazy, verbose=self.verbose)
        notify_done()
        return pq

    def process_multisearch_scop_results(self):
        multisearch = self._read_multisearch()
        multisearch_metadata = self._add_query_match_scop_metadata(multisearch)
        multisearch_metadata = self._add_if_query_match_cols_are_same(
            multisearch_metadata
        )
        multisearch_metadata = self._add_columns(multisearch_metadata)

        self._save_parquet(
            multisearch_metadata,
            filtered=False,
        )

        notify("Removing self-matches and matches with only one hash")
        # Remove self matches and likely spurious matches
        multisearch_metadata_filtered = multisearch_metadata.filter(
            (pl.col("query_md5") != pl.col("match_md5"))
            & (pl.col("intersect_hashes") > 1)
        )
        notify_done()

        self._save_parquet(
            multisearch_metadata_filtered,
            filtered=True,
        )

        self.multisearch = multisearch_metadata
        self.multisearch_filtered = multisearch_metadata_filtered
        if self.tempfile:
            notify("Closing temporary file")
            self._input_fp.close()
            notify_done()

        return multisearch_metadata_filtered


# --- Tests! --- #


@pytest.fixture
def testdata_folder():

    this_folder = os.path.join(os.path.dirname(__file__))
    data_folder = os.path.join(
        this_folder, "test-data", "process_scop_sourmash_multisearch"
    )
    return data_folder


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
    test_multisearch_processed_filtered = (
        parser.process_multisearch_scop_results().collect()
    )

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
