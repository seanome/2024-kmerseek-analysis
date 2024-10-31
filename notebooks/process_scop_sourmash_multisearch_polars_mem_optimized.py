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
        chunk_size: int = 100000,  # Added parameter for controlling chunk size
        schema=None,
    ):
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
        self.chunk_size = chunk_size
        self.schema = schema

        self._setup_logging()

    def _setup_logging(self):
        if self.verbose:
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            logger.addHandler(handler)
        else:
            logging.basicConfig(level=None)

    def _process_chunk(self, chunk: pl.LazyFrame) -> pl.LazyFrame:
        """Process a single chunk of data with all transformations"""
        # Filter first to reduce data size early
        notify("Removing self-matches and spurious single-hash matches")
        chunk = chunk.filter(
            (pl.col("query_md5") != pl.col("match_md5"))
            & (pl.col("intersect_hashes") > 1)
        )

        notify("Joining chunk with metadata")
        # Join with metadata
        chunk = chunk.join(
            self.query_metadata, left_on="query_name", right_on="query_name"
        ).join(self.match_metadata, left_on="match_name", right_on="match_name")

        # Add same columns
        for col in self.check_same_cols:
            query, match = f"query_{col}", f"match_{col}"
            chunk = chunk.with_columns(
                (pl.col(query) == pl.col(match)).alias(f"same_{col}")
            )

        # Add additional columns
        notify("Adding ksize, moltype and log10 versions of score value columns")
        chunk = chunk.with_columns(
            [
                pl.lit(self.ksize).alias("ksize"),
                pl.lit(self.moltype).alias("moltype"),
                pl.col("prob_overlap_adjusted")
                .log10()
                .alias("log10_prob_overlap_adjusted"),
                pl.col("containment").log10().alias("log10_containment"),
                pl.col("max_containment").log10().alias("log10_max_containment"),
                pl.col("tf_idf_score").log10().alias("log10_tf_idf_score"),
                pl.col("jaccard").log10().alias("log10_jaccard"),
            ]
        )

        return chunk

    def _stream_process_file(self, filename: str) -> pl.LazyFrame:
        """Process the file in chunks to reduce memory usage"""
        if filename.startswith("s3://"):
            notify(f"Downloading {os.path.basename(filename)} locally ...")
            self.tempfile = True
            temp_fp = self._download_from_s3(filename)
            filename = temp_fp.name

        # Create a LazyFrame for streaming
        if self.input_filetype == "csv":
            reader = pl.scan_csv(filename, schema=self.schema)
        else:  # parquet
            reader = pl.scan_parquet(filename, schema=self.schema)

        # Process in chunks and save to temporary parquet files
        temp_files = []
        for i, chunk in enumerate(
            reader.collect(streaming=True, chunk_size=self.chunk_size)
        ):
            notify(f"Processing chunk {i+1}")
            processed_chunk = self._process_chunk(pl.LazyFrame(chunk))

            # Save chunk to temporary parquet file
            temp_file = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
            processed_chunk.collect().write_parquet(temp_file.name)
            temp_files.append(temp_file.name)
            temp_file.close()

        # Combine all temporary files into a single LazyFrame
        combined = pl.scan_parquet(temp_files, low_memory=True)

        # Clean up temporary files
        for temp_file in temp_files:
            os.unlink(temp_file)

        if self.tempfile:
            temp_fp.close()

        return combined

    def process_multisearch_scop_results(self):
        notify(f"\n--- moltype: {self.moltype}, ksize: {self.ksize} --")

        # Get input filename
        input_file = self._make_multisearch_input_file()
        notify(f"Processing {input_file} ...")

        # Process file in streaming fashion
        result = self._stream_process_file(input_file)
        self._save_parquet(
            result,
            filtered=True,
        )
        # # Show optimization info if needed
        # result.show_graph(optimized=False)
        # result.explain(optimized=False)
        # result.show_graph(optimized=True)
        # result.explain(optimized=True)

        return result

    # Other helper methods remain the same
    def _download_from_s3(self, s3_path):
        fp = tempfile.NamedTemporaryFile(
            delete_on_close=True, prefix="/home/ec2-user/tmp/"
        )
        session = boto3.Session()
        bucket_key = s3_path.split("s3://")[-1]
        bucket, key = bucket_key.split("/", 1)
        download_object_from_s3(session, bucket=bucket, key=key, filename=fp.name)
        return fp

    def _make_multisearch_input_file(
        self,
        query="astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.part_001.fa",
        against="astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.part_001.fa",
    ):
        basename = f"{query}--in--{against}.{self.moltype}.{self.ksize}.multisearch.{self.input_filetype}"
        return f"{self.pipeline_outdir}/sourmash/multisearch/{basename}"
