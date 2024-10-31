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
        chunk_size: int = 100000,
    ):
        self.query_metadata = query_metadata
        self.match_metadata = match_metadata
        self.pipeline_outdir = pipeline_outdir
        self.moltype = moltype
        self.ksize = ksize
        self.analysis_outdir = analysis_outdir
        self.check_same_cols = check_same_cols
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.tempfile = False

        # Expected columns in input parquet
        self.expected_columns = [
            "query_name",
            "query_md5",
            "match_name",
            "match_md5",
            "containment",
            "max_containment",
            "jaccard",
            "intersect_hashes",
            "prob_overlap",
            "prob_overlap_adjusted",
            "containment_adjusted",
            "containment_adjusted_log10",
            "tf_idf_score",
        ]

        self._setup_logging()

    def _setup_logging(self):
        if self.verbose:
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            logger.addHandler(handler)
        else:
            logging.basicConfig(level=None)

    def _validate_columns(self, df: pl.LazyFrame, stage: str) -> None:
        """Validate that required columns are present"""
        cols = df.columns
        notify(f"Columns at {stage}: {cols}")
        return df

    def _process_chunk(self, chunk: pl.LazyFrame) -> pl.LazyFrame:
        """Process a single chunk of data with all transformations"""
        chunk = self._validate_columns(chunk, "start of chunk processing")

        notify("Filtering self-matches and low hash matches")
        chunk = chunk.filter(
            (pl.col("query_md5") != pl.col("match_md5"))
            & (pl.col("intersect_hashes") > 1)
        )
        chunk = self._validate_columns(chunk, "after filtering")

        notify("Joining with metadata")
        # First join with query metadata
        chunk = chunk.join(
            self.query_metadata, left_on="query_name", right_on="query_name", how="left"
        )
        chunk = self._validate_columns(chunk, "after query metadata join")

        # Then join with match metadata
        chunk = chunk.join(
            self.match_metadata, left_on="match_name", right_on="match_name", how="left"
        )
        chunk = self._validate_columns(chunk, "after match metadata join")

        notify("Adding same columns checks")
        for col in self.check_same_cols:
            query, match = f"query_{col}", f"match_{col}"
            chunk = chunk.with_columns(
                [(pl.col(query) == pl.col(match)).alias(f"same_{col}")]
            )

        notify("Adding additional columns")
        # Pre-calculate columns for better memory efficiency
        additional_cols = [
            pl.lit(self.ksize).alias("ksize"),
            pl.lit(self.moltype).alias("moltype"),
        ]

        # Add log10 columns for metrics that exist
        metrics = [
            "prob_overlap_adjusted",
            "containment",
            "max_containment",
            "tf_idf_score",
            "jaccard",
        ]
        for metric in metrics:
            if metric in chunk.columns:
                additional_cols.append(pl.col(metric).log10().alias(f"log10_{metric}"))

        chunk = chunk.with_columns(additional_cols)
        chunk = self._validate_columns(chunk, "final")

        return chunk

    def _stream_process_file(self, filename: str) -> pl.LazyFrame:
        """Process the file in chunks to reduce memory usage"""
        notify(f"Processing file: {filename}")

        if filename.startswith("s3://"):
            notify("Downloading from S3...")
            self.tempfile = True
            temp_fp = self._download_from_s3(filename)
            filename = temp_fp.name

        # Read parquet schema first
        notify("Reading parquet schema...")
        schema = pl.read_parquet_schema(filename)
        notify(f"File schema: {schema}")

        # Create reader with explicit schema
        reader = pl.scan_parquet(filename, schema=schema)
        notify(f"Available columns in file: {reader.columns}")

        # Process in chunks
        temp_files = []
        try:
            for i, chunk in enumerate(
                reader.collect(streaming=True, chunk_size=self.chunk_size)
            ):
                notify(f"Processing chunk {i+1}")
                processed_chunk = self._process_chunk(chunk)

                # Save chunk to temporary parquet file
                temp_file = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
                processed_chunk.collect().write_parquet(temp_file.name)
                temp_files.append(temp_file.name)
                temp_file.close()

            # Combine all temporary files
            notify("Combining processed chunks...")
            combined = pl.scan_parquet(temp_files)

        finally:
            # Clean up temporary files
            notify("Cleaning up temporary files...")
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

            if self.tempfile:
                temp_fp.close()

        return combined

    def process_multisearch_scop_results(self):
        notify(f"\n--- moltype: {self.moltype}, ksize: {self.ksize} --")

        input_file = self._make_multisearch_input_file()
        result = self._stream_process_file(input_file)

        return result

    def _make_multisearch_input_file(
        self,
        query="astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.part_001.fa",
        against="astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.part_001.fa",
    ):
        basename = f"{query}--in--{against}.{self.moltype}.{self.ksize}.multisearch.pq"
        return f"{self.pipeline_outdir}/sourmash/multisearch/{basename}"

    def _download_from_s3(self, s3_path):
        fp = tempfile.NamedTemporaryFile(delete=True, prefix="/home/ec2-user/tmp/")
        session = boto3.Session()
        bucket_key = s3_path.split("s3://")[-1]
        bucket, key = bucket_key.split("/", 1)
        download_object_from_s3(session, bucket=bucket, key=key, filename=fp.name)
        return fp
