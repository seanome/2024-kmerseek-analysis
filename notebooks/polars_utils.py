import tempfile
from typing import Literal, Union

import polars as pl
import s3fs

# import boto3
from s3_io import temp_download_s3_path, simple_upload_s3_path
from notifications import notify, notify_done

# Typing Constants
polars_frames = pl.DataFrame | pl.LazyFrame
csv_pq = Literal["csv", "pq"]


def iter_slices(df: pl.LazyFrame, batch_size: int):
    """Itereate over row slices of a LazyFrame"""

    # From https://github.com/pola-rs/polars/issues/10683#issuecomment-2167802219
    def get_batch(df, offset, batch_size):
        batch = df.slice(offset, batch_size)
        batch = batch.collect(streaming=True)
        return batch

    batch = get_batch(df, 0, batch_size)
    # Yield once even if we got passed an empty LazyFrame
    yield batch
    offset = len(batch)
    if offset:
        while True:
            batch = get_batch(df, offset, batch_size)
            len_ = len(batch)
            if len_:
                offset += len_
                yield batch
            else:
                break


def add_log10_col(df: pl.DataFrame, col: str):
    notify(f"Creating log10 version of {col}")
    df = df.with_columns(pl.col(col).log10().alias(f"{col}_log10"))
    notify_done()
    return df


def _to_s3(writer, filename, use_temp=True):
    if use_temp:
        # Write to a local temporary file first
        with tempfile.NamedTemporaryFile() as f:
            writer(f.name)
            simple_upload_s3_path(f.name, filename)
    else:
        fs = s3fs.S3FileSystem()
        with fs.open(filename, mode="wb") as f:
            writer(f)


def _to_filename(writer, filename, use_temp=True):
    if filename.startswith("s3://"):
        # Writing Parquet files with arrow can be troublesome, need to use s3fs explicitly
        _to_s3(writer, filename, use_temp=use_temp)
    else:
        writer(filename)


def scan_filename(filename: str, filetype: csv_pq, **kwargs):
    if filetype == "csv":
        return pl.scan_csv(filename, **kwargs)
    elif filetype == "pq":
        return pl.scan_parquet(filename, **kwargs)


def read_filename(filename: str, filetype: csv_pq, **kwargs):
    if filetype == "csv":
        return pl.read_csv(filename, **kwargs)
    elif filetype == "pq":
        return pl.read_parquet(filename, **kwargs)


def load_filename(
    filename: str, filetype: csv_pq, lazy: bool, **kwargs
) -> polars_frames:
    """Combined interface for loading csv or parquet data either lazily or not

    Args:
        filename (str): Name of the file to load
        filetype (csv_pq): Either "csv" or "pq" for the parquet filetype
        lazy (bool): if True, then use `scan_csv` or `scan_parquet` and return
        a pl.LazyFrame. Otherwise, return a pl.DataFrame

    Returns:
        df: Either a pl.LazyFrame or pl.DataFrame
    """
    if lazy:
        return scan_filename(filename, filetype, **kwargs)
    else:
        return read_filename(filename, filetype, **kwargs)


def sink_parquet(df: pl.LazyFrame, pq: str, verbose=False, **kwargs):
    if verbose:
        try:
            print(
                f"\nWriting {df.select(pl.len()).collect().item()} rows and {len(df.columns)} columns to {pq} ..."
            )
        except pl.exceptions.ComputeError:
            pass
    # _to_filename(df.sink_parquet, pq)
    if pq.startswith("s3://"):
        # LazyFrame sink_parquet can't stream to cloud currently (polars version 1.9.0)
        # Need to write to a local file and then push the file to S3 with s3fs
        with tempfile.NamedTemporaryFile() as f:
            df.sink_parquet(f.name, **kwargs)
            simple_upload_s3_path(f.name, pq)
    else:
        df.sink_parquet(pq, **kwargs)

    print("\tDone.")


def write_parquet(df: pl.DataFrame, pq: str, verbose=False, **kwargs):
    if verbose:
        print(f"\nWriting {df.height} rows and {df.width} columns to {pq} ...")
    _to_filename(df.write_parquet, pq, **kwargs)
    if verbose:
        print("\tDone.")


def save_parquet(df: polars_frames, pq: str, lazy, verbose=False, **kwargs):
    if lazy:
        return sink_parquet(df, pq, verbose, **kwargs)
    else:
        return write_parquet(df, pq, verbose, **kwargs)


def scan_csv_sink_parquet(csv, parquet=None, verbose=False):
    """Scan a CSV and sink into a parquet file with Polars. Memory-efficient"""
    if parquet == None:
        parquet = csv.replace(".csv", ".pq")

    temp_fp = temp_download_s3_path(csv)
    df = pl.scan_csv(temp_fp.name)
    sink_parquet(df, parquet, verbose=verbose)
    return parquet
