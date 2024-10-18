import tempfile

import polars as pl
import s3fs

# import boto3


def add_log10_col(df: pl.DataFrame, col: str):
    return df.with_columns(pl.col(col).log10().alias(f"{col}_log10"))


def _to_s3(writer, filename, use_temp=False):
    fs = s3fs.S3FileSystem()
    with fs.open(filename, mode="wb") as f:
        writer(f)


def _to_filename(writer, filename):
    if filename.startswith("s3://"):
        # Writing Parquet files with arrow can be troublesome, need to use s3fs explicitly
        _to_s3(writer, filename)
    else:
        writer(filename)


def write_parquet(df: pl.DataFrame, pq: str, verbose=False):
    if verbose:
        print(f"\nWriting {df.height} rows and {df.width} columns to {pq} ...")
    _to_filename(df.write_parquet, pq)
    if verbose:
        print("\tDone.")


def sink_parquet(df: pl.LazyFrame, pq: str, verbose=False):
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
            df.sink_parquet(f.name)

            fs = s3fs.S3FileSystem()
            fs.put(f.name, pq)
    else:
        df.sink_parquet(pq)

    print("\tDone.")
