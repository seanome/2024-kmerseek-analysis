import os
from tempfile import NamedTemporaryFile
import polars as pl
from tqdm import tqdm


from scop_constants import same_scop_cols, n_scop_cols
from notifications import notify, notify_done
from sourmash_constants import sourmash_score_cols


class MultisearchSensitivityCalculator:
    TIDY_SCHEMA = pl.Schema(
        {
            "query_scop_id": str,
            "lineage": str,
            "sensitivity": float,
            "sensitivity_rank": int,
        }
    )

    MULTISEARCH_SCHEMA = {
        # Only include columns we actually need
        "query_name": pl.String,
        "match_name": pl.String,
        "query_scop_id": pl.String,
        "match_scop_id": pl.String,
        "query_family": pl.Categorical(ordering="physical"),
        "query_superfamily": pl.Categorical(ordering="physical"),
        "query_fold": pl.Categorical(ordering="physical"),
        "query_class": pl.Categorical(ordering="physical"),
        "n_family": pl.Int64,
        "n_superfamily": pl.Int64,
        "n_fold": pl.Int64,
        "n_class": pl.Int64,
        "match_family": pl.Categorical(ordering="physical"),
        "match_superfamily": pl.Categorical(ordering="physical"),
        "match_fold": pl.Categorical(ordering="physical"),
        "match_class": pl.Categorical(ordering="physical"),
        "same_family": pl.Boolean,
        "same_superfamily": pl.Boolean,
        "same_fold": pl.Boolean,
        "same_class": pl.Boolean,
    }

    def __init__(
        self,
        moltype,
        ksize,
        in_dir,
        out_dir,
        in_basename_template=r"scope40.multisearch.{moltype}.k{ksize}.filtered.pq",
        out_basename_template=r"scope40.multisearch.{moltype}.{ksize}.sensitivity_to_first_fp.pq",
        skip_if_out_exists=True,
        chunk_size=10000,  # Added parameter for controlling chunk size
    ):
        self.moltype = moltype
        self.ksize = ksize
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.in_basename_template = in_basename_template
        self.out_basename_template = out_basename_template
        self.in_pq = self.make_parquet_file(in_dir, self.in_basename_template)
        self.out_pq = self.make_parquet_file(out_dir, self.out_basename_template)
        self.skip_if_out_exists = skip_if_out_exists
        self.chunk_size = chunk_size
        notify(f"Processing {self.in_pq} with chunk size {chunk_size}")

    def make_parquet_file(self, folder, template):
        return os.path.join(
            folder,
            template.format(moltype=self.moltype, ksize=self.ksize),
        )

    def tidify_sensitivity(self, sensitivity):
        tidy = sensitivity.melt(
            id_vars=["query_scop_id"],
            value_vars=["family", "superfamily", "fold", "class"],
            value_name="sensitivity",
            variable_name="lineage",
        )

        kwargs = {} if isinstance(tidy, pl.DataFrame) else dict(schema=self.TIDY_SCHEMA)
        return tidy.group_by("lineage").map_groups(
            lambda x: x.with_columns(
                pl.col("sensitivity")
                .sort(descending=True)
                .rank(method="ordinal", descending=True)
                .alias("sensitivity_rank")
            ),
            **kwargs,
        )

    def process_chunk(self, chunk, sourmash_col):
        df = (
            chunk.sort(sourmash_col, descending=True)
            .group_by("query_scop_id")
            .agg(sensitivity_until_first_false_positive(same_scop_cols, n_scop_cols))
        ).fill_nan(0)

        tidy = self.tidify_sensitivity(df)
        return tidy.with_columns(pl.lit(sourmash_col).alias("sourmash_score"))

    def calculate_sensitivity(self, sourmash_cols=sourmash_score_cols):
        if self.skip_if_out_exists and os.path.exists(self.out_pq):
            notify(f"{self.out_pq} already exists, skipping")
            return

        temp_files = []

        # Get total number of rows for progress bar
        total_rows = pl.scan_parquet(self.in_pq).select(pl.count()).collect().item()
        chunks = total_rows // self.chunk_size + (
            1 if total_rows % self.chunk_size else 0
        )

        for sourmash_col in sourmash_cols:
            notify(f"Processing {sourmash_col}")
            temp_chunk_files = []

            # Process data in chunks
            for chunk_idx in tqdm(range(chunks)):
                chunk = pl.scan_parquet(
                    self.in_pq,
                    schema=self.MULTISEARCH_SCHEMA,
                    row_count_name="row_count",
                    row_count_offset=chunk_idx * self.chunk_size,
                    row_count_length=self.chunk_size,
                ).collect()

                if len(chunk) == 0:
                    continue

                result = self.process_chunk(chunk, sourmash_col)

                # Save chunk results
                temp_file = NamedTemporaryFile(
                    suffix=f".chunk{chunk_idx}.parquet",
                    delete=False,
                    prefix="/tmp/sensitivity",
                )
                result.collect().write_parquet(temp_file.name)
                temp_chunk_files.append(temp_file.name)
                temp_file.close()

            # Combine chunks for this sourmash_col
            combined_temp = NamedTemporaryFile(
                suffix=f".{sourmash_col}.parquet",
                delete=False,
                prefix="/tmp/sensitivity",
            )
            pl.concat([pl.scan_parquet(f) for f in temp_chunk_files]).sink_parquet(
                combined_temp.name
            )
            temp_files.append(combined_temp.name)

            # Clean up chunk files
            for f in temp_chunk_files:
                os.unlink(f)

        # Combine all results and add metadata
        notify("Combining all results...")
        final_df = pl.scan_parquet(temp_files).with_columns(
            [pl.lit(self.moltype).alias("moltype"), pl.lit(self.ksize).alias("ksize")]
        )

        notify(f"Writing final results to {self.out_pq}")
        final_df.sink_parquet(self.out_pq, row_group_size=10000)

        # Clean up temporary files
        for f in temp_files:
            os.unlink(f)


def sensitivity_until_first_false_positive(same_scop_cols, n_scop_cols):
    return [
        (pl.col(same_col).cast(pl.Float64).arg_min() / (pl.col(n_col) - 1))
        .first()
        .fill_null(0)
        .alias(same_col.replace("same_", ""))
        for same_col, n_col in zip(same_scop_cols, n_scop_cols)
    ]
