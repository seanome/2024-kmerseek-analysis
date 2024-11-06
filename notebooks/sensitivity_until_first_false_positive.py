import os
from tempfile import NamedTemporaryFile

import polars as pl
from tqdm import tqdm


from scop_constants import same_scop_cols, n_scop_cols
from notifications import notify, notify_done
from sourmash_constants import sourmash_score_cols


class MultisearchSensitivityCalculator:

    TIDY_SCHEMA = pl.Schema(
        dict(
            [
                ("query_scop_id", str),
                ("lineage", str),
                ("sensitivity", float),
                ("sensitivity_rank", int),
            ]
        )
    )

    MULTISEARCH_SCHEMA = {
        "query_name": pl.String,
        "query_md5": pl.String,
        "match_name": pl.String,
        "match_md5": pl.String,
        "containment": pl.Float64,
        "max_containment": pl.Float64,
        "jaccard": pl.Float64,
        "intersect_hashes": pl.Float64,
        "prob_overlap": pl.Float64,
        "prob_overlap_adjusted": pl.Float64,
        "containment_adjusted": pl.Float64,
        "containment_adjusted_log10": pl.Float64,
        "tf_idf_score": pl.Float64,
        "query_family": pl.Categorical(ordering="physical"),
        "query_superfamily": pl.Categorical(ordering="physical"),
        "query_fold": pl.Categorical(ordering="physical"),
        "query_class": pl.Categorical(ordering="physical"),
        "n_family": pl.Int64,
        "n_superfamily": pl.Int64,
        "n_fold": pl.Int64,
        "n_class": pl.Int64,
        "query_scop_id": pl.String,
        "match_family": pl.Categorical(ordering="physical"),
        "match_superfamily": pl.Categorical(ordering="physical"),
        "match_fold": pl.Categorical(ordering="physical"),
        "match_class": pl.Categorical(ordering="physical"),
        "match_scop_id": pl.String,
        "same_family": pl.Boolean,
        "same_superfamily": pl.Boolean,
        "same_fold": pl.Boolean,
        "same_class": pl.Boolean,
        "ksize": pl.Int32,
        "moltype": pl.String,
        "log10_prob_overlap_adjusted": pl.Float64,
        "log10_containment": pl.Float64,
        "log10_max_containment": pl.Float64,
        "log10_tf_idf_score": pl.Float64,
        "log10_jaccard": pl.Float64,
    }

    def __init__(
        self,
        moltype,
        ksize,
        in_dir,
        out_dir,
        in_basename_template=r"scope40.multisearch.{moltype}.k{ksize}.filtered.pq",
        out_basename_template=r"scope40.multisearch.{moltype}.{ksize}.sensitivity_to_first_fp.pq",
        skip_if_out_pq_exists=True,
    ):
        notify(f"--- moltype: {moltype}, ksize: {ksize} --")
        self.moltype = moltype
        self.ksize = ksize
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.in_basename_template = in_basename_template
        self.out_basename_template = out_basename_template
        self.in_pq = self.make_parquet_file(in_dir, self.in_basename_template)
        self.out_pq = self.make_parquet_file(out_dir, self.out_basename_template)
        self.skip_if_out_pq_exists = skip_if_out_pq_exists
        notify(f"pq out: {self.out_pq}")
        self.read_multisearch()

    def read_multisearch(self):
        notify(f"Reading {self.in_pq} ...")
        self.multisearch = pl.scan_parquet(
            self.in_pq, schema=self.MULTISEARCH_SCHEMA, parallel="row_groups"
        )
        notify_done()

    def make_parquet_file(self, folder, template):
        return os.path.join(
            folder,
            template.format(moltype=self.moltype, ksize=self.ksize),
        )

    def tidify_sensitivity(self, sensitivity):
        tidy = sensitivity.melt(
            id_vars=["query_scop_id"],
            value_vars=[
                "family",
                "superfamily",
                "fold",
                "class",
            ],
            value_name="sensitivity",
            variable_name="lineage",
        )

        # Add ranks for all SCOP ids, descending from most sensitive first
        tidy = tidy.with_columns(pl.col("sensitivity").sort(descending=True))

        # DataFrame: No schema
        # Lazyframe: Must provide schema for map_groups
        kwargs = {} if isinstance(tidy, pl.DataFrame) else dict(schema=self.TIDY_SCHEMA)
        tidy = tidy.group_by("lineage").map_groups(
            lambda x: x.with_columns(
                pl.col("sensitivity")
                .sort(descending=True)
                .rank(method="ordinal", descending=True)
                .alias("sensitivity_rank")
            ),
            **kwargs,
        )
        return tidy

    def calculate_sensitivity(
        self,
        sourmash_cols=sourmash_score_cols,
        same_scop_cols=same_scop_cols,
        n_scop_cols=n_scop_cols,
    ):
        if self.skip_if_out_pq_exists:
            if os.path.exists(self.out_pq):
                notify(f"{self.out_pq} already exists, not doing anything")
                return

        temp_files = []

        for sourmash_col in tqdm(sourmash_cols):
            df = (
                self.multisearch.sort(sourmash_col, descending=True)
                # .head(1000)
                .group_by("query_scop_id").agg(
                    sensitivity_until_first_false_positive(same_scop_cols, n_scop_cols)
                )
            ).fill_nan(0)
            tidy = self.tidify_sensitivity(df)
            tidy = tidy.with_columns(pl.lit(sourmash_col).alias("sourmash_score"))
            # Save chunk to temporary parquet file
            temp_file = NamedTemporaryFile(suffix=".parquet", delete=False)
            notify(
                f"Writing '{sourmash_col}' sensitivity dataframe to {temp_file.name} ..."
            )
            tidy.collect().write_parquet(temp_file.name)
            temp_files.append(temp_file.name)
            temp_file.close()
            notify_done()

        notify("Concatenating dataframes ...")
        sensitivity = pl.scan_parquet(temp_files, low_memory=True)
        notify_done()
        # print("sensitivity.shape:", sensitivity.shape)
        sensitivity = sensitivity.with_columns(pl.lit(self.moltype).alias("moltype"))
        sensitivity = sensitivity.with_columns(pl.lit(self.ksize).alias("ksize"))

        notify(f"Writing {self.out_pq} ... ")
        sensitivity.sink_parquet(self.out_pq, row_group_size=1000)
        notify_done()

        # Clean up temporary files
        for temp_file in temp_files:
            os.unlink(temp_file)

        return sensitivity


def sensitivity_until_first_false_positive(same_scop_cols, n_scop_cols):
    return [
        (pl.col(same_col).cast(pl.Float64).arg_min() / (pl.col(n_col) - 1))
        .first()
        .fill_null(0)
        .alias(same_col.replace("same_", ""))
        for same_col, n_col in zip(same_scop_cols, n_scop_cols)
    ]
