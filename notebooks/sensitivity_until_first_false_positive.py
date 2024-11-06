import os
from tempfile import NamedTemporaryFile

import polars as pl
from tqdm import tqdm


from scop_constants import same_scop_cols, n_scop_cols
from notifications import notify, notify_done


def sensitivity_until_first_false_positive(same_scop_cols, n_scop_cols):
    return [
        (pl.col(same_col).cast(pl.Float64).arg_min() / (pl.col(n_col) - 1))
        .first()
        .fill_null(0)
        .alias(same_col.replace("same_", ""))
        for same_col, n_col in zip(same_scop_cols, n_scop_cols)
    ]


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


def tidify_sensitivity(sensitivity):
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
    kwargs = {} if isinstance(tidy, pl.DataFrame) else dict(schema=TIDY_SCHEMA)
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


def compute_sensitivity(
    multisearch,
    sourmash_cols,
    moltype,
    ksize,
    same_scop_cols=same_scop_cols,
    n_scop_cols=n_scop_cols,
):
    temp_files = []

    for sourmash_col in tqdm(sourmash_cols):
        df = (
            multisearch.sort(sourmash_col, descending=True)
            # .head(1000)
            .group_by("query_scop_id").agg(
                sensitivity_until_first_false_positive(same_scop_cols, n_scop_cols)
            )
        ).fill_nan(0)
        tidy = tidify_sensitivity(df)
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
    sensitivity = sensitivity.with_columns(pl.lit(moltype).alias("moltype"))
    sensitivity = sensitivity.with_columns(pl.lit(ksize).alias("ksize"))

    # Clean up temporary files
    for temp_file in temp_files:
        os.unlink(temp_file)

    return sensitivity
