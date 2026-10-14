import glob, sys
import polars as pl

O = sys.argv[1]
fs = sorted(glob.glob(O + "/*/scored/*/*.metrics.parquet"))
m = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed")
m = m.filter((pl.col("split") == "all") & (pl.col("stratum_axis") == "all"))
m.write_csv(O + "/dark_fmax_metrics.csv")
print(len(fs), "files,", m.height, "rows")
print(
    m.group_by("species", "truth_set", "tool")
    .len()
    .sort("species", "truth_set", "tool")
)
