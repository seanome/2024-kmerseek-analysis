#!/usr/bin/env python3
"""Download the ELM instance and class tables and keep the experimentally supported instances.

ELM (the Eukaryotic Linear Motif resource) curates short linear motifs from the
literature. Each *instance* is one motif at one position in one protein; each *class* is a
motif type with a regular expression that describes it. An instance's InstanceLogic is
"true positive" when the curators accept the experiments as showing a working motif;
the other values are "false positive", "true negative" and "unknown", and all three are
dropped here.

Writes into --out-dir:

  elm_instances.tsv          the instance table as downloaded, header comments included
  elm_classes.tsv            the class table as downloaded
  elm_instances_tp.parquet   one row per true-positive instance, with the class regex

Columns of elm_instances_tp.parquet:

  elm_instance     ELM instance accession (ELMI...)
  elm_class        class identifier, e.g. LIG_SH3_3
  elm_type         the class prefix: LIG, MOD, DOC, DEG, TRG or CLV
  accession        UniProt accession the instance is on (Primary_Acc; may be an isoform)
  start, end       0-based, end-exclusive (ELM's 1-based inclusive Start minus 1, End as is),
                   the convention kmerseek's region tables use
  motif_length     end - start
  organism         ELM's organism name
  regex            the class regular expression, from the class table
  methods, pmids   ELM's evidence columns, kept as text

The download URLs are in the block below and are printed on every run, with the release
version and date ELM writes into each file's header.

Usage:
    python scripts/fetch_elm.py --out-dir /Users/olga/data/elm-motif-transfer
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import polars as pl

# ---- download URLs ------------------------------------------------------------------
# Both are the tab-separated exports linked from http://elm.eu.org/downloads.html.
# `q=*` asks the instance search for every instance.
ELM_INSTANCES_URL = "http://elm.eu.org/instances.tsv?q=*"
ELM_CLASSES_URL = "http://elm.eu.org/elms/elms_index.tsv"
# --------------------------------------------------------------------------------------

DEFAULT_OUT = Path("/Users/olga/data/elm-motif-transfer")


def download(url: str, out: Path, refresh: bool) -> str:
    if out.exists() and not refresh:
        print(f"  cached: {out}")
        return out.read_text()
    with urllib.request.urlopen(url, timeout=300) as r:
        text = r.read().decode("utf-8")
    # ELM serves errors as an HTML page with status 200; a TSV export starts with '#'.
    if not text.startswith("#"):
        raise SystemExit(
            f"{url} did not return an ELM TSV (first bytes: {text[:80]!r})"
        )
    out.write_text(text)
    print(f"  downloaded: {out} ({len(text):_} bytes)")
    return text


def header_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("#")]


def read_tsv(text: str) -> pl.DataFrame:
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    return pl.read_csv(io.StringIO(body), separator="\t", infer_schema_length=0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--refresh", action="store_true", help="download again even if cached"
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("ELM download URLs (check these):")
    print(f"  instances: {ELM_INSTANCES_URL}")
    print(f"  classes:   {ELM_CLASSES_URL}")
    inst_text = download(
        ELM_INSTANCES_URL, args.out_dir / "elm_instances.tsv", args.refresh
    )
    cls_text = download(ELM_CLASSES_URL, args.out_dir / "elm_classes.tsv", args.refresh)
    for name, text in (("instances", inst_text), ("classes", cls_text)):
        print(
            f"  {name} header: " + " | ".join(h.lstrip("#") for h in header_lines(text))
        )

    inst = read_tsv(inst_text)
    cls = read_tsv(cls_text)
    print(f"\ninstances in file: {inst.height:_}")
    print(inst["InstanceLogic"].value_counts().sort("count", descending=True))

    tp = inst.filter(pl.col("InstanceLogic") == "true positive")
    regex = cls.select(
        pl.col("ELMIdentifier").alias("elm_class"), pl.col("Regex").alias("regex")
    )
    out = (
        tp.select(
            pl.col("Accession").alias("elm_instance"),
            pl.col("ELMIdentifier").alias("elm_class"),
            pl.col("ELMType").alias("elm_type"),
            pl.col("Primary_Acc").alias("accession"),
            (pl.col("Start").cast(pl.Int64) - 1).alias("start"),
            pl.col("End").cast(pl.Int64).alias("end"),
            pl.col("Organism").alias("organism"),
            pl.col("Methods").alias("methods"),
            pl.col("References").alias("pmids"),
        )
        .with_columns(motif_length=pl.col("end") - pl.col("start"))
        .join(regex, on="elm_class", how="left")
        .sort("elm_class", "accession", "start")
    )
    no_regex = out.filter(pl.col("regex").is_null())
    if no_regex.height:
        print(
            f"\nWARNING: {no_regex.height} instances have a class missing from the class table:"
        )
        print(no_regex["elm_class"].unique().to_list())
    bad = out.filter(pl.col("motif_length") <= 0)
    if bad.height:
        sys.exit(f"{bad.height} instances have End before Start:\n{bad}")

    path = args.out_dir / "elm_instances_tp.parquet"
    out.write_parquet(path)
    human = out.filter(pl.col("organism") == "Homo sapiens")
    print(
        f"\ntrue-positive instances: {out.height:_} in {out['elm_class'].n_unique()} classes, "
        f"on {out['accession'].n_unique():_} proteins"
    )
    print(
        f"  human: {human.height:_} instances on {human['accession'].n_unique():_} proteins"
    )
    print(
        f"  other species: {out.height - human.height:_} instances on "
        f"{out.filter(pl.col('organism') != 'Homo sapiens')['accession'].n_unique():_} proteins"
    )
    print(
        f"motif length (aa): median {out['motif_length'].median():.0f}, "
        f"25-75% {out['motif_length'].quantile(0.25):.0f}-{out['motif_length'].quantile(0.75):.0f}, "
        f"max {out['motif_length'].max()}, >= 30 aa: {(out['motif_length'] >= 30).sum()}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
