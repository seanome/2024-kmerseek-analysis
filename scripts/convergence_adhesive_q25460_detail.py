"""Per-list detail for the mussel foot protein 1 (Q25460) search.

bin/score.py gives counts per ranked list. For a case study of one query the notebook also
needs to see which proteins those are, so this writes, for every ranked list (a kmerseek
setting, a sequence tool, a Foldseek mode or a scan):

  top_hits.tsv            the real query's top --top rows: rank, target, score, whether
                          the target is in the adhesive truth set and its lineage, and
                          the UniProt protein name
  truth_at_threshold.tsv  for every query (real and null), how many truth entries of
                          each lineage score above the real query's first false positive:
                          the same cutoff for the real query and its 40 null draws. Q25460's
                          own entry is not counted, for any query
  first_false_positive.tsv  the real query's first false positive: target, name, score
  truth_ranks.tsv         the real query's rank and score for every truth entry in the
                          whole list; an entry the list never reported gets no row. A
                          scan keeps its top 2_000 rows plus every truth entry, so for a
                          scan the rank is the one scans.py computed over all of the
                          reference (its `rank` column, ties sharing the lowest rank), not
                          the position among the rows kept

A list with no false positive for the real query gets no cutoff, and its rows in
truth_at_threshold.tsv count every reported row instead (`cutoff` is empty there).

Run on Sherlock inside the pipeline image (it has polars), from the pipeline folder:
  python3 ../../scripts/convergence_adhesive_q25460_detail.py \
      --hits results-adhesive-q25460/hits \
      --draws results-adhesive-q25460/inputs/draws.tsv \
      --truth ../../data/convergence_adhesive/truth_set.tsv \
      --swissprot data/uniprot_sprot.2026_03.fasta.gz \
      --outdir results-adhesive-q25460/detail
"""

import argparse
import gzip
import sys
from pathlib import Path

import polars as pl

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "nextflow-runs/convergence-afp/bin")
)
import score  # noqa: E402  (bin/score.py: the same list readers the scoring step uses)

QUERY = "Q25460__real"


def protein_names(swissprot: Path) -> dict[str, str]:
    """Accession -> protein name, from the UniProt FASTA header (text before OS=)."""
    names = {}
    with gzip.open(swissprot, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                first, _, rest = line[1:].partition(" ")
                acc = first.split("|")[1]
                names[acc] = rest.split(" OS=")[0].strip()
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hits", type=Path, required=True)
    ap.add_argument("--draws", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--swissprot", type=Path, required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    truth = pl.read_csv(args.truth, separator="\t")
    truth_lineage = dict(zip(truth["Entry"], truth["lineage"]))
    names = protein_names(args.swissprot)
    names.update(dict(zip(truth["Entry"], truth["Protein names"])))
    draws = pl.read_csv(args.draws, separator="\t")
    kind_of = dict(zip(draws["query_id"], draws["kind"]))
    lineages = sorted(truth["lineage"].unique())

    sources = list(score.lists_from_scans(args.hits / "scans" / "scans.parquet", [0]))
    sources += list(score.lists_from_tools(args.hits))
    sources += list(score.lists_from_kmerseek(args.hits))

    top_rows, count_rows, fp_rows, rank_rows = [], [], [], []
    for method, setting, _, frame in sources:
        ranked = frame.sort(["query_id", "score"], descending=[False, True])
        real = ranked.filter(pl.col("query_id") == QUERY).with_columns(
            rank=pl.int_range(1, pl.len() + 1)
        )
        fp = real.filter(~pl.col("target").is_in(list(truth_lineage)))
        cutoff = float(fp["score"][0]) if fp.height else None
        fp_rows.append(
            {
                "method": method,
                "setting_label": setting,
                "n_reported": real.height,
                "first_fp_rank": int(fp["rank"][0]) if fp.height else None,
                "first_fp_target": fp["target"][0] if fp.height else None,
                "first_fp_name": names.get(fp["target"][0], "") if fp.height else None,
                "cutoff": cutoff,
            }
        )
        for r in real.filter(pl.col("target").is_in(list(truth_lineage))).iter_rows(
            named=True
        ):
            rank_rows.append(
                {
                    "method": method,
                    "setting_label": setting,
                    "target": r["target"],
                    "truth_lineage": truth_lineage[r["target"]],
                    "rank": r["rank"],
                    "score": r["score"],
                    "n_reported": real.height,
                    "cutoff": cutoff,
                }
            )
        for r in real.head(args.top).iter_rows(named=True):
            top_rows.append(
                {
                    "method": method,
                    "setting_label": setting,
                    "rank": r["rank"],
                    "target": r["target"],
                    "score": r["score"],
                    "truth_lineage": truth_lineage.get(r["target"], ""),
                    "protein_name": names.get(r["target"], ""),
                }
            )
        for qid, g in ranked.group_by("query_id"):
            qid = qid[0] if isinstance(qid, tuple) else qid
            kept = g if cutoff is None else g.filter(pl.col("score") > cutoff)
            # The query's own Swiss-Prot entry is left out for every query: a null draw
            # that matches the protein it was shuffled from has not reached another one.
            found = kept.filter(
                pl.col("target").is_in(list(truth_lineage))
                & (pl.col("target") != QUERY.split("__")[0])
            )
            by = dict(
                zip(
                    *(
                        found.with_columns(
                            lin=pl.col("target").replace_strict(truth_lineage)
                        )
                        .group_by("lin")
                        .len()
                        .to_dict(as_series=False)
                        .values()
                    )
                )
            )
            for lin in lineages:
                count_rows.append(
                    {
                        "method": method,
                        "setting_label": setting,
                        "query_id": qid,
                        "kind": kind_of[qid],
                        "target_lineage": lin,
                        "cutoff": cutoff,
                        "n_above_real_cutoff": int(by.get(lin, 0)),
                    }
                )
        # A query the list never reported found nothing; it still gets rows of 0.
        for qid in set(kind_of) - set(ranked["query_id"].unique()):
            for lin in lineages:
                count_rows.append(
                    {
                        "method": method,
                        "setting_label": setting,
                        "query_id": qid,
                        "kind": kind_of[qid],
                        "target_lineage": lin,
                        "cutoff": cutoff,
                        "n_above_real_cutoff": 0,
                    }
                )

    pl.DataFrame(top_rows).write_csv(args.outdir / "top_hits.tsv", separator="\t")
    pl.DataFrame(count_rows).write_csv(
        args.outdir / "truth_at_threshold.tsv", separator="\t"
    )
    # A scan's true rank over the whole reference, in place of the position among kept rows.
    scan = (
        pl.read_parquet(args.hits / "scans" / "scans.parquet")
        .filter(
            (pl.col("query_id") == QUERY)
            & pl.col("target").is_in(list(truth_lineage))
            & (pl.col("overlap").is_null() | (pl.col("overlap") >= 0))
        )
        .select(
            "method",
            pl.col("list_name").alias("setting_label"),
            "target",
            pl.col("rank").alias("scan_rank"),
            "n_ranked",
        )
        .unique()
    )
    rank_rows = (
        pl.DataFrame(rank_rows)
        .join(scan, on=["method", "setting_label", "target"], how="left")
        .with_columns(
            rank=pl.coalesce("scan_rank", "rank"),
            n_ranked=pl.coalesce("n_ranked", "n_reported"),
        )
        .drop("scan_rank")
    )
    rank_rows.write_csv(args.outdir / "truth_ranks.tsv", separator="\t")
    pl.DataFrame(fp_rows).write_csv(
        args.outdir / "first_false_positive.tsv", separator="\t"
    )
    print(
        f"{len(sources)} ranked lists; {len(top_rows)} top-hit rows; "
        f"{len(count_rows)} count rows ({len(kind_of)} queries x {len(lineages)} lineages per list)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
