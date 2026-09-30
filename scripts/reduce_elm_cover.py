#!/usr/bin/env python3
"""Score the ELM cover search (notebook 250) one arm at a time.

The search (`make run-elm-search`, conf/elm_cover.config) put the human proteins with a
usable ELM instance against whole target proteomes. This reads one arm's region table and
asks, for every human ELM instance: did a call cover it?

A call COVERS a motif when its human-side region contains at least 80% of the motif's
residues (--min-cover). Each query's hit list is first cut to its top --list-length target
proteins, ranked by each target's best call; ties go to the target accession that sorts
first, so the cut is the same on every run. kmerseek ranks by `region_mean_idf`; every
other tool by the score column its own output carries.

Placement null. A long call covers every motif inside it, so each covering call gets
p_place: the share of the positions a window of the call's length can take on the human
protein at which it would also cover the motif. It is a count, not a simulation:
a window [s, s+L) holds at least `need` residues of the motif [a, b) exactly when
L >= need, s >= a + need - L and s <= b - need, so the count is the length of that
interval clipped to [0, N - L]. A covering call "beats placement" when p_place < --alpha.

The placement null looks at the human side only, so it rejects exactly the long, correct
alignments of a whole ortholog: on the chicken dry run phmmer covered 532 of 550 motifs on
the ortholog, and 10 of them survived the null. The second check does not have that
problem. A covering call to the 1:1 ortholog is ON POSITION when its target-side region
holds at least --min-cover of the motif projected onto the ortholog through the Stage 0
MAFFT alignment (assets/elm_cover_projections.tsv). That asks whether the call put the
motif where the alignment puts it, which is what copying a motif label needs.

Coordinates: kmerseek regions and the ELM instances are 0-based and end-exclusive; the
comparator tables are 1-based inclusive, and their starts are shifted down by one here.

Foldseek and Reseek count positions along the AlphaFold model, not the QfO sequence. For 32
of the 1_303 queries the model's sequence differs; for some it is longer (Q9Y4K1: 1_723 aa
in QfO, 2_131 in the model), so every call on them sits in a different frame from the
motif. For these two tools a query is scored only when its model (--structures) has the
QfO length and at least 95% identical residues; its motifs are otherwise marked
`query_scoreable = false` and left out of that tool's denominators, not counted as missed.

A covering call that ends past its query's length is a malformed row (2 of 27.6 million
HHblits rows on the chicken run). It is dropped and counted; more than 1% of an arm's
covering calls doing so stops the arm, because that is a coordinate convention gone wrong.

Writes, per arm, into --out-dir:
  <arm>.instances.parquet  one row per ELM instance: the best rank of a covering call to
                           any target and to the query's 1:1 ortholog, each with and without
                           the placement filter, the best rank of an ortholog call on
                           position, and how many covering calls there were
  <arm>.summary.json       row counts, whether the arm had output at all, the Spearman
                           correlation of its ranking score with region length on a fixed
                           sample of calls
An arm whose region file is empty is recorded as status "empty" with no instances table,
so it reads as a missing arm, not as an arm that covered nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

KMERSEEK_RANK_BY = "region_mean_idf"
# Tools whose coordinates count along the AlphaFold model rather than the query sequence.
STRUCTURE_TOOLS = {"foldseek", "reseek"}
MIN_MODEL_IDENTITY = 0.95
MAX_PAST_END_SHARE = 0.01
THREE_TO_ONE = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G",
                    HIS="H", ILE="I", LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S",
                    THR="T", TRP="W", TYR="Y", VAL="V", SEC="U", PYL="O")
# Calls kept for the length check: a hash of the call's coordinates picks the same rows on
# every run, about one in LENGTH_SAMPLE_EVERY.
LENGTH_SAMPLE_EVERY = 1_000


def read_sequences(fasta: Path) -> dict[str, str]:
    """Query accession -> sequence, from a UniProt FASTA (sp|ACC|NAME ...)."""
    seqs, acc = {}, None
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                head = line[1:].split()[0]
                acc = head.split("|")[1] if "|" in head else head
                seqs[acc] = []
            elif acc is not None:
                seqs[acc].append(line.strip())
    return {a: "".join(v) for a, v in seqs.items()}


def lengths_frame(seqs: dict[str, str]) -> pl.DataFrame:
    return pl.DataFrame({"query_acc": list(seqs), "query_length": [len(v) for v in seqs.values()]},
                        schema={"query_acc": pl.String, "query_length": pl.Int64})


def model_sequence(cif: Path) -> str:
    """The residues an AlphaFold model's ATOM records carry, in label_seq_id order."""
    res = {}
    with open(cif) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                f = line.split()
                res[int(f[8])] = THREE_TO_ONE.get(f[5], "X")
    return "".join(res[i] for i in sorted(res))


def structure_status(seqs: dict[str, str], structures: Path) -> pl.DataFrame:
    """Per query: `same` when the model has the QfO length and >= MIN_MODEL_IDENTITY
    identical residues, else `differs`, or `no model` when there is no AF-<acc>-F1*.cif."""
    rows = []
    for acc, seq in seqs.items():
        cifs = sorted(structures.glob(f"AF-{acc}-F1*.cif"))
        if not cifs:
            rows.append((acc, "no model", None))
            continue
        m = model_sequence(cifs[0])
        same = len(m) == len(seq) and sum(a == b for a, b in zip(m, seq)) >= MIN_MODEL_IDENTITY * len(seq)
        rows.append((acc, "same" if same else "differs", len(m)))
    return pl.DataFrame(rows, schema={"query_acc": pl.String, "structure_status": pl.String,
                                      "model_length": pl.Int64}, orient="row")


def load_calls(path: Path, qfo_bin: Path) -> pl.LazyFrame | None:
    """One arm's calls as query_acc, target_acc, qstart, qend, tstart, tend (0-based,
    end-exclusive), score.

    Reads through the region benchmark's own loader, so accession parsing, the malformed-row
    guard and the compressed-CSV handling are the ones the Pfam scoring uses. No Bonferroni
    filter: this benchmark cuts lists by length instead.
    """
    sys.path.insert(0, str(qfo_bin))
    import evaluate_domain_calls as edc  # noqa: E402

    lf = edc.load_regions(path, direct=False, rank_by=KMERSEEK_RANK_BY, max_bonferroni_p=None)
    if lf is None:
        return None
    shift = 0 if path.suffix == ".parquet" else 1
    return lf.select("query_acc", "target_acc",
                     (pl.col("qstart") - shift).alias("qstart"), "qend",
                     (pl.col("tstart") - shift).alias("tstart"), "tend", "score")


def top_targets(calls: pl.LazyFrame, list_length: int) -> pl.DataFrame:
    """Each query's top `list_length` targets by best call, with their 1-based rank."""
    best = (calls.group_by("query_acc", "target_acc").agg(pl.col("score").max().alias("best"))
            .collect(engine="streaming"))
    return (best.sort(["query_acc", "best", "target_acc"], descending=[False, True, False])
            .with_columns(target_rank=pl.int_range(1, pl.len() + 1).over("query_acc"))
            .filter(pl.col("target_rank") <= list_length)
            .select("query_acc", "target_acc", "target_rank"))


def placement_p(qstart: pl.Expr, qend: pl.Expr, start: pl.Expr, end: pl.Expr,
                n: pl.Expr, need: pl.Expr) -> pl.Expr:
    """Share of window positions on a protein of length n at which a window of the call's
    length also holds `need` residues of the motif [start, end). See the module docstring."""
    length = qend - qstart
    lo = pl.max_horizontal(start + need - length, pl.lit(0))
    hi = pl.min_horizontal(end - need, n - length)
    count = pl.when(length >= need).then(pl.max_horizontal(hi - lo + 1, pl.lit(0))).otherwise(0)
    return count / (n - length + 1)


def covering_calls(calls: pl.LazyFrame, instances: pl.DataFrame, keep: pl.DataFrame,
                   lengths: pl.DataFrame, min_cover: float) -> pl.DataFrame:
    """Every call on the cut lists that covers an instance on its query, with p_place."""
    inst = instances.select(
        pl.col("accession").alias("query_acc"), "elm_instance",
        pl.col("start").alias("m_start"), pl.col("end").alias("m_end"),
        (pl.col("motif_length") * min_cover).ceil().cast(pl.Int64).alias("need"))
    overlap = (pl.min_horizontal("qend", "m_end") - pl.max_horizontal("qstart", "m_start"))
    return (calls.join(inst.lazy(), on="query_acc")
            .filter(overlap >= pl.col("need"))
            .join(keep.lazy(), on=["query_acc", "target_acc"])
            .collect(engine="streaming")
            .join(lengths, on="query_acc", how="left")
            .with_columns(p_place=placement_p(pl.col("qstart"), pl.col("qend"),
                                              pl.col("m_start"), pl.col("m_end"),
                                              pl.col("query_length"), pl.col("need"))))


def per_instance(cover: pl.DataFrame, instances: pl.DataFrame, orthologs: pl.DataFrame,
                 projections: pl.DataFrame, alpha: float, min_cover: float) -> pl.DataFrame:
    """One row per ELM instance, including the ones no call covered (ranks null)."""
    proj = projections.select(
        "elm_instance", pl.col("ortholog").alias("target_acc"), "proj_start", "proj_end",
        ((pl.col("proj_end") - pl.col("proj_start")) * min_cover).ceil().cast(pl.Int64)
        .alias("proj_need"))
    t_overlap = pl.min_horizontal("tend", "proj_end") - pl.max_horizontal("tstart", "proj_start")
    c = (cover.join(orthologs.select(pl.col("accession").alias("query_acc"),
                                     pl.col("ortholog").alias("target_acc"),
                                     pl.lit(True).alias("is_ortholog")),
                    on=["query_acc", "target_acc"], how="left")
         .join(proj, on=["elm_instance", "target_acc"], how="left")
         .with_columns(pl.col("is_ortholog").fill_null(False),
                       placed=pl.col("p_place") < alpha,
                       on_position=(t_overlap >= pl.col("proj_need")).fill_null(False)))
    agg = c.group_by("elm_instance").agg(
        n_covering_calls=pl.len(),
        n_covering_targets=pl.col("target_acc").n_unique(),
        best_rank=pl.col("target_rank").min(),
        best_rank_placed=pl.col("target_rank").filter("placed").min(),
        ortholog_rank=pl.col("target_rank").filter("is_ortholog").min(),
        ortholog_rank_placed=pl.col("target_rank").filter(pl.col("is_ortholog") & pl.col("placed")).min(),
        ortholog_rank_on_position=pl.col("target_rank").filter(pl.col("is_ortholog") & pl.col("on_position")).min(),
        min_p_place=pl.col("p_place").min(),
    )
    has_orth = orthologs.select(pl.col("accession"), has_ortholog=pl.lit(True)).unique()
    has_proj = projections.select("elm_instance", has_projection=pl.lit(True)).unique()
    return (instances.select("elm_instance", "elm_class", "accession", "start", "end",
                             "motif_length", "length_bin")
            .join(agg, on="elm_instance", how="left")
            .join(has_orth, on="accession", how="left")
            .join(has_proj, on="elm_instance", how="left")
            .with_columns(pl.col("n_covering_calls", "n_covering_targets").fill_null(0),
                          pl.col("has_ortholog", "has_projection").fill_null(False)))


def length_check(calls: pl.LazyFrame) -> dict:
    """Spearman correlation of the ranking score with call length, on a fixed sample."""
    s = (calls.filter(pl.struct("query_acc", "target_acc", "qstart", "qend").hash(seed=0)
                      % LENGTH_SAMPLE_EVERY == 0)
         .select("score", (pl.col("qend") - pl.col("qstart")).alias("length"))
         .collect(engine="streaming"))
    if s.height < 3:
        return {"n_sampled": s.height, "spearman_vs_length": None}
    rho = s.select(pl.corr("score", "length", method="spearman")).item()
    return {"n_sampled": s.height, "spearman_vs_length": rho}


def score_arm(path: Path, arm: str, args) -> dict:
    out = args.out_dir
    summary = {"arm": arm, "path": str(path), "min_cover": args.min_cover,
               "list_length": args.list_length, "alpha": args.alpha}
    calls = load_calls(path, args.qfo_bin)
    if calls is None:
        summary["status"] = "empty"
        (out / f"{arm}.summary.json").write_text(json.dumps(summary, indent=1))
        print(f"{arm}: empty region file, recorded as a missing arm")
        return summary

    tool = arm if not (path.suffix == ".parquet") else "kmerseek"
    instances = pl.read_csv(args.instances, separator="\t")
    orthologs = (pl.read_csv(args.orthologs, separator="\t")
                 .filter(pl.col("species") == args.species))
    projections = (pl.read_csv(args.projections, separator="\t")
                   .filter(pl.col("species") == args.species))
    seqs = read_sequences(args.query_fasta)
    lengths = lengths_frame(seqs)
    scoreable = lengths.select("query_acc", query_scoreable=pl.lit(True))
    if tool in STRUCTURE_TOOLS:
        if args.structures is None:
            raise SystemExit(f"{arm} counts positions along the AlphaFold model: pass --structures")
        st = structure_status(seqs, args.structures)
        scoreable = st.select("query_acc", query_scoreable=pl.col("structure_status") == "same")
        bad = scoreable.filter(~pl.col("query_scoreable"))["query_acc"].to_list()
        summary["structure_status"] = dict(st["structure_status"].value_counts().rows())
        calls = calls.filter(~pl.col("query_acc").is_in(bad))

    keep = top_targets(calls, args.list_length)
    cover = covering_calls(calls, instances, keep, lengths, args.min_cover)
    past_end = cover.filter(pl.col("qend") > pl.col("query_length"))
    summary["n_covering_calls_past_query_end"] = past_end.height
    if past_end.height > MAX_PAST_END_SHARE * max(cover.height, 1):
        raise SystemExit(f"{arm}: {past_end.height} of {cover.height} covering calls end past their "
                         f"query's length; the coordinate convention for this tool is wrong")
    if past_end.height:
        print(f"{arm}: dropped {past_end.height} covering calls that end past their query's length "
              f"(malformed rows), on {past_end['query_acc'].n_unique()} queries", file=sys.stderr)
        cover = cover.filter(pl.col("qend") <= pl.col("query_length"))
    if cover["query_length"].null_count():
        raise SystemExit(f"{arm}: {cover['query_length'].null_count()} covering calls on a query "
                         f"missing from {args.query_fasta}")
    table = per_instance(cover, instances, orthologs, projections, args.alpha, args.min_cover)
    table = table.join(scoreable.rename({"query_acc": "accession"}), on="accession", how="left").with_columns(
        pl.col("query_scoreable").fill_null(False))
    table.insert_column(0, pl.lit(arm).alias("arm"))
    table.write_parquet(out / f"{arm}.instances.parquet")

    summary.update(
        status="ok",
        n_queries_with_calls=keep["query_acc"].n_unique(),
        n_listed_pairs=keep.height,
        n_covering_calls=cover.height,
        n_instances=table.height,
        n_instances_scoreable=int(table["query_scoreable"].sum()),
        n_covered=int(table["best_rank"].is_not_null().sum()),
        n_covered_placed=int(table["best_rank_placed"].is_not_null().sum()),
        n_with_ortholog=int(table["has_ortholog"].sum()),
        n_ortholog_covered=int(table["ortholog_rank"].is_not_null().sum()),
        n_ortholog_covered_placed=int(table["ortholog_rank_placed"].is_not_null().sum()),
        n_with_projection=int(table["has_projection"].sum()),
        n_ortholog_on_position=int(table["ortholog_rank_on_position"].is_not_null().sum()),
        length_check=length_check(calls),
    )
    (out / f"{arm}.summary.json").write_text(json.dumps(summary, indent=1))
    import evaluate_domain_calls as edc  # noqa: E402  (on sys.path since load_calls)
    edc.release_inflated(path)
    print(f"{arm}: {summary['n_covered_placed']} of {summary['n_instances']} instances covered "
          f"past the placement null; {summary['n_ortholog_covered_placed']} of "
          f"{summary['n_with_ortholog']} with an ortholog covered on it; "
          f"{summary['n_ortholog_on_position']} of {summary['n_with_projection']} on position")
    return summary


def arm_files(results: Path, species: str, kmerseek_subdir: str = "kmerseek") -> list[tuple[str, Path]]:
    """(arm name, region file) for every arm of one target species, sorted by name."""
    arms = []
    for f in sorted((results / kmerseek_subdir).glob(f"human_vs_{species}.*.regions.parquet")):
        arms.append((f.name.removeprefix(f"human_vs_{species}.").removesuffix(".regions.parquet"), f))
    for d in sorted((results / "regions").iterdir()):
        for f in sorted(d.glob(f"human_vs_{species}.{d.name}.tsv.gz")):
            arms.append((d.name, f))
    return arms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="data/elm-cover/results")
    ap.add_argument("--species", default="chicken")
    ap.add_argument("--kmerseek-subdir", default="kmerseek",
                    help="kmerseek's store under --results: kmerseek_top<N> for a --kmerseek_top_targets run")
    ap.add_argument("--instances", type=Path, required=True, help="assets/elm_cover_instances.tsv")
    ap.add_argument("--orthologs", type=Path, required=True, help="assets/elm_cover_orthologs.tsv")
    ap.add_argument("--projections", type=Path, required=True, help="assets/elm_cover_projections.tsv")
    ap.add_argument("--query-fasta", type=Path, required=True,
                    help="data/elm-cover/qfo/Eukaryota/UP000005640_9606.fasta")
    ap.add_argument("--structures", type=Path, default=None,
                    help="the query AlphaFold models the structure tools searched (AF-<acc>-F1*.cif)")
    ap.add_argument("--qfo-bin", type=Path, required=True, help="the region benchmark's bin/")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-cover", type=float, default=0.8)
    ap.add_argument("--list-length", type=int, default=1_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--arm", action="append", help="score only these arms (repeatable)")
    ap.add_argument("--list-arms", action="store_true")
    args = ap.parse_args()

    arms = arm_files(args.results, args.species, args.kmerseek_subdir)
    if args.list_arms:
        for a, f in arms:
            print(a, f.stat().st_size, sep="\t")
        return
    if args.arm:
        arms = [(a, f) for a, f in arms if a in set(args.arm)]
    mine = arms[args.task::args.n_tasks]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"task {args.task} of {args.n_tasks}: {len(mine)} of {len(arms)} arms")
    for arm, f in mine:
        if (args.out_dir / f"{arm}.summary.json").exists():
            print(f"{arm}: already scored, skipped")
            continue
        score_arm(f, arm, args)


if __name__ == "__main__":
    main()
